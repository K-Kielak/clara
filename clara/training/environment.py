import logging
from bittrex.apis.bittrex_api import Interval
from bittrex.apis.bittrex_api import CLOSE_LABEL
from bittrex.invalid_ticks_exception import InvalidTicksException
from clara.agent.position import Position
from clara.daos.processed_data_dao import ProcessedDataDAO
from clara.daos.processed_data_dao import TIMESPAN_LABEL, EMA_LABEL, TICKS_LABEL
from collections import deque
from datetime import timedelta


class Environment(object):
    TESTING_MINUTES = 5 * 1440  # How many of the last market minutes to use for testing (1 day = 1440 minutes)
    # How many states in the future to load from the database
    # (the higher value the more RAM is required, the lower the slower getting the data will be)
    MAX_BATCH_SIZE = 2000  # 50000 ~= 0.4GB

    def __init__(self, market_interval, db_uri, exchange_transaction_fee):
        self.dao = ProcessedDataDAO(db_uri)
        try:
            self.interval_value = Interval[market_interval].value
        except KeyError:
            raise InvalidTicksException('Invalid ticks interval: {}'.format(market_interval))
        self.tick_types = self.dao.get_all_tick_types_for_interval(market_interval)
        if not self.tick_types:
            raise ValueError('There are no states for given market_interval ({}) '
                             'in the database'.format(market_interval))

        self.current_market = self.tick_types[0]
        self.exchange_transaction_fee = exchange_transaction_fee
        self.loaded_market_data = {}
        self.current_agent_positions = {}
        self.coin_prices_at_last_entry = {}
        self.timespan_of_last_entry = {}
        self.is_test = False
        self.test_start_timespan = self._get_latest_timespan() - timedelta(minutes=Environment.TESTING_MINUTES)
        for market in self.tick_types:
            self.loaded_market_data[market] = deque()
            self.current_agent_positions[market] = Position.IDLE
            self._update_market_data_batch(market)
            self.coin_prices_at_last_entry[market] = self._get_current_price(market)
            self.timespan_of_last_entry[market] = None

        self.current_timespan = self._get_earliest_timespan()
        self._change_market()
        logging.info('Starting training from {}'.format(self.current_timespan))

        # data needed for logging
        self.successful_trades = 0
        self.failed_trades = 0
        self.successful_trades_length = 0
        self.failed_trades_length = 0
        self.total_profit = 0
        self.total_loss = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dao.close()

    def get_curr_state_vector(self):
        current_state = self.loaded_market_data[self.current_market][0]
        ticks = current_state[TICKS_LABEL]
        ema = current_state[EMA_LABEL]
        state_vector = [ema]
        for t in ticks:
            state_vector.extend(t.values())

        state_vector.extend(self.current_agent_positions[self.current_market].value)
        return state_vector, self.is_test

    def make_action(self, new_agent_position, trade_writer=None):
        current_agent_position = self.current_agent_positions[self.current_market]
        coin_price_at_last_entry = self.coin_prices_at_last_entry[self.current_market]

        # data needed for logging
        start_timespan = self.loaded_market_data[self.current_market][0][TIMESPAN_LABEL]
        end_timespan = self.loaded_market_data[self.current_market][1][TIMESPAN_LABEL]
        starting_position = self.current_agent_positions[self.current_market]

        reward = 0
        current_coin_price = self._get_current_price(self.current_market)
        if trade_writer:
            trade_writer.writerow([self.current_market, start_timespan, current_coin_price, new_agent_position])

        # process action ###
        # apply transaction fees and update coin_price_at_last_change)
        if current_agent_position.exits_trade(new_agent_position):
            coin_price_change_over_trade = current_coin_price - coin_price_at_last_entry
            percentage_change_over_trade = (100 * coin_price_change_over_trade) / coin_price_at_last_entry
            percentage_earned_over_trade = current_agent_position.get_multiplier() * percentage_change_over_trade
            total_percentage_owned = (100 - self.exchange_transaction_fee) * (100 + percentage_earned_over_trade) / 100
            total_fee = (total_percentage_owned / 100) * self.exchange_transaction_fee
            reward -= total_fee

            trade_profitability = total_percentage_owned - 100 - total_fee - self.exchange_transaction_fee
            trade_length = self.current_timespan - self.timespan_of_last_entry[self.current_market]
            if trade_profitability < 0:
                self.failed_trades += 1
                self.failed_trades_length += trade_length.seconds // 60
                self.total_loss -= trade_profitability
            else:
                self.successful_trades += 1
                self.successful_trades_length += trade_length.seconds // 60
                self.total_profit += trade_profitability

        if current_agent_position.enters_trade(new_agent_position):
            reward -= self.exchange_transaction_fee
            self.coin_prices_at_last_entry[self.current_market] = current_coin_price
            self.timespan_of_last_entry[self.current_market] = self.current_timespan

        # update agent and the market ###
        self.current_agent_positions[self.current_market] = new_agent_position
        self.loaded_market_data[self.current_market].popleft()
        if not self.is_test and self.current_timespan > self.test_start_timespan:
            logging.info('Starting tests at {}'.format(self.current_timespan))
            self.is_test = True

        previous_coin_price = current_coin_price
        current_coin_price = self._get_current_price(self.current_market)

        # process new state consequences ###
        coin_price_change = current_coin_price - previous_coin_price
        percentage_change = (100 * coin_price_change) / self.coin_prices_at_last_entry[self.current_market]
        # update reward according to market change and current agent position (SHORT, IDLE, or LONG)
        reward += self.current_agent_positions[self.current_market].get_multiplier() * percentage_change
        following_state_vector, _ = self.get_curr_state_vector()

        if len(self.loaded_market_data[self.current_market]) < 2:
            self._update_market_data_batch(self.current_market)

        self._change_market()

        if abs(reward) > 20:
            logging.warning('From {} to {}'.format(start_timespan, end_timespan))
            logging.warning('Unusual reward: {}'.format(reward))
            logging.warning('Starting from {} ending at {}'
                            .format(starting_position, self.current_agent_positions[self.current_market]))
            logging.warning('Change in price: {} to {}'.format(previous_coin_price, current_coin_price))
            logging.warning('Trade entered at {} with coin price: {}\n'
                            .format(self.timespan_of_last_entry, self.coin_prices_at_last_entry[self.current_market]))

        return reward, following_state_vector

    def _get_current_price(self, market):
        current_state = self.loaded_market_data[market][0]
        ema = current_state[EMA_LABEL]
        current_tick = current_state[TICKS_LABEL][-1]
        current_price = current_tick[CLOSE_LABEL]
        current_price = ema*(current_price + 100)
        return current_price

    def _update_market_data_batch(self, market):
        """
        Puts next batch of data from the states database in the loaded_market_data collection
        :param last_state: last state in the loaded_market_data collection
        """
        new_batch_start_time = None
        if self.loaded_market_data[market]:
            new_batch_start_time = self.loaded_market_data[market][0][TIMESPAN_LABEL] + \
                                   timedelta(minutes=self.interval_value)

        max_size = int(Environment.MAX_BATCH_SIZE / len(self.tick_types))
        states = self.dao.get_states(market, starting_from=new_batch_start_time, limit=max_size)

        self.loaded_market_data[market].extend(states)

    def _change_market(self):
        # check whether there is still market left with unprocessed data for the current timespan
        for name, data in self.loaded_market_data.items():
            if data[0][TIMESPAN_LABEL] == self.current_timespan and len(data) > 1:
                self.current_market = name
                return

        # increment current timespan and find first better market matching it
        self.current_timespan += timedelta(minutes=self.interval_value)
        for name, data in self.loaded_market_data.items():
            if data[0][TIMESPAN_LABEL] == self.current_timespan and len(data) > 1:
                self.current_market = name
                return

        # If there is no market matching timespan even after it's incrementation it means we reached end of the data
        # and we need to start from the beginning
        logging.info('Data finished at {}'.format(self.current_timespan))
        for market in self.tick_types:
            self.loaded_market_data[market] = deque()
            self._update_market_data_batch(market)

        self.current_timespan = self._get_earliest_timespan()
        self.is_test = False
        self.current_agent_positions = {market: Position.IDLE for market in self.tick_types}
        self._change_market()

        logging.info('Staring again from the beginning at {}'.format(self.current_timespan))

    def _get_earliest_timespan(self):
        return min([market[0][TIMESPAN_LABEL] for market in self.loaded_market_data.values()])

    def _get_latest_timespan(self):
        return max(self.dao.get_latest_state_timespan(tick_type) for tick_type in self.tick_types)

    def _get_test_start_timespan(self, market):
        latest_market_timespan = self.dao.get_latest_state_timespan(market)
        return latest_market_timespan - timedelta(minutes=Environment.TESTING_MINUTES)