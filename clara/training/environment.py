from bittrex.apis.bittrex_api import Interval
from bittrex.apis.bittrex_api import CLOSE_LABEL
from bittrex.invalid_ticks_exception import InvalidTicksException
from clara.agent.position import Position
from clara.daos.processed_data_dao import ProcessedDataDAO
from clara.daos.processed_data_dao import TIMESPAN_LABEL, EMA_LABEL, TICKS_LABEL
from collections import deque


class Environment(object):
    # How many states in the future to load from the database
    # (the higher value the more RAM is required, the lower the slower getting the data will be)
    MAX_BATCH_SIZE = 25000  # 50000 ~= 0.4GB

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
        self.current_tick_type_index = 0
        print('starting market is ' + self.tick_types[self.current_tick_type_index])

        self.exchange_transaction_fee = exchange_transaction_fee
        self.loaded_market_data = deque()
        self.current_agent_position = Position.IDLE

        self._update_market_data_batch()
        # reward is % of profit or loss based on starting_trade_price, updated each time agent enters the trade
        self.coin_price_at_last_entry = self._get_current_price()

        self.average_trade_profitability = 0
        self.trades_so_far = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dao.close()

    def get_curr_state_vector(self):
        current_state = self.loaded_market_data[0]
        ticks = current_state[TICKS_LABEL]
        state_vector = [current_state[EMA_LABEL]]
        for t in ticks:
            state_vector.extend(t.values())

        state_vector.extend(self.current_agent_position.value)
        return state_vector

    def make_action(self, new_agent_position):
        reward = 0

        # process action ###
        current_coin_price = self._get_current_price()
        # apply transaction fees and update coin_price_at_last_change)
        if self.current_agent_position.exits_trade(new_agent_position):
            coin_price_change_over_trade = current_coin_price - self.coin_price_at_last_entry
            percentage_change_over_trade = (100 * coin_price_change_over_trade) / self.coin_price_at_last_entry
            percentage_earned_over_trade = self.current_agent_position.get_multiplier() * percentage_change_over_trade
            total_percentage_owned = (100 - self.exchange_transaction_fee) * (100 + percentage_earned_over_trade) / 100
            total_fee = (total_percentage_owned / 100) * self.exchange_transaction_fee
            reward -= total_fee

            self.trades_so_far += 1
            trade_profitability = total_percentage_owned - 100 - total_fee - self.exchange_transaction_fee
            distance_from_average = trade_profitability - self.average_trade_profitability
            self.average_trade_profitability += distance_from_average / self.trades_so_far

        if self.current_agent_position.enters_trade(new_agent_position):
            reward -= self.exchange_transaction_fee
            self.coin_price_at_last_entry = current_coin_price

        # update agent and the market ###
        self.current_agent_position = new_agent_position
        self.loaded_market_data.popleft()
        previous_coin_price = current_coin_price
        current_coin_price = self._get_current_price()

        # process new state consequences ###
        coin_price_change = current_coin_price - previous_coin_price
        percentage_change = (100 * coin_price_change) / self.coin_price_at_last_entry
        # update reward according to market change and current agent position (SHORT, IDLE, or LONG)
        reward += self.current_agent_position.get_multiplier() * percentage_change
        following_state_vector = self.get_curr_state_vector()

        if len(self.loaded_market_data) == 1:
            self._update_market_data_batch(last_state=self.loaded_market_data[0])

        return reward, following_state_vector

    def _get_current_price(self,):
        current_state = self.loaded_market_data[0]
        ema = current_state[EMA_LABEL]
        current_tick = current_state[TICKS_LABEL][-1]
        current_price = current_tick[CLOSE_LABEL]
        current_price = ema*(current_price + 1)
        return current_price

    def _update_market_data_batch(self, last_state=None):
        """
        Puts next batch of data from the states database in the loaded_market_data collection
        :param last_state: last state in the loaded_market_data collection
        """
        new_batch_start_time = None
        if last_state:
            new_batch_start_time = last_state[TIMESPAN_LABEL]

        states = self.dao.get_states(self.tick_types[self.current_tick_type_index],
                                     starting_from=new_batch_start_time, limit=Environment.MAX_BATCH_SIZE)
        # get rid of timespans as Clara doesn't need the current time for trading
        # and change list to deque for efficiency reasons
        self.loaded_market_data = deque(states)

        if len(self.loaded_market_data) <= 1:
            self._increment_tick_type_index()
            print('market finished and changed to {}'.format(self.tick_types[self.current_tick_type_index]))
            self._update_market_data_batch()

    def _increment_tick_type_index(self):
        self.current_tick_type_index += 1
        if self.current_tick_type_index >= len(self.tick_types):
            self.current_tick_type_index = 0
