from bittrex.apis.bittrex_api import Interval
from bittrex.invalid_ticks_exception import InvalidTicksException
from clara.daos.processed_data_dao import ProcessedDataDAO
from clara.daos.processed_data_dao import TICKS_LABEL
from datetime import timedelta


class Environment(object):
    def __init__(self, market_interval, db_uri):
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
        self.recent_batch_end_timespan = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dao.close()

    def get_next_batch(self, max_size):
        """
        Gets next batch of data
        :param max_size: maximum size of the batch, if it's last batch before the end of the market batch may be smaller
        :return: batch of states from the market and
                True if that was the last batch in the current market, False otherwise
        """
        states = self.dao.get_states(self.tick_types[self.current_tick_type_index],
                                     starting_from=self.recent_batch_end_timespan,
                                     limit=max_size)
        # get rid of timespans as Clara doesn't need the current time for trading
        states = [state[TICKS_LABEL] for state in states]

        if len(states) < max_size:  # if result is not full it means that the data for the market has ended
            self._increment_tick_type_index()
            self.recent_batch_end_timespan = None
            return states, True

        self.recent_batch_end_timespan += timedelta(minutes=max_size*self.interval_value)
        return states, False

    def _increment_tick_type_index(self):
        self.current_tick_type_index += 1
        if self.current_tick_type_index >= len(self.tick_types):
            self.current_tick_type_index = 0

