import pymongo
from bittrex.apis.bittrex_api import TIMESPAN_LABEL  # processed data uses the same labels for ticks as bittrex data
from pymongo import MongoClient
from pymongo.errors import AutoReconnect
from pymongo.errors import BulkWriteError
from retry import retry

EMA_LABEL = 'EMA'
TICKS_LABEL = 'Ticks'


class ProcessedDataDAO(object):
    def __init__(self, database_uri):
        host, db_name = database_uri.rsplit('/', 1)
        self.db_client = MongoClient(host)
        self.database = self.db_client[db_name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.db_client.close()

    def close(self):
        self.db_client.close()

    @retry(AutoReconnect, tries=5, delay=1, backoff=2)
    def get_states(self, ticks_type, starting_from=None, limit=None):
        """
        :param ticks_type: ticks type, should be in format:
                <bittrex-interval><base_coin_symbol><quote_coin_symbol>; i.e: oneMinBTCOMG
        :param starting_from: starting date given in the date object
        :param limit: returns the <limit> number of earliest states
        :return: states retrieved from the database
        """
        states_collection = self.database[ticks_type]
        if starting_from is None and limit is None:
            return list(states_collection.find({}, {'_id': False}))

        if limit is None:
            return list(states_collection.find({TIMESPAN_LABEL: {'$gte': starting_from}}, {'_id': False}))

        if starting_from is None:
            return list(states_collection.find({}, {'_id': False}).sort(TIMESPAN_LABEL, pymongo.ASCENDING).limit(limit))

        return list(states_collection.find({TIMESPAN_LABEL: {'$gte': starting_from}}, {'_id': False})
                    .sort(TIMESPAN_LABEL, pymongo.ASCENDING)
                    .limit(limit))

    @retry(AutoReconnect, tries=5, delay=1, backoff=2)
    def save_states(self, states, ticks_type):
        """
        Saves states to database
        :param states: ticks preprocessed to state format
        :param ticks_type: ticks type, should be in format:
                <bittrex-interval><base_coin_symbol><quote_coin_symbol>; i.e: oneMinBTCOMG
        """
        collection = self.database[ticks_type]
        # create index to guarantee timespans uniqueness in case when the collection doesn't exist yet
        collection.create_index(TIMESPAN_LABEL, unique=True)
        try:
            collection.insert_many(states, ordered=False)
        except BulkWriteError:
            # if there is even one duplicate in inserted data database throws BulkWriteError,
            # just ignore it, all non-duplicates were inserted successfully
            pass

    @retry(AutoReconnect, tries=5, delay=1, backoff=2)
    def get_latest_state_timespan(self, ticks_type):
        """
        :param ticks_type: ticks type, should be in format:
            <bittrex-interval><base_coin_symbol><quote_coin_symbol>; i.e: oneMinBTCOMG
        :return: timespan date object of the most recent state in the database by timespan
            or None if there are no states in the database
        """
        db_response = self.database[ticks_type]\
            .find({}, {TIMESPAN_LABEL: True, '_id': False})\
            .sort(TIMESPAN_LABEL, pymongo.DESCENDING)\
            .limit(1)

        if db_response.count(True) != 1:
            return None

        return db_response.next()[TIMESPAN_LABEL]

    @retry(AutoReconnect, tries=5, delay=1, backoff=2)
    def get_all_tick_types_for_interval(self, interval):
        collection_names = self.database.collection_names()
        return [name for name in collection_names if interval in name]
