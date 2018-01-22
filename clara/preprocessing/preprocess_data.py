import os

from clara.preprocessing.data_preprocessor import update_preprocessed_data

# Specify what data to preprocess
# Time intervals for which to preprocess data (look at Bittrex API for more information)
INTERVALS = ['oneMin']
# Name of a collection containing names of the markets for which to preprocess data
MARKETSSET_COLLECTION_NAME = 'preprocessMarkets'

BITTREX_DB_URI_ENV = 'BITTREX_DATA_DB_URI'
if BITTREX_DB_URI_ENV not in os.environ:
    raise EnvironmentError('Bittrex Data Database URI is not set under {}, '
                           'please set it before running the script again'.format(BITTREX_DB_URI_ENV))

STATES_DB_URI_ENV = 'STATES_DATA_DB_URI'
if STATES_DB_URI_ENV not in os.environ:
    raise EnvironmentError('States Data Database URI is not set under {}, '
                           'please set it before running the script again'.format(STATES_DB_URI_ENV))

if __name__ == '__main__':
    raw_data_db_uri = os.environ[BITTREX_DB_URI_ENV]
    preprocessed_data_db_uri = os.environ[STATES_DB_URI_ENV]
    update_preprocessed_data(raw_data_db_uri, MARKETSSET_COLLECTION_NAME, INTERVALS, preprocessed_data_db_uri)
