import re
import itertools
from bittrex.apis.bittrex_api import Interval
from bittrex.apis.bittrex_api import OPEN_LABEL, HIGH_LABEL, LOW_LABEL, \
    CLOSE_LABEL, VOLUME_LABEL, TIMESPAN_LABEL, BASE_VOLUME_LABEL
from bittrex.daos.bittrex_dao import BittrexDAO
from bittrex.invalid_ticks_exception import InvalidTicksException
from clara.daos.processed_data_dao import ProcessedDataDAO
from clara.daos.processed_data_dao import TICKS_LABEL
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

EMA_SIZE = 20
STATE_SIZE = 200
MAGNITUDE_CHANGE = 10**8


def update_preprocessed_data(raw_data_db_uri, marketsset_collection, intervals, preprocessed_data_db_uri):
    with BittrexDAO(raw_data_db_uri) as bittrex_dao, ProcessedDataDAO(preprocessed_data_db_uri) as processed_data_dao:
        markets = bittrex_dao.get_market_names(marketsset_collection)
        market_interval_pairs = itertools.product(markets, intervals)
        pool_args = [(market, interval, bittrex_dao, processed_data_dao) for market, interval in market_interval_pairs]
        print('{} seperate datasets to update'.format(len(pool_args)))
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = executor.map(lambda args: _get_preprocess_and_save_ticks(*args), pool_args)
            # check for exceptions
            for _ in futures:
                pass


def _get_preprocess_and_save_ticks(market_name, interval, bittrex_dao, processed_data_dao):
    try:
        interval_value = Interval[interval].value
    except KeyError:
        raise InvalidTicksException('Invalid ticks interval: {}'.format(interval))

    ticks_type = interval + re.sub(r'\W+', '', market_name)
    latest_processed_timespan = processed_data_dao.get_latest_state_timespan(ticks_type)
    print('last updated entry for {} for {} interval is from {}'
          .format(market_name, interval, str(latest_processed_timespan)))
    if latest_processed_timespan:
        earliest_timespan_needed = latest_processed_timespan - timedelta(minutes=interval_value*(STATE_SIZE + EMA_SIZE))
        raw_ticks = bittrex_dao.get_ticks(ticks_type, starting_from=earliest_timespan_needed)
    else:
        raw_ticks = bittrex_dao.get_ticks(ticks_type)

    raw_ticks = fill_empty_timespans(raw_ticks, interval_value)
    states = convert_ticks_to_states(raw_ticks)
    if not states:
        print('data for {} for {} is up to date'.format(market_name, interval))
        return

    processed_data_dao.save_states(states, ticks_type)
    print('{} for {} interval updated'.format(market_name, interval))


def fill_empty_timespans(ticks, interval):
    """
    :param ticks: list of market ticks
    :param interval: interval in minutes, if data is for 1 min interval it should be 1, if it's for daily interval
            it should be 1440
    :return: ticks with all the gaps filled with ticks that have all prices set to the same value as the last tick
            closing price and with volume set to 0
    """
    i = 0
    while i < (len(ticks) - 1):
        time_difference = (ticks[i + 1][TIMESPAN_LABEL] - ticks[i][TIMESPAN_LABEL]) / timedelta(minutes=1)
        if time_difference < interval:
            raise InvalidTicksException('One of the ticks has a wrong timespan or given interval is wrong. '
                                        'Tick occured sooner than the expected timespan '
                                        '(previous tick timespan + interval)')

        if time_difference > interval:
            filler_tick = {

                OPEN_LABEL: ticks[i][CLOSE_LABEL],
                HIGH_LABEL: ticks[i][CLOSE_LABEL],
                LOW_LABEL: ticks[i][CLOSE_LABEL],
                CLOSE_LABEL: ticks[i][CLOSE_LABEL],
                VOLUME_LABEL: 0,
                TIMESPAN_LABEL: ticks[i][TIMESPAN_LABEL] + timedelta(minutes=interval),
                BASE_VOLUME_LABEL: 0
            }
            ticks.insert(i + 1, filler_tick)

        i += 1

    return ticks


def convert_ticks_to_states(ticks):
    """
    Convert whole ticks to the list of states
    :param ticks: market ticks that will be used to generate states
    :return: list of states, each state consists of state_size state ticks that were related to its EMA and state
            timespan
    """
    # change magnitude to avoid vanishing values due to low altcoin prices
    ticks = _change_ticks_magnitude(ticks, MAGNITUDE_CHANGE)
    ticks_needed_for_one_state = STATE_SIZE + EMA_SIZE
    states = []
    for i in range(0, len(ticks) - ticks_needed_for_one_state):
        # take ticks to calculate ema, reverse them (step -1) so they are sorted in descending importance
        ema_ticks = ticks[(i+EMA_SIZE):i:-1]
        ema = calculate_ema(ema_ticks)  # TODO can you calculate ema iteratively so it runs faster?
        state_ticks = convert_ticks_to_state_ticks(ticks[i+EMA_SIZE:i+ticks_needed_for_one_state], ema)
        state = {
            TIMESPAN_LABEL: ticks[i+ticks_needed_for_one_state][TIMESPAN_LABEL],
            TICKS_LABEL: state_ticks
        }
        states.append(state)

    return states


def convert_ticks_to_state_ticks(ticks, ema):
    """
    Converts ticks to state ticks, i.e. relates them to EMA and drops not needed labels
    :param ticks: ticks to convert
    :param ema: exponential moving average to which the ticks are related
    :return: state ticks
    """
    state_ticks = []
    for t in ticks:
        state_ticks.append({
            OPEN_LABEL: 100 * (t[OPEN_LABEL] - ema) / ema,
            HIGH_LABEL: 100 * (t[HIGH_LABEL] - ema) / ema,
            LOW_LABEL: 100 * (t[LOW_LABEL] - ema) / ema,
            CLOSE_LABEL: 100 * (t[CLOSE_LABEL] - ema) / ema,
            VOLUME_LABEL: t[VOLUME_LABEL]
        })

    return state_ticks


def calculate_ema(ticks):
    """
    Calculates exponential moving average tick for given list of ticks
    :param ticks: Ticks for which the EMA is calculated, the higher the index, the less important the tick is
    :return: Exponential moving average based on closing prices of given ticks
    """
    ema = 0
    ticks_left = len(ticks)
    discount = 1
    for t in ticks:
        multiplier = (2.0 / (ticks_left + 1))
        ema += t[CLOSE_LABEL] * discount * multiplier
        discount *= (1 - multiplier)
        ticks_left -= 1

    return ema


def _change_ticks_magnitude(ticks, magnitude_change):
    ticks = ticks[:]
    for t in ticks:
        t[OPEN_LABEL] = t[OPEN_LABEL] * magnitude_change
        t[HIGH_LABEL] = t[HIGH_LABEL] * magnitude_change
        t[LOW_LABEL] = t[LOW_LABEL] * magnitude_change
        t[CLOSE_LABEL] = t[CLOSE_LABEL] * magnitude_change

    return ticks
