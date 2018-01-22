import datetime
import math
from random import random

import pytest
from clara.daos.processed_data_dao import TICKS_LABEL

from bittrex.apis.bittrex_api import OPEN_LABEL, HIGH_LABEL, LOW_LABEL, \
    CLOSE_LABEL, VOLUME_LABEL, TIMESPAN_LABEL, BASE_VOLUME_LABEL
from bittrex.invalid_ticks_exception import InvalidTicksException
from clara.preprocessing import data_preprocessor
from clara.preprocessing.data_preprocessor import STATE_SIZE, EMA_SIZE
from tests.preprocessing.mock_data import one_min_ticks_with_gaps, filled_one_min_ticks_with_gaps, \
    fifteen_min_ticks_with_gaps, filled_fifteen_min_ticks_with_gaps, invalid_two_min_ticks, same_ticks, \
    same_state_ticks, different_ticks, different_state_ticks


def test_fill_empty_timespans_one_min_ticks_with_gaps():
    assert data_preprocessor.fill_empty_timespans(one_min_ticks_with_gaps, 1) == filled_one_min_ticks_with_gaps


def test_fill_empty_timespans_one_min_ticks_without_gaps():
    assert data_preprocessor.fill_empty_timespans(filled_one_min_ticks_with_gaps, 1) == filled_one_min_ticks_with_gaps


def test_fill_empty_timespans_fifteen_min_ticks_with_gaps():
    assert data_preprocessor.fill_empty_timespans(fifteen_min_ticks_with_gaps, 15) \
           == filled_fifteen_min_ticks_with_gaps


def test_fill_empty_timespans_fifteen_min_ticks_without_gaps():
    assert data_preprocessor.fill_empty_timespans(filled_fifteen_min_ticks_with_gaps, 15) \
           == filled_fifteen_min_ticks_with_gaps


def test_fill_empty_timespans_invalid_two_min_ticks():
    with pytest.raises(InvalidTicksException) as excinfo:
        data_preprocessor.fill_empty_timespans(invalid_two_min_ticks, 2)

    assert 'wrong timespan' in str(excinfo.value)


def test_convert_ticks_to_states_random_ticks():
    random_ticks_length = 1000
    random_ticks = []
    for i in range(0, random_ticks_length):
        random_ticks.append({
            OPEN_LABEL: random(),
            HIGH_LABEL: random(),
            LOW_LABEL: random(),
            CLOSE_LABEL: random(),
            VOLUME_LABEL: random(),
            TIMESPAN_LABEL: datetime.datetime.fromtimestamp(random()),
            BASE_VOLUME_LABEL: random()
        })

    states = data_preprocessor.convert_ticks_to_states(random_ticks)
    assert len(states) == random_ticks_length - (EMA_SIZE + STATE_SIZE)
    for s in states:
        assert len(s[TICKS_LABEL]) == STATE_SIZE

    for i in range(0, len(states)):
        assert states[i][TIMESPAN_LABEL] == random_ticks[i + EMA_SIZE + STATE_SIZE][TIMESPAN_LABEL]


def test_convert_ticks_to_state_ticks_same_ticks():
    assert data_preprocessor.convert_ticks_to_state_ticks(same_ticks, 10) == same_state_ticks


def test_convert_ticks_to_state_ticks_different_ticks():
    assert data_preprocessor.convert_ticks_to_state_ticks(different_ticks, 10) == different_state_ticks


def test_calculate_ema_same_ticks():
    assert math.isclose(data_preprocessor.calculate_ema(same_ticks), same_ticks[0][CLOSE_LABEL])
