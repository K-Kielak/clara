from bittrex.apis.bittrex_api import OPEN_LABEL, HIGH_LABEL, LOW_LABEL, \
    CLOSE_LABEL, VOLUME_LABEL, TIMESPAN_LABEL, BASE_VOLUME_LABEL
from datetime import datetime

one_min_ticks_with_gaps = [
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 00, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 30,
        LOW_LABEL: 18,
        CLOSE_LABEL: 28,
        VOLUME_LABEL: 50,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 4, 00),
        BASE_VOLUME_LABEL: 2
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 5, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 32,
        HIGH_LABEL: 60,
        LOW_LABEL: 30,
        CLOSE_LABEL: 55,
        VOLUME_LABEL: 75,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 6, 00),
        BASE_VOLUME_LABEL: 3
    },
    {
        OPEN_LABEL: 55,
        HIGH_LABEL: 70,
        LOW_LABEL: 40,
        CLOSE_LABEL: 40,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 8, 00),
        BASE_VOLUME_LABEL: 1
    }
]

filled_one_min_ticks_with_gaps = [
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 00, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 18,
        LOW_LABEL: 18,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 1, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 18,
        LOW_LABEL: 18,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 2, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 18,
        LOW_LABEL: 18,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 3, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 30,
        LOW_LABEL: 18,
        CLOSE_LABEL: 28,
        VOLUME_LABEL: 50,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 4, 00),
        BASE_VOLUME_LABEL: 2
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 5, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 32,
        HIGH_LABEL: 60,
        LOW_LABEL: 30,
        CLOSE_LABEL: 55,
        VOLUME_LABEL: 75,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 6, 00),
        BASE_VOLUME_LABEL: 3
    },
    {
        OPEN_LABEL: 55,
        HIGH_LABEL: 55,
        LOW_LABEL: 55,
        CLOSE_LABEL: 55,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 7, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 55,
        HIGH_LABEL: 70,
        LOW_LABEL: 40,
        CLOSE_LABEL: 40,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 8, 00),
        BASE_VOLUME_LABEL: 1
    }
]

fifteen_min_ticks_with_gaps = [
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 30,
        LOW_LABEL: 18,
        CLOSE_LABEL: 28,
        VOLUME_LABEL: 50,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 48, 00),
        BASE_VOLUME_LABEL: 2
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 32,
        HIGH_LABEL: 60,
        LOW_LABEL: 30,
        CLOSE_LABEL: 55,
        VOLUME_LABEL: 75,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 33, 00),
        BASE_VOLUME_LABEL: 3
    },
    {
        OPEN_LABEL: 55,
        HIGH_LABEL: 70,
        LOW_LABEL: 40,
        CLOSE_LABEL: 40,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 20, 3, 00),
        BASE_VOLUME_LABEL: 1
    }
]

filled_fifteen_min_ticks_with_gaps = [
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 18,
        LOW_LABEL: 18,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 18, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 18,
        LOW_LABEL: 18,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 33, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 30,
        LOW_LABEL: 18,
        CLOSE_LABEL: 28,
        VOLUME_LABEL: 50,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 48, 00),
        BASE_VOLUME_LABEL: 2
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 32,
        HIGH_LABEL: 32,
        LOW_LABEL: 32,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 18, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 32,
        HIGH_LABEL: 60,
        LOW_LABEL: 30,
        CLOSE_LABEL: 55,
        VOLUME_LABEL: 75,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 33, 00),
        BASE_VOLUME_LABEL: 3
    },
    {
        OPEN_LABEL: 55,
        HIGH_LABEL: 55,
        LOW_LABEL: 55,
        CLOSE_LABEL: 55,
        VOLUME_LABEL: 0,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 48, 00),
        BASE_VOLUME_LABEL: 0
    },
    {
        OPEN_LABEL: 55,
        HIGH_LABEL: 70,
        LOW_LABEL: 40,
        CLOSE_LABEL: 40,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 20, 3, 00),
        BASE_VOLUME_LABEL: 1
    }
]

invalid_two_min_ticks = [
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 30,
        LOW_LABEL: 18,
        CLOSE_LABEL: 28,
        VOLUME_LABEL: 50,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 9, 00),
        BASE_VOLUME_LABEL: 2
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 11, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 12, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 28,
        HIGH_LABEL: 32,
        LOW_LABEL: 22,
        CLOSE_LABEL: 32,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 19, 19, 00),
        BASE_VOLUME_LABEL: 1
    },
]

same_ticks = [
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 4, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 5, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 6, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 7, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 8, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 9, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 10, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 11, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 12, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 13, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 14, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 10,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 15, 00),
        BASE_VOLUME_LABEL: 1
    },
]

same_state_ticks = [
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 0.0,
        HIGH_LABEL: 100.0,
        LOW_LABEL: -50.0,
        CLOSE_LABEL: 80.0,
        VOLUME_LABEL: 23,
    },
]

different_ticks = [
    {
        OPEN_LABEL: 15,
        HIGH_LABEL: 20,
        LOW_LABEL: 5,
        CLOSE_LABEL: 18,
        VOLUME_LABEL: 23,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 3, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 18,
        HIGH_LABEL: 25,
        LOW_LABEL: 15,
        CLOSE_LABEL: 20,
        VOLUME_LABEL: 15,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 4, 00),
        BASE_VOLUME_LABEL: 1
    },
    {
        OPEN_LABEL: 20,
        HIGH_LABEL: 35,
        LOW_LABEL: 14,
        CLOSE_LABEL: 14,
        VOLUME_LABEL: 56,
        TIMESPAN_LABEL: datetime(2018, 1, 20, 18, 5, 00),
        BASE_VOLUME_LABEL: 1
    },
]

different_state_ticks = [
    {
        OPEN_LABEL: 50,
        HIGH_LABEL: 100,
        LOW_LABEL: -50,
        CLOSE_LABEL: 80,
        VOLUME_LABEL: 23,
    },
    {
        OPEN_LABEL: 80,
        HIGH_LABEL: 150,
        LOW_LABEL: 50,
        CLOSE_LABEL: 100,
        VOLUME_LABEL: 15,
    },
    {
        OPEN_LABEL: 100,
        HIGH_LABEL: 250,
        LOW_LABEL: 40,
        CLOSE_LABEL: 40,
        VOLUME_LABEL: 56,
    },
]