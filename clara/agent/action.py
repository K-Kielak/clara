from enum import Enum


class Action(Enum):
    LONG = [1, 0, 0]
    IDLE = [0, 1, 0]
    SHORT = [0, 0, 1]
