from enum import Enum


class Position(Enum):
    LONG = [1, 0, 0]
    IDLE = [0, 1, 0]
    SHORT = [0, 0, 1]

    def get_multiplier(self):
        if self == Position.LONG:
            return 1

        if self == Position.SHORT:
            return -1

        return 0

    def exits_trade(self, new_position):
        return self != new_position and self != Position.IDLE

    def enters_trade(self, new_position):
        return self != new_position and new_position != Position.IDLE