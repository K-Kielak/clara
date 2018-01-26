from enum import Enum


class Position(Enum):
    LONG = 1
    IDLE = 0
    SHORT = -1

    def exits_trade(self, new_position):
        return self != new_position and self != Position.IDLE

    def enters_trade(self, new_position):
        return self != new_position and new_position != Position.IDLE