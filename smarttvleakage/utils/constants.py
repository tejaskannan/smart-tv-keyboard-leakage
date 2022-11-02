from enum import Enum, auto


class SmartTVType(Enum):
    SAMSUNG = auto()
    APPLE_TV = auto()
    UNKNOWN = auto()


class KeyboardType(Enum):
    SAMSUNG = auto()
    APPLE_TV_SEARCH = auto()
    APPLE_TV_PASSWORD = auto()


class Direction(Enum):
    ANY = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
DONE = '<DONE>'
CANCEL = '<CANCEL>'
BACKSPACE = '<BACK>'


START_CHAR = '<S>'
END_CHAR = '<E>'


SMALL_NUMBER = 1e-9
BIG_NUMBER = 1e9

