from enum import Enum, auto


class SmartTVType(Enum):
    SAMSUNG = auto()
    APPLE_TV = auto()


class KeyboardType(Enum):
    SAMSUNG = auto()
    APPLE_TV_SEARCH = auto()
    APPLE_TV_PASSWORD = auto()


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
DONE = '<DONE>'
CANCEL = '<CANCEL>'
BACKSPACE = '<BACK>'

SMALL_NUMBER = 1e-9
BIG_NUMBER = 1e9

