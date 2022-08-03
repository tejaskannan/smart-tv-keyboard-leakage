from enum import Enum, auto


class SmartTVType(Enum):
    SAMSUNG = auto()
    APPLE_TV = auto()


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
DONE = '<DONE>'
CANCEL = '<CANCEL>'
BACKSPACE = '<BACK>'

SMALL_NUMBER = 1e-9
BIG_NUMBER = 1e9

