from enum import Enum, auto


class SmartTVType(Enum):
    SAMSUNG = auto()
    APPLE_TV = auto()
    UNKNOWN = auto()


class KeyboardType(Enum):
    SAMSUNG = auto()
    APPLE_TV_SEARCH = auto()
    APPLE_TV_PASSWORD = auto()
    ABC = auto()


class Direction(Enum):
    ANY = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()


class SuggestionsType(Enum):
    SUGGESTIONS = auto()
    STANDARD = auto()


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
DONE = '<DONE>'
CANCEL = '<CANCEL>'
BACKSPACE = '<BACK>'


START_CHAR = '<S>'
END_CHAR = '<E>'


SUGGESTIONS_CUTOFF = 0.6  # Gives over 99.5% accuracy on passwords generated using optimal move sequences


SMALL_NUMBER = 1e-9
BIG_NUMBER = 1e9
