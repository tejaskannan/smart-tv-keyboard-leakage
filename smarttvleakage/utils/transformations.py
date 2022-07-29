from typing import Set, Dict, List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE
from smarttvleakage.graphs.keyboard_graph import APPLETV_ALPHABET, APPLETV_NUMBERS, APPLETV_SPECIAL
from smarttvleakage.dictionary import UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.dictionary import CHANGE, CAPS, BACKSPACE
from .constants import SmartTVType


def filter_and_normalize_scores(key_counts: Dict[str, int], candidate_keys: List[str]) -> Dict[str, float]:
    """
    Returns a dictionary of normalized scores for present keys.
    """
    filtered_scores = { key: float(key_counts[key]) for key in candidate_keys if key in key_counts }
    score_sum = sum(key_counts.values())
    return { key: (score / score_sum) for key, score in filtered_scores.items() }


def get_keyboard_mode(key: str, mode: str, tv_type: SmartTVType) -> str:
    """
    Fetches the keyboard mode based on the current key (change or not)
    """
    if key != CHANGE:
        return mode

    if tv_type == SmartTVType.SAMSUNG:
        if mode == SAMSUNG_STANDARD:
            return SAMSUNG_SPECIAL_ONE
        elif mode == SAMSUNG_SPECIAL_ONE:
            return SAMSUNG_STANDARD
        else:
            raise ValueError('Unknown mode {}'.format(mode))
    elif tv_type == SmartTVType.APPLE_TV:
        if mode == APPLETV_ALPHABET:
            return APPLETV_NUMBERS
        elif mode == APPLETV_NUMBERS:
            return APPLETV_SPECIAL
        elif mode == APPLETV_SPECIAL:
            return APPLETV_ALPHABET
        else:
            raise ValueError('Unknown mode {}'.format(mode))


def get_string_from_keys(keys: List[str]) -> str:
    """
    Returns the string produced by the given sequence of keys.
    """
    characters: List[str] = []

    caps_lock = False
    prev_turn_off_caps_lock = False

    for idx, key in enumerate(keys):
        if key == CAPS:
            if caps_lock:
                caps_lock = False
                prev_turn_off_caps_lock = True
            elif (idx > 0) and (keys[idx - 1] == CAPS):
                caps_lock = True
                prev_turn_off_caps_lock = False
        elif key == BACKSPACE:
            if len(characters) > 0:
                characters.pop()
        elif key not in UNPRINTED_CHARACTERS:
            character = CHARACTER_TRANSLATION.get(key, key)

            if caps_lock or ((idx > 0) and (keys[idx - 1] == CAPS) and (not prev_turn_off_caps_lock)):
                character = character.upper()

            characters.append(character)
            prev_turn_off_caps_lock = False

    return ''.join(characters)


def get_bit(val: int, bit_idx: int) -> int:
    return (val >> bit_idx) & 1


def capitalization_combinations(string: str) -> Set[str]:
    result: Set[str] = set()

    for mask in range(pow(2, len(string))):
        characters: List[str] = []

        for idx, character in enumerate(string):
            if get_bit(mask, bit_idx=idx) == 1:
                character = character.upper()
            else:
                character = character.lower()

            characters.append(character)

        transformed = ''.join(characters)
        result.add(transformed)

    return result
