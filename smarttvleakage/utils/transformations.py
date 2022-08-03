from typing import Set, Dict, List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE
from smarttvleakage.graphs.keyboard_graph import APPLETV_SEARCH_ALPHABET, APPLETV_SEARCH_NUMBERS, APPLETV_SEARCH_SPECIAL
from smarttvleakage.graphs.keyboard_graph import APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL
from smarttvleakage.dictionary import UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.dictionary import CHANGE, CAPS, BACKSPACE
from .constants import KeyboardType


def filter_and_normalize_scores(key_counts: Dict[str, int], candidate_keys: List[str]) -> Dict[str, float]:
    """
    Returns a dictionary of normalized scores for present keys.
    """
    filtered_scores = { key: float(key_counts[key]) for key in candidate_keys if key in key_counts }
    score_sum = sum(key_counts.values())
    return { key: (score / score_sum) for key, score in filtered_scores.items() }


def get_keyboard_mode(key: str, mode: str, keyboard_type: KeyboardType) -> str:
    """
    Fetches the keyboard mode based on the current key (change or not)
    """
    if key != CHANGE:
        return mode

    if keyboard_type == KeyboardType.SAMSUNG:
        keyboards = [SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE]
    elif keyboard_type == KeyboardType.APPLE_TV_SEARCH:
        keyboards = [APPLETV_SEARCH_ALPHABET, APPLETV_SEARCH_NUMBERS, APPLETV_SEARCH_SPECIAL]
    elif keyboard_type == KeyboardType.APPLE_TV_PASSWORD:
        keyboards = [APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL]
    else:
        raise ValueError('Unknown keyboard type: {}'.format(keyboard_type))

    idx = keyboards.index(mode)
    return keyboards[(idx + 1) % len(keyboards)]


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
