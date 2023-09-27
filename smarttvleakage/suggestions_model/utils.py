import random
import io
import string
from argparse import ArgumentParser
from typing import List, Dict, Tuple

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.dictionary.english_dictionary import SQLEnglishDictionary
from smarttvleakage.utils.file_utils import read_json


def get_suggestions(english_dictionary: SQLEnglishDictionary,
                    single_suggestions: Dict[str, List[str]],
                    prefix: str,
                    count: int) -> List[str]:
    """
    Returns simulated autocomplete suggestions using an EnglishDictionary
    """
    if len(prefix) == 1:  # Use single suggestions (hard-coded)
        return single_suggestions[prefix.lower()]

    char_dict = english_dictionary.get_letter_counts(prefix)

    char_list = []
    for c in char_dict:
        char_list.append((c, char_dict[c]))
    char_list.sort(key=(lambda x: x[1]), reverse=True)

    suggestions = []
    for idx in range(min(len(char_list), count)):
        suggestions.append(char_list[idx][0])

    return suggestions


def find_path_suggestions(word: str,
                          english_dictionary: SQLEnglishDictionary,
                          keyboard: MultiKeyboardGraph,
                          single_suggestions: Dict[str, List[str]]) -> List[int]:
    """
    Simulates the path for a word using autocomplete
    """
    path = []
    mode = keyboard.get_start_keyboard_mode()
    prev = START_KEYS[mode]

    letters = word.lower()
    did_last_use_suggestions = False

    for character in letters:

        if len(path) == 0:
            prefix = ''
        else:
            prefix = word[0:len(path)]

        suggestions = get_suggestions(english_dictionary, single_suggestions, prefix, count=4)

        if (len(path) > 0) and (len(suggestions) > 0) and (character == suggestions[0]):
            if did_last_use_suggestions:
                path.append(0)
            else:
                path.append(1)
            did_last_use_suggestions = True
        elif (len(path) > 0) and (character in suggestions):
            path.append(1)  # Assume the user has one move to make to go to the suggested key
            did_last_use_suggestions = True
        else:
            distance = keyboard.get_moves_from_key(prev, character, False, False, mode)

            # Potentially change keyboards if handling characters in other views
            while distance == -1:
                path.append(keyboard.get_moves_from_key(prev, '<CHANGE>', False, False, mode))
                prev = '<CHANGE>'
                mode = get_keyboard_mode(prev, mode, keyboard_type=KeyboardType.SAMSUNG)
                prev = START_KEYS[mode]
                distance = keyboard.get_moves_from_key(prev, character, False, False, mode)

            if did_last_use_suggestions and len(path) > 0:
                distance += 1

            path.append(distance)
            prev = character
            did_last_use_suggestions = False

    return path


def add_mistakes(ms: List[int], count: int) -> List[int]:
    """
    Randomly adds up to count suboptimal movements to the given move sequence count
    """
    if len(ms) == 0:
        return ms

    # Randomly adds 2 moves to places in the sequence
    for _ in range(count):
        place = random.randint(0, len(ms) - 1)
        ms[place] += 2

    return ms


def simulate_move_sequence(word: str,
                           single_suggestions: Dict[str, List[str]],
                           use_suggestions: bool,
                           english_dictionary: SQLEnglishDictionary,
                           keyboard: MultiKeyboardGraph,
                           num_mistakes: int):
    """
    Simulates a move sequence
    """
    if not use_suggestions:
        move_seq = findPath(word, False, False, False, 0, 0, 0, keyboard, 'q')
        move_counts = list(map(lambda m: m.num_moves, move_seq))
    else:
        move_counts = find_path_suggestions(english_dictionary=english_dictionary,
                                            single_suggestions=single_suggestions,
                                            word=word,
                                            keyboard=keyboard)

    add_mistakes(move_counts, num_mistakes)
    return move_counts


def read_passwords(path: str, count: int) -> List[str]:
    """
    Reads passwords from the given path, filtering
    out invalid characters
    """
    results: List[str] = []

    with open(path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all(should_keep_password(c) for c in word):
                cleaned = ''.join([c for c in word if c not in ('_', '`', ' ', '\n', ';')])
                results.append(cleaned)

            if len(results) >= count:
                break

    return results


def read_english_words(path: str, count: int) -> List[str]:
    """
    Reads english words from the given path
    """
    results: List[str] = []

    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            word = tokens[0]

            if all(should_keep_english(c) for c in word):
                results.append(word)

            if len(results) >= count:
                break

    return results


def should_keep_password(character: str) -> bool:
    return (character in string.printable)


def should_keep_english(character: str) -> bool:
    return (character in string.ascii_letters)
