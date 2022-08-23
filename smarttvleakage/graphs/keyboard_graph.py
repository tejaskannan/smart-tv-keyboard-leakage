import os.path
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set, Union
import csv
from smarttvleakage.dictionary.dictionaries import SPACE, CHANGE, BACKSPACE
from smarttvleakage.keyboard_utils.graph_search import breadth_first_search
from smarttvleakage.utils.constants import KeyboardType, BIG_NUMBER
from smarttvleakage.utils.file_utils import read_json, read_json_gz
from smarttvleakage.audio.constants import SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.audio.move_extractor import Direction
from .keyboard_linker import KeyboardLinker, KeyboardPosition


SAMSUNG_STANDARD = 'samsung_standard'
SAMSUNG_SPECIAL_ONE = 'samsung_special_1'
SAMSUNG_CAPS = 'samsung_standard_caps'
APPLETV_SEARCH_ALPHABET = 'appletv_search_alphabet'
APPLETV_SEARCH_NUMBERS = 'appletv_search_numbers'
APPLETV_SEARCH_SPECIAL = 'appletv_search_special'
APPLETV_PASSWORD_STANDARD = 'appletv_password_standard'
APPLETV_PASSWORD_CAPS = 'appletv_password_caps'
APPLETV_PASSWORD_SPECIAL = 'appletv_password_special'


START_KEYS = {
    SAMSUNG_STANDARD: 'q',
    SAMSUNG_SPECIAL_ONE: CHANGE,
    SAMSUNG_CAPS: 'Q',
    APPLETV_SEARCH_ALPHABET: 't',
    APPLETV_SEARCH_NUMBERS: CHANGE,
    APPLETV_SEARCH_SPECIAL: CHANGE,
    APPLETV_PASSWORD_STANDARD: 'a',
    APPLETV_PASSWORD_CAPS: CHANGE,  # TODO: Fix This (should be <ABC>)
    APPLETV_PASSWORD_SPECIAL: CHANGE  # TODO: Fix This (should be <SPECIAL>)
}

#If change key is the same as the select key leave it empty and we will default to select key
CHANGE_KEYS = {
    SAMSUNG_STANDARD: SAMSUNG_SELECT,
    SAMSUNG_SPECIAL_ONE: SAMSUNG_SELECT,
}


SELECT_KEYS = {
    SAMSUNG_STANDARD: SAMSUNG_KEY_SELECT,
    SAMSUNG_SPECIAL_ONE: SAMSUNG_KEY_SELECT,
    APPLETV_PASSWORD_SPECIAL: APPLETV_KEYBOARD_SELECT,
    APPLETV_PASSWORD_CAPS: APPLETV_KEYBOARD_SELECT,
    APPLETV_PASSWORD_STANDARD: APPLETV_KEYBOARD_SELECT,
    APPLETV_SEARCH_ALPHABET: APPLETV_KEYBOARD_SELECT,
    APPLETV_SEARCH_NUMBERS: APPLETV_KEYBOARD_SELECT,
    APPLETV_SEARCH_SPECIAL: APPLETV_KEYBOARD_SELECT
}


def parse_graph_distances(path: str) -> Dict[str, DefaultDict[int, Set[str]]]:
    distance_matrix = read_json_gz(path)

    result: Dict[str, DefaultDict[int, Set[str]]] = dict()

    for start_key, neighbors_dict in distance_matrix.items():
        result[start_key] = defaultdict(set)

        for neighbor, distance in neighbors_dict.items():
            result[start_key][distance].add(neighbor)

    return result


class MultiKeyboardGraph:

    def __init__(self, keyboard_type: KeyboardType):
        self._keyboard_type = keyboard_type

        dir_name = os.path.dirname(__file__)
        linker_path: Optional[str] = None

        if keyboard_type == KeyboardType.SAMSUNG:
            standard_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard.json')
            special_one_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard_special_1.json')
            caps_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard_caps.json')
            self._start_mode = SAMSUNG_STANDARD

            self._keyboards = {
                SAMSUNG_STANDARD: SingleKeyboardGraph(path=standard_path, start_key=START_KEYS[SAMSUNG_STANDARD]),
                SAMSUNG_SPECIAL_ONE: SingleKeyboardGraph(path=special_one_path, start_key=START_KEYS[SAMSUNG_SPECIAL_ONE]),
                SAMSUNG_CAPS: SingleKeyboardGraph(path=caps_path, start_key=START_KEYS[SAMSUNG_CAPS])
            }

            linker_path = os.path.join(dir_name, 'samsung', 'link.json')
        elif keyboard_type == KeyboardType.APPLE_TV_SEARCH:
            alphabet_path = os.path.join(dir_name, 'apple_tv', 'alphabet.json')
            numbers_path = os.path.join(dir_name, 'apple_tv', 'numbers.json')
            special_path = os.path.join(dir_name, 'apple_tv', 'special.json')
            self._start_mode = APPLETV_SEARCH_ALPHABET

            self._keyboards = {
                APPLETV_SEARCH_ALPHABET: SingleKeyboardGraph(path=alphabet_path, start_key=START_KEYS[APPLETV_SEARCH_ALPHABET]),
                APPLETV_SEARCH_NUMBERS: SingleKeyboardGraph(path=numbers_path, start_key=START_KEYS[APPLETV_SEARCH_NUMBERS]),
                APPLETV_SEARCH_SPECIAL: SingleKeyboardGraph(path=special_path, start_key=START_KEYS[APPLETV_SEARCH_SPECIAL])
            }

            linker_path = os.path.join(dir_name, 'apple_tv', 'link.json')
        elif keyboard_type == KeyboardType.APPLE_TV_PASSWORD:
            standard_path = os.path.join(dir_name, 'apple_tv_password', 'standard.json')
            caps_path = os.path.join(dir_name, 'apple_tv_password', 'caps.json')
            special_path = os.path.join(dir_name, 'apple_tv_password', 'special.json')

            self._start_mode = APPLETV_PASSWORD_STANDARD

            self._keyboards = {
                APPLETV_PASSWORD_STANDARD: SingleKeyboardGraph(path=standard_path, start_key=START_KEYS[APPLETV_PASSWORD_STANDARD]),
                APPLETV_PASSWORD_CAPS: SingleKeyboardGraph(path=caps_path, start_key=START_KEYS[APPLETV_PASSWORD_CAPS]),
                APPLETV_PASSWORD_SPECIAL: SingleKeyboardGraph(path=special_path, start_key=START_KEYS[APPLETV_PASSWORD_SPECIAL])
            }

            linker_path = os.path.join(dir_name, 'apple_tv_password', 'link.json')
        else:
            raise ValueError('Unknown TV type: {}'.format(tv_type.name))

        # Make the keyboard linker
        self._linker = KeyboardLinker(linker_path)

    def get_start_keyboard_mode(self) -> str:
        return self._start_mode

    def get_keyboard_type(self) -> KeyboardType:
        return self._keyboard_type

    def is_unclickable(self, key: str, mode: str) -> bool:
        return self._keyboards[mode].is_unclickable(key)

    def get_linked_states(self, current_key: str, keyboard_mode: str) -> List[KeyboardPosition]:
        return self._linker.get_linked_states(current_key, keyboard_mode)

    def get_characters(self) -> List[str]:
        merged: Set[str] = set()

        for keyboard in self._keyboards.values():
            merged.update(keyboard.get_characters())

        return list(merged)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, mode: str, use_shortcuts: bool, use_wraparound: bool, directions: Union[Direction, List[Direction]]) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_from(start_key=start_key, num_moves=num_moves, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound, directions=directions)

    def get_moves_from_key(self, start_key: str, end_key: str, use_shortcuts: bool, use_wraparound: bool, mode: str) -> int:
        return self._keyboards[mode].get_moves_from_key(start_key, end_key, use_shortcuts, use_wraparound)

    def get_keyboards(self) -> List:
        return self._keyboards

    def get_nearest_link(self, current_key: str, mode: str, target_mode: str, use_shortcuts: bool, use_wraparound: bool) -> str:
        nearest_dist = BIG_NUMBER
        nearest_key = ''

        for i in self._keyboards[mode].get_characters():
            if (self._linker.get_linked_states(i, mode) != []) and (target_mode in [j[1] for j in self._linker.get_linked_states(i, mode)]):
                num_moves = self.get_moves_from_key(current_key, i, use_shortcuts, use_wraparound, mode)

                if (num_moves is not None) and (num_moves < nearest_dist):
                    nearest_dist = num_moves
                    nearest_key = i

        return nearest_key


class SingleKeyboardGraph:

    def __init__(self, path: str, start_key: str):
        # Set the start key
        self._start_key = start_key

        # Read in the graph config to get the unclickable keys
        graph_config = read_json(path)
        self._adj_list = graph_config['adjacency_list']
        self._wraparound_list = graph_config['wraparound']
        self._shortcuts_list = graph_config['shortcuts']
        self._characters = list(self._adj_list.keys())
        self._unclickable_keys: Set[str] = set(graph_config['unclickable'])

        # Read in the precomputed shortest paths maps
        self._no_wraparound_distances: Dict[str, DefaultDict[int, Set[str]]] = parse_graph_distances(path=path.replace('.json', '_normal.json.gz'))
        self._no_wraparound_distances_shortcuts: Dict[str, DefaultDict[int, Set[str]]] = parse_graph_distances(path=path.replace('.json', '_shortcuts.json.gz'))

        wraparound_path = path.replace('.json', '_wraparound.json.gz')
        self._wraparound_distances: Dict[str, DefaultDict[int, Set[str]]] = parse_graph_distances(path=wraparound_path)
        if os.path.exists(wraparound_path):
            self._wraparound_distances = parse_graph_distances(path=wraparound_path)

        wraparound_shortcuts_path = path.replace('.json', '_all.json.gz')
        self._wraparound_distances_shortcuts: Dict[str, DefaultDict[int, Set[str]]] = dict()
        if os.path.exists(wraparound_shortcuts_path):
            self._wraparound_distances_shortcuts = parse_graph_distances(path=wraparound_shortcuts_path)

    def get_characters(self) -> List[str]:
        return self._characters

    def is_unclickable(self, key: str) -> bool:
        return (key in self._unclickable_keys)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, use_shortcuts: bool, use_wraparound: bool, directions: Union[Direction, List[Direction]]) -> List[str]:
        assert num_moves >= 0, 'Must provide a non-negative number of moves. Got {}'.format(num_moves)

        if num_moves == 0:
            return [start_key]

        if isinstance(directions, list) and isinstance(self._adj_list[start_key], dict):
            assert len(directions) == num_moves, 'Must provide the same number of directionsi ({}) as moves ({})'.format(len(directions), num_moves)

            result_set: Set[str] = set()

            standard = breadth_first_search(start_key=start_key,
                                            distance=num_moves,
                                            adj_list=self._adj_list,
                                            wraparound=None,
                                            shortcuts=None,
                                            directions=directions)
            result_set.update(standard)

            if use_wraparound:
                wraparound = breadth_first_search(start_key=start_key,
                                                  distance=num_moves,
                                                  adj_list=self._adj_list,
                                                  wraparound=self._wraparound_list,
                                                  shortcuts=None,
                                                  directions=directions)
                result_set.update(wraparound)

            if use_shortcuts:
                shortcuts = breadth_first_search(start_key=start_key,
                                                 distance=num_moves,
                                                 adj_list=self._adj_list,
                                                 wraparound=self._wraparound_list,
                                                 shortcuts=self._shortcuts_list,
                                                 directions=directions)
                result_set.update(shortcuts)

            return list(sorted(result_set))
        else:
            no_wraparound_neighbors = self._no_wraparound_distances.get(start_key, dict()).get(num_moves, set())
            wraparound_neighbors = self._wraparound_distances.get(start_key, dict()).get(num_moves, set())

            if use_shortcuts:
                no_wraparound_neighbors.update(self._no_wraparound_distances_shortcuts.get(start_key, dict()).get(num_moves, set()))
                wraparound_neighbors.update(self._wraparound_distances_shortcuts.get(start_key, dict()).get(num_moves, set()))

            if (len(no_wraparound_neighbors) == 0) and (len(wraparound_neighbors) == 0):
                return []

            if use_wraparound:
                combined = no_wraparound_neighbors.union(wraparound_neighbors)
                return list(sorted(combined))
            else:
                return list(sorted(no_wraparound_neighbors))

    def get_moves_from_key(self, start_key: str, end_key: str, use_shortcuts: bool, use_wraparound: bool) -> int:
        if end_key not in list(self._no_wraparound_distances.keys()):
            return -1
        if end_key == start_key:
            return 0
        if use_shortcuts:
            if use_wraparound:
                for i in self._wraparound_distances_shortcuts[start_key]:
                    if end_key in self._wraparound_distances_shortcuts[start_key][i]:
                        return i
            else:
                for i in self._no_wraparound_distances_shortcuts[start_key]:
                    if end_key in self._no_wraparound_distances_shortcuts[start_key][i]:
                        return i
        else:
            if use_wraparound:
                for i in self._wraparound_distances[start_key]:
                    if end_key in self._wraparound_distances[start_key][i]:
                        return i
            else:
                for i in self._no_wraparound_distances[start_key]:
                    if end_key in self._no_wraparound_distances[start_key][i]:
                        return i
