import os.path
import json
import csv
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set, Union, Optional
from smarttvleakage.dictionary.dictionaries import SPACE, CHANGE, BACKSPACE, NEXT
from smarttvleakage.keyboard_utils.graph_search import breadth_first_search, follow_path, bfs
from smarttvleakage.utils.constants import KeyboardType, BIG_NUMBER
from smarttvleakage.utils.file_utils import read_json, read_json_gz
from smarttvleakage.audio.sounds import SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.audio.move_extractor import Direction
from .keyboard_linker import KeyboardLinker, KeyboardPosition


SAMSUNG_STANDARD = 'samsung_standard'
SAMSUNG_SPECIAL_ONE = 'samsung_special_1'
SAMSUNG_SPECIAL_TWO = 'samsung_special_2'
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
    SAMSUNG_SPECIAL_TWO: NEXT,
    SAMSUNG_CAPS: 'Q',
    APPLETV_SEARCH_ALPHABET: 't',
    APPLETV_SEARCH_NUMBERS: CHANGE,
    APPLETV_SEARCH_SPECIAL: CHANGE,
    APPLETV_PASSWORD_STANDARD: 'a',
    APPLETV_PASSWORD_CAPS: CHANGE,  # TODO: Fix This (should be <ABC>)
    APPLETV_PASSWORD_SPECIAL: CHANGE  # TODO: Fix This (should be <SPECIAL>)
}


# If change key is the same as the select key leave it empty and we will default to select key
CHANGE_KEYS = {
    SAMSUNG_STANDARD: SAMSUNG_SELECT,
    SAMSUNG_SPECIAL_ONE: SAMSUNG_SELECT,
    SAMSUNG_SPECIAL_TWO: SAMSUNG_SELECT
}


SELECT_KEYS = {
    SAMSUNG_STANDARD: SAMSUNG_KEY_SELECT,
    SAMSUNG_SPECIAL_ONE: SAMSUNG_KEY_SELECT,
    SAMSUNG_SPECIAL_TWO: SAMSUNG_KEY_SELECT,
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
            special_two_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard_special_2.json')
            caps_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard_caps.json')
            self._start_mode = SAMSUNG_STANDARD

            self._keyboards = {
                SAMSUNG_STANDARD: SingleKeyboardGraph(path=standard_path),
                SAMSUNG_SPECIAL_ONE: SingleKeyboardGraph(path=special_one_path),
                SAMSUNG_SPECIAL_TWO: SingleKeyboardGraph(path=special_two_path),
                SAMSUNG_CAPS: SingleKeyboardGraph(path=caps_path)
            }

            linker_path = os.path.join(dir_name, 'samsung', 'link.json')
        elif keyboard_type == KeyboardType.APPLE_TV_SEARCH:
            alphabet_path = os.path.join(dir_name, 'apple_tv', 'alphabet.json')
            numbers_path = os.path.join(dir_name, 'apple_tv', 'numbers.json')
            special_path = os.path.join(dir_name, 'apple_tv', 'special.json')
            self._start_mode = APPLETV_SEARCH_ALPHABET

            self._keyboards = {
                APPLETV_SEARCH_ALPHABET: SingleKeyboardGraph(path=alphabet_path),
                APPLETV_SEARCH_NUMBERS: SingleKeyboardGraph(path=numbers_path),
                APPLETV_SEARCH_SPECIAL: SingleKeyboardGraph(path=special_path)
            }

            linker_path = os.path.join(dir_name, 'apple_tv', 'link.json')
        elif keyboard_type == KeyboardType.APPLE_TV_PASSWORD:
            standard_path = os.path.join(dir_name, 'apple_tv_password', 'standard.json')
            caps_path = os.path.join(dir_name, 'apple_tv_password', 'caps.json')
            special_path = os.path.join(dir_name, 'apple_tv_password', 'special.json')

            self._start_mode = APPLETV_PASSWORD_STANDARD

            self._keyboards = {
                APPLETV_PASSWORD_STANDARD: SingleKeyboardGraph(path=standard_path),
                APPLETV_PASSWORD_CAPS: SingleKeyboardGraph(path=caps_path),
                APPLETV_PASSWORD_SPECIAL: SingleKeyboardGraph(path=special_path)
            }

            linker_path = os.path.join(dir_name, 'apple_tv_password', 'link.json')
        else:
            raise ValueError('Unknown TV type: {}'.format(keyboard_type.name))

        # Make the keyboard linker
        self._linker = KeyboardLinker(linker_path)

    @property
    def keyboard_type(self) -> KeyboardType:
        return self._keyboard_type

    def get_start_keyboard_mode(self) -> str:
        return self._start_mode

    def get_keyboard_type(self) -> KeyboardType:
        return self._keyboard_type

    def is_unclickable(self, key: str, mode: str) -> bool:
        return self._keyboards[mode].is_unclickable(key)

    def get_linked_states(self, current_key: str, keyboard_mode: str) -> List[KeyboardPosition]:
        return self._linker.get_linked_states(current_key, keyboard_mode)

    def get_linked_key_to(self, current_key: str, current_mode: str, target_mode: str) -> Optional[str]:
        return self._linker.get_linked_key_to(current_key=current_key,
                                              current_mode=current_mode,
                                              target_mode=target_mode)

    def get_characters(self) -> List[str]:
        merged: Set[str] = set()

        for keyboard in self._keyboards.values():
            merged.update(keyboard.get_characters())

        return list(merged)

    def get_start_character_set(self) -> List[str]:
        return self.get_keyboard_characters(keyboard_mode=self.get_start_keyboard_mode())

    def get_keyboard_characters(self, keyboard_mode: str) -> List[str]:
        return self._keyboards[keyboard_mode].get_characters()

    def get_keyboard_with_character(self, character: str) -> Optional[str]:
        for keyboard_mode in self._keyboards.keys():
            if character in self.get_keyboard_characters(keyboard_mode):
                return keyboard_mode

        return None

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, mode: str, use_shortcuts: bool, use_wraparound: bool, directions: Union[Direction, List[Direction]]) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_from(start_key=start_key, num_moves=num_moves, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound, directions=directions)

    def get_keys_for_moves_to(self, end_key: str, num_moves: int, mode: str, use_shortcuts: bool, use_wraparound: bool) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_to(end_key=end_key, num_moves=num_moves, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound)

    def follow_path(self, start_key: str, use_shortcuts: bool, use_wraparound: bool, directions: List[Direction], mode: str) -> Optional[str]:
        return self._keyboards[mode].follow_path(start_key=start_key, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound, directions=directions)

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

    def get_adjacency_list(self, mode, use_shortcuts, use_wraparound):
        return self._keyboards[mode].get_adjacency_list(use_shortcuts, use_wraparound)


class SingleKeyboardGraph:

    def __init__(self, path: str):
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

    def follow_path(self, start_key: str, use_shortcuts: bool, use_wraparound: bool, directions: List[Direction]) -> Optional[str]:
        if use_shortcuts and use_wraparound:
            return follow_path(start_key=start_key,
                               adj_list=self._adj_list,
                               shortcuts=self._shortcuts_list,
                               wraparound=self._wraparound_list,
                               directions=directions)
        elif use_shortcuts:
            return follow_path(start_key=start_key,
                               adj_list=self._adj_list,
                               shortcuts=self._shortcuts_list,
                               wraparound=None,
                               directions=directions)
        elif use_wraparound:
            return follow_path(start_key=start_key,
                               adj_list=self._adj_list,
                               shortcuts=None,
                               wraparound=self._wraparound_list,
                               directions=directions)
        else:
            return follow_path(start_key=start_key,
                               adj_list=self._adj_list,
                               shortcuts=None,
                               wraparound=None,
                               directions=directions)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, use_shortcuts: bool, use_wraparound: bool, directions: Union[Direction, List[Direction]]) -> List[str]:
        assert num_moves >= 0, 'Must provide a non-negative number of moves. Got {}'.format(num_moves)

        if num_moves == 0:
            return [start_key]

        result_set: Set[str] = set()

        if isinstance(directions, list) and isinstance(self._adj_list[start_key], dict):
            assert len(directions) == num_moves, 'Must provide the same number of directions ({}) as moves ({})'.format(len(directions), num_moves)

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
        else:
            no_wraparound_neighbors = self._no_wraparound_distances.get(start_key, dict()).get(num_moves, set())
            result_set.update(no_wraparound_neighbors)

            if use_wraparound:
                wraparound_neighbors = self._wraparound_distances.get(start_key, dict()).get(num_moves, set())
                result_set.update(wraparound_neighbors)

            if use_shortcuts:
                result_set.update(self._no_wraparound_distances_shortcuts.get(start_key, dict()).get(num_moves, set()))

                if use_wraparound:
                    result_set.update(self._wraparound_distances_shortcuts.get(start_key, dict()).get(num_moves, set()))

        return list(sorted(result_set))

    def get_keys_for_moves_to(self, end_key: str, num_moves: int, use_shortcuts: bool, use_wraparound: bool) -> List[str]:
        result: List[str] = []
        for key in self.get_characters():
            if end_key in self._no_wraparound_distances.get(key, dict()).get(num_moves, set()):
                result.append(key)
            elif use_shortcuts and (end_key in self._no_wraparound_distances_shortcuts.get(key, dict()).get(num_moves, set())):
                result.append(key)
            elif use_wraparound and (end_key in self._wraparound_distances.get(key, dict()).get(num_moves, set())):
                result.append(key)
            elif (use_wraparound and use_shortcuts) and (end_key in self._wraparound_distances_shortcuts.get(key, dict()).get(num_moves, set())):
                result.append(key)

        return result

    def get_moves_from_key(self, start_key: str, end_key: str, use_shortcuts: bool, use_wraparound: bool) -> int:
        if end_key not in self.get_characters():
            return -1
        elif end_key == start_key:
            return 0
        elif use_shortcuts:
            if use_wraparound:
                for dist in self._wraparound_distances_shortcuts[start_key].keys():
                    if end_key in self._wraparound_distances_shortcuts[start_key][dist]:
                        return dist
            else:
                for dist in self._no_wraparound_distances_shortcuts[start_key].keys():
                    if end_key in self._no_wraparound_distances_shortcuts[start_key][dist]:
                        return dist
        else:
            if use_wraparound:
                for dist in self._wraparound_distances[start_key].keys():
                    if end_key in self._wraparound_distances[start_key][dist]:
                        return dist
            else:
                for dist in self._no_wraparound_distances[start_key].keys():
                    if end_key in self._no_wraparound_distances[start_key][dist]:
                        return dist

    def get_adjacency_list(self, use_shortcuts, use_wraparound):
        if use_shortcuts:
            if use_wraparound:
                with open(self._path.replace('.json', '_all.json.gz'), 'r') as f:
                    return json.load(f)['adjacency_list']
            else:
                with open(self._path.replace('.json', '_shortcuts.json.gz'), 'r') as f:
                    return json.load(f)['adjacency_list']
        else:
            if use_wraparound:
                with open(self._path.replace('.json', '_wraparound.json.gz'), 'r') as f:
                    return json.load(f)['adjacency_list']
            else:
                with open(self._path, 'r') as f:
                    return json.load(f)['adjacency_list']
