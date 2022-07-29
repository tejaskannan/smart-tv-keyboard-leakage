import os.path
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set
import csv

from smarttvleakage.dictionary.dictionaries import SPACE, CHANGE, BACKSPACE
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_json, read_json_gz


SAMSUNG_STANDARD = 'standard'
SAMSUNG_SPECIAL_ONE = 'special_1'
APPLETV_ALPHABET = 'alphabet'
APPLETV_NUMBERS = 'numbers'
APPLETV_SPECIAL = 'special'


START_KEYS = {
    SAMSUNG_STANDARD: 'q',
    SAMSUNG_SPECIAL_ONE: CHANGE,
    APPLETV_ALPHABET: 't',
    APPLETV_NUMBERS: CHANGE,
    APPLETV_SPECIAL: CHANGE
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

    def __init__(self, tv_type: SmartTVType):
        self._tv_type = tv_type

        dir_name = os.path.dirname(__file__)

        if tv_type == SmartTVType.SAMSUNG:
            standard_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard.json')
            special_one_path = os.path.join(dir_name, 'samsung', 'samsung_keyboard_special_1.json')

            self._keyboards = {
                SAMSUNG_STANDARD: SingleKeyboardGraph(path=standard_path, start_key=START_KEYS[SAMSUNG_STANDARD]),
                SAMSUNG_SPECIAL_ONE: SingleKeyboardGraph(path=special_one_path, start_key=START_KEYS[SAMSUNG_SPECIAL_ONE])
            }
        elif tv_type == SmartTVType.APPLE_TV:
            alphabet_path = os.path.join(dir_name, 'apple_tv', 'alphabet.json')
            numbers_path = os.path.join(dir_name, 'apple_tv', 'numbers.json')
            special_path = os.path.join(dir_name, 'apple_tv', 'special.json')

            self._keyboards = {
                APPLETV_ALPHABET: SingleKeyboardGraph(path=alphabet_path, start_key=START_KEYS[APPLETV_ALPHABET]),
                APPLETV_NUMBERS: SingleKeyboardGraph(path=numbers_path, start_key=START_KEYS[APPLETV_NUMBERS]),
                APPLETV_SPECIAL: SingleKeyboardGraph(path=special_path, start_key=START_KEYS[APPLETV_SPECIAL])
            }
        else:
            raise ValueError('Unknown TV type: {}'.format(tv_type.name))

    def is_unclickable(self, key: str, mode: str) -> bool:
        return self._keyboards[mode].is_unclickable(key)

    def get_characters(self) -> List[str]:
        merged: Set[str] = set()

        for keyboard in self._keyboards.values():
            merged.update(keyboard.get_characters())

        return list(merged)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, mode: str, use_shortcuts: bool, use_wraparound: bool) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_from(start_key=start_key, num_moves=num_moves, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound)

    def printerthing(self, num_moves: int, mode: str) -> List[str]:
        return self._keyboards[mode].get_keys_for_moves(num_moves)


class SingleKeyboardGraph:

    def __init__(self, path: str, start_key: str):
        # Set the start key
        self._start_key = start_key

        # Read in the graph config to get the unclickable keys
        graph_config = read_json(path)
        self._characters = list(graph_config['adjacency_list'].keys())
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

    def get_keys_for_moves(self, num_moves: int) -> List[str]:
        return self.get_keys_for_moves_from(start_key=self._start_key, num_moves=num_moves, use_space=False)

    def get_characters(self) -> List[str]:
        return self._characters

    def is_unclickable(self, key: str) -> bool:
        return (key in self._unclickable_keys)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, use_shortcuts: bool, use_wraparound: bool) -> List[str]:
        assert num_moves >= 0, 'Must provide a non-negative number of moves. Got {}'.format(num_moves)

        if num_moves == 0:
            return [start_key]

        no_wraparound_distance_dict = self._no_wraparound_distances.get(start_key, dict())
        wraparound_distance_dict = self._wraparound_distances.get(start_key, dict())

        if use_shortcuts:
            no_wraparound_distance_dict.update(self._no_wraparound_distances_shortcuts.get(start_key, dict()))
            wraparound_distance_dict.update(self._wraparound_distances_shortcuts.get(start_key, dict()))

        if (len(no_wraparound_distance_dict) == 0) and (len(wraparound_distance_dict) == 0):
            return []

        no_wraparound_neighbors = no_wraparound_distance_dict.get(num_moves, set())
        wraparound_neighbors = wraparound_distance_dict.get(num_moves, set())

        if use_wraparound:
            combined = no_wraparound_neighbors.union(wraparound_neighbors)
            return list(sorted(combined))
        else:
            return list(sorted(no_wraparound_neighbors))
