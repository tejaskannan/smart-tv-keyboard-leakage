import os.path
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set
import csv

from smarttvleakage.dictionary.dictionaries import SPACE, CHANGE, BACKSPACE
from smarttvleakage.utils.file_utils import read_json


class KeyboardMode(Enum):
    STANDARD = auto()
    SPECIAL_ONE = auto()


START_KEYS = {
    KeyboardMode.STANDARD: 'q',
    KeyboardMode.SPECIAL_ONE: CHANGE
}


FILTERED_KEYS = [BACKSPACE]


def parse_graph_distances(path: str) -> Dict[str, DefaultDict[int, Set[str]]]:
    with open(path, 'r') as fin:
        distance_matrix = list(csv.reader(fin))

    result: Dict[str, DefaultDict[int, Set[str]]] = dict()

    for start_idx, start_key in enumerate(distance_matrix[0]):
        result[start_key] = defaultdict(set)

        for end_idx, end_key in enumerate(distance_matrix[0]):
            distance = int(result[start_idx + 1][end_idx])
            result[start_key][distance].add(end_key)

    return result


class MultiKeyboardGraph:

    def __init__(self):
        dir_name = os.path.dirname(__file__)
        standard_path = os.path.join(dir_name, 'samsung_keyboard.csv')
        special_one_path = os.path.join(dir_name, 'samsung_keyboard_special_1.csv')

        self._keyboards = {
            KeyboardMode.STANDARD: SingleKeyboardGraph(path=standard_path, start_key=START_KEYS[KeyboardMode.STANDARD]),
            KeyboardMode.SPECIAL_ONE: SingleKeyboardGraph(path=special_one_path, start_key=START_KEYS[KeyboardMode.SPECIAL_ONE])
        }

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, mode: KeyboardMode, use_space: bool) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_from(start_key=start_key, num_moves=num_moves, use_space=use_space)

    def printerthing(self, num_moves: int, mode: KeyboardMode) -> List[str]:
        return self._keyboards[mode].get_keys_for_moves(num_moves)


class SingleKeyboardGraph:

    def __init__(self, path: str, start_key: str):
        self._start_key = start_key

        # Get the graph without wraparound
        with open(path, 'r') as fin:
            self._no_wraparound = list(csv.reader(fin))

        # Get the graph with wraparound
        with open(path.replace('.csv', '_wraparound.csv'), 'r') as fin:
            self._wraparound = list(csv.reader(fin))

        # Read in the precomputed shortest paths map
        start_idx = self._no_wraparound[0].index(start_key)

        self._no_wraparound_distances: Dict[str, DefaultDict[int, Set[str]]] = parse_graph_distances(path=path)
        self._wraparound_distances: Dict[str, DefaultDict[int, Set[str]]] = parse_graph_distances(path=path.replace('.csv', '_wraparound.csv')

    def get_keys_for_moves(self, num_moves: int) -> List[str]:
        return self.get_keys_for_moves_from(start_key=self._start_key, num_moves=num_moves, use_space=False)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, use_space: bool) -> List[str]:
        no_wraparound_distance_dict = self._no_wraparound_distances.get(start_key, dict())
        wraparound_distance_dict = self._wraparound_distances.get(start_key, dict())

        if (len(no_wraparound_distance_dict) == 0) and (len(wraparound_distance_dict) == 0):
            return []

        no_wraparound_neighbors = no_wraparound_distance_dict.get(num_moves, set())
        wraparound_neighbors = wraparound_distance_dict.get(num_moves, set())

        combined = no_wraparound_neighbors.union(wraparound_neighbors)
        return list(sorted(combined))
