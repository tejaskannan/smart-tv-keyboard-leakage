import os.path
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set

from smarttvleakage.utils.file_utils import read_json



class KeyboardMode(Enum):
    STANDARD = auto()
    SPECIAL_ONE = auto()


START_KEYS = {
    KeyboardMode.STANDARD: 'q',
    KeyboardMode.SPECIAL_ONE: '<CHANGE>'
}


FILTERED_KEYS = ['<BACK>', '<SPACE>']


class MultiKeyboardGraph:

    def __init__(self):
        dir_name = os.path.dirname(__file__)
        standard_path = os.path.join(dir_name, 'samsung_keyboard.json')
        special_one_path = os.path.join(dir_name, 'samsung_keyboard_special_1.json')

        self._keyboards = {
            KeyboardMode.STANDARD: SingleKeyboardGraph(path=standard_path, start_key=START_KEYS[KeyboardMode.STANDARD]),
            KeyboardMode.SPECIAL_ONE: SingleKeyboardGraph(path=special_one_path, start_key=START_KEYS[KeyboardMode.SPECIAL_ONE])
        }

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, mode: KeyboardMode) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_from(start_key=start_key, num_moves=num_moves)


class SingleKeyboardGraph:

    def __init__(self, path: str, start_key: str):
        self._adjacency_list = read_json(path)

        # Verify that all edges go two ways
        for key, neighbors in self._adjacency_list.items():
            for neighbor in neighbors:
                assert (key in self._adjacency_list[neighbor]), '{} should be a neighbor of {}'.format(key, neighbor)

        # Get the shortest paths to all nodes
        self._shortest_distances: Dict[str, int] = dict()
        self._keys_for_distance: DefaultDict[int, List[str]] = defaultdict(list)

        visited: Set[str] = set()

        frontier = deque()
        frontier.append((start_key, 0))

        while len(frontier) > 0:
            (key, dist) = frontier.popleft()

            if key in visited:
                continue

            self._shortest_distances[key] = dist
            self._keys_for_distance[dist].append(key)
            visited.add(key)

            for neighbor in self._adjacency_list[key]:
                if neighbor not in visited:
                    frontier.append((neighbor, dist + 1))

    def get_keys_for_moves(self, num_moves: int) -> List[str]:
        return self._keys_for_distance.get(num_moves, [])

    def get_moves_for_key(self, key: str) -> int:
        return self._shortest_distances.get(key, -1)

    def get_keys_for_moves_from(self, start_key: str, num_moves: int) -> List[str]:
        visited: Set[str] = set()

        frontier = deque()
        frontier.append((start_key, 0))

        candidates: Set[str] = set()

        while len(frontier) > 0:
            (key, dist) = frontier.popleft()

            if not key.startswith('<'):
                key = key.lower()

            if key in visited:
                continue

            visited.add(key)

            if dist > num_moves:
                continue
            elif (dist == num_moves) and (key not in FILTERED_KEYS):
                candidates.add(key)

            assert key in self._adjacency_list, 'Found invalid key: {}. Start Key: {}, Num Moves: {}'.format(key, start_key, num_moves)

            for neighbor in self._adjacency_list[key]:
                if neighbor not in visited:
                    frontier.append((neighbor, dist + 1))

        return list(sorted(candidates))
