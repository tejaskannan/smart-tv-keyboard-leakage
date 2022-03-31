import os.path
from collections import deque, defaultdict
from typing import Dict, DefaultDict, List, Set

from smarttvleakage.utils.file_utils import read_json


START_KEY = 'q'


class KeyboardGraph:

    def __init__(self):
        dir_name = os.path.dirname(__file__)
        self._adjacency_list = read_json(os.path.join(dir_name, 'samsung_keyboard.json'))

        # Verify that all edges go two ways
        for key, neighbors in self._adjacency_list.items():
            for neighbor in neighbors:
                assert (key in self._adjacency_list[neighbor]), '{} should be a neighbor of {}'.format(key, neighbor)

        # Get the shortest paths to all nodes
        self._shortest_distances: Dict[str, int] = dict()
        self._keys_for_distance: DefaultDict[int, List[str]] = defaultdict(list)

        visited: Set[str] = set()

        frontier = deque()
        frontier.append((START_KEY, 0))

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
