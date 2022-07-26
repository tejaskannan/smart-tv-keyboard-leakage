import os.path
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set

from utils.file_utils import read_json



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
    def printerthing(self, num_moves: int) -> List[str]:
        return self.keyboards[mode].get_keys_for_moves(num_moves)


class SingleKeyboardGraph:

    def __init__(self, path: str, start_key: str):
        f = open('../graphs/samsung_keyboard.csv')
        self.no_wraparound = list(csv.reader(f))
        f.close()
        f = open('../graphs/samsung_keyboard_wraparound.csv')
        self.wraparound = list(csv.reader(f))
        f.close()
        start_idx = self.nowraparound.index(start_key)
        temp = []
        self._keys_for_distance = {}
        for x,i in enumerate(no_wraparound[start_idx]):
            self._keys_for_distance[i].append(no_wraparound[x])
        for x,i in enumerate(wraparound[0][start_idx]):
            if wraparound[0][start_idx] in self._keys_for_distance[i]:
                continue
            else:
                self._keys_for_distance[i].append(no_wraparound[x])
        self._keys_for_distance = 

    def get_keys_for_moves(self, num_moves: int) -> List[str]:
        return self._keys_for_distance.get(num_moves, [])


    # def get_moves_for_key(self, key: str) -> int:
    #     return self._shortest_distances.get(key, -1)
    # NOT USED

    def get_keys_for_moves_from(self, start_key: str, num_moves: int) -> List[str]:
        
        idx = no_wraparound.index(start_key)
        candidates = []
        for x,i in enumerate(no_wraparound[idx]):
            if i == num_moves:
                candidates.append(no_wraparound[0][x])
        for x,i in enumerate(wraparound[idx]):
            if i == num_moves and i not in candidates:
                candidates.append(wraparound[0][x])

        return list(sorted(candidates))