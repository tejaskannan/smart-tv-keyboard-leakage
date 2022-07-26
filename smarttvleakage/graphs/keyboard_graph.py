import os.path
from collections import deque, defaultdict
from enum import Enum, auto
from typing import Dict, DefaultDict, List, Set
import csv

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
        standard_path = os.path.join(dir_name, 'samsung_keyboard.csv')
        special_one_path = os.path.join(dir_name, 'samsung_keyboard_special_1.csv')

        self._keyboards = {
            KeyboardMode.STANDARD: SingleKeyboardGraph(path=standard_path, start_key=START_KEYS[KeyboardMode.STANDARD]),
            KeyboardMode.SPECIAL_ONE: SingleKeyboardGraph(path=special_one_path, start_key=START_KEYS[KeyboardMode.SPECIAL_ONE])
        }

    def get_keys_for_moves_from(self, start_key: str, num_moves: int, mode: KeyboardMode) -> List[str]:
        if num_moves < 0:
            return []

        return self._keyboards[mode].get_keys_for_moves_from(start_key=start_key, num_moves=num_moves)

    def printerthing(self, num_moves: int, mode: KeyboardMode) -> List[str]:
        return self._keyboards[mode].get_keys_for_moves(num_moves)


class SingleKeyboardGraph:

    def __init__(self, path: str, start_key: str):
        f = open(path)
        self.no_wraparound = list(csv.reader(f))
        f.close()
        f = open(path[:-4]+"_wraparound"+[-4:])
        self.wraparound = list(csv.reader(f))
        f.close()
        start_idx = self.no_wraparound[0].index(start_key)
        temp = []
        self._keys_for_distance = {}
        for x,i in enumerate(self.no_wraparound[start_idx]):
            if i in self._keys_for_distance.keys():
                self._keys_for_distance[i].append(self.no_wraparound[0][x])
            else:
                self._keys_for_distance[i] = [self.no_wraparound[0][x]]
        for x,i in enumerate(self.wraparound[start_idx]):
            if self.wraparound[0][start_idx] in self._keys_for_distance[i]:
                continue
            else:
                if i in self._keys_for_distance.keys():
                    self._keys_for_distance[i].append(self.wraparound[0][x])
                else:
                    self._keys_for_distance[i] = [self.wraparound[0][x]]

    def get_keys_for_moves(self, num_moves: int) -> List[str]:
        return self._keys_for_distance.get(num_moves, [])


    def get_keys_for_moves_from(self, start_key: str, num_moves: int) -> List[str]:
        idx = self.no_wraparound[0].index(start_key)+1
        candidates = []

        for x,i in enumerate(self.no_wraparound[idx]):

            i=int(float(i))
            if i == num_moves:

                candidates.append(self.no_wraparound[0][x])
        for x,i in enumerate(self.wraparound[idx]):

            i=int(float(i))
            if i == num_moves and int(i) not in candidates:
                candidates.append(self.wraparound[0][x])


        return list(sorted(candidates))