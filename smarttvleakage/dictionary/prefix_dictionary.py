from collections import defaultdict
from typing import DefaultDict


class PrefixDictionary:

    def __init__(self, max_length: int):
        self._dictionary: Dict[str, DefaultDict[str, int]] = dict()
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        return self._max_length

    def add_string(self, string: str):
        if len(string) == 0:
            return

        self._dictionary['<ROOT>'][string[0]] += 1

        for idx in range(1, len(string) - 1):
            prefix_length = min(idx, self.max_length)
            prefix = string[0:prefix_length]
            next_char = string[idx]

            if prefix not in self._dictionary:
                self._dictionary[prefix] = defaultdict(int)

            self._dictionary[prefix][next_char] += 1

    def get_next_characters(self, prefix: str) -> DefaultDict[str, int]:
        default = defaultdict(int)
        return self._dictionary.get(prefix, default)
