from collections import Counter
from typing import Dict

from smarttvleakage.utils.file_utils import read_json


class EnglishDictionary:

    def __init__(self, path: str):
        self._dictionary = read_json(path)

    def get_letter_freq(self, prefix: str, total_length: int) -> Dict[str, float]:
        assert total_length > len(prefix), 'The total length must be longer than the prefix.'

        letter_counts: Counter = Counter()
        total_count = 0
        current_position = len(prefix)

        for word in self._dictionary.keys():
            if (len(word) == total_length) and (word.startswith(prefix)):
                letter_counts[word[current_position]] += 1
                total_count += 1

        result: Dict[str, float] = dict()
        for letter, count in letter_counts.items():
            result[letter] = count / total_count

        return result
