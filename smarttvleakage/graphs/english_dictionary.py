import string
import os.path
import io
from collections import Counter
from typing import Dict, List

from smarttvleakage.utils.file_utils import read_json


standard_graph = read_json(os.path.join(os.path.dirname(__file__), 'samsung_keyboard.json'))
special_graph = read_json(os.path.join(os.path.dirname(__file__), 'samsung_keyboard_special_1.json'))

CHARACTERS: List[str] = list(sorted(standard_graph.keys())) + list(sorted(special_graph.keys()))


class CharacterDictionary:

    def get_letter_freq(self, prefix: str, total_length: int) -> Dict[str, float]:
        raise NotImplementedError()


class UniformDictionary(CharacterDictionary):

    def get_letter_freq(self, prefix: str, total_length: int) -> Dict[str, float]:
        return { c: 1.0 / len(CHARACTERS) for c in CHARACTERS }


class EnglishDictionary(CharacterDictionary):

    def __init__(self, path: str):
        if path.endswith('.json'):
            self._dictionary = read_json(path)
        elif path.endswith('.txt'):
            self._dictionary: Dict[str, int] = dict()

            with open(path, 'rb') as fin:
                io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

                for line in io_wrapper:
                    line = line.strip()
                    if len(line) > 0:
                        self._dictionary[line] = 0

    def get_letter_freq(self, prefix: str, total_length: int) -> Dict[str, float]:
        assert total_length > len(prefix), 'The total length must be longer than the prefix.'

        letter_counts: Counter = Counter()
        total_count = 0
        current_position = len(prefix)

        for word in self._dictionary.keys():
            if (len(word) <= total_length) and (word.startswith(prefix)):
                letter_counts[word[current_position]] += 1
                total_count += 1

        # Use laplace smoothing
        for c in CHARACTERS:
            letter_counts[c] += 1

        result: Dict[str, float] = dict()
        for letter, count in letter_counts.items():
            result[letter] = count / total_count

        return result
