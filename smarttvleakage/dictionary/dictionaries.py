import string
import os.path
import io
import gzip
from collections import Counter
from typing import Dict, List, Optional, Iterable, Tuple

from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, save_pickle_gz
from smarttvleakage.dictionary.trie import Trie

UNPRINTED_CHARACTERS = frozenset({ '<CHANGE>', '<RIGHT>', '<LEFT>', '<UP>', '<DOWN>', '<BACK>', '<CAPS>', '<NEXT>' })
SELECT_SOUND_KEYS = frozenset({ '<CHANGE>', '<CAPS>', '<NEXT>', '<SPACE>', '<LEFT>', '<RIGHT>', '<UP>', '<DOWN>', '<LANGUAGE>', '<DONE>', '<CANCEL>'})
DELETE_SOUND_KEYS = frozenset({ '<BACK>', '<DELETEALL>' })


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
BACKSPACE = '<BACK>'
NEXT = '<NEXT>'
SPACE = '<SPACE>'
DISCOUNT_FACTOR = 0.1


CHARACTER_TRANSLATION = {
    '<MULT>': '×',
    '<DIV>': '÷',
    '<SPACE>': ' ',
    '<WWW>': 'www',
    '<COM>': 'com',
    '<POUND>': '£',
    '<EURO>': '€'
}

REVERSE_CHARACTER_TRANSLATION = { value: key for key, value in CHARACTER_TRANSLATION.items() }


class CharacterDictionary:

    def __init__(self):
        self._characters: List[str] = []

    @property
    def characters(self) -> List[str]:
        return self._characters

    def set_characters(self, characters: List[str]):
        self._characters = characters

    def smooth_character_counts(self, prefix: str, counts: Dict[str, int]) -> Dict[str, int]:
        return counts

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        assert len(self.characters) > 0, 'Must call set_characters() first'
        raise NotImplementedError()


class UniformDictionary(CharacterDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        return { REVERSE_CHARACTER_TRANSLATION.get(char, char): 1 for char in self.characters }

# for CCs
class NumericDictionary(CharacterDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        nd = {}
        for c in CHARACTERS:
            if c.isnumeric():
                nd[c] = 1000
            else:
                nd[c] = 0
        return nd

## could do stronger weighting?
## add more for first digits?
class CreditCardDictionary(CharacterDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        length = len(prefix)
        nd = {}
        firsts = ["3", "4", "5", "6"]

        if length > 1: # uniform
            for c in CHARACTERS:
                if c.isnumeric():
                    nd[c] = 1000
                else:
                    nd[c] = 0

        elif length == 0:
            for c in CHARACTERS:
                if c in firsts:
                    nd[c] = 1000
                elif c.isnumeric():
                    nd[c] = 100
                else:
                    nd[c] = 0
        elif length == 1:
            if prefix == "3":
                for c in CHARACTERS:
                    if c == "4" or c == "7":
                        nd[c] = 1000
                    elif c.isnumeric():
                        nd[c] = 100
                    else:
                        nd[c] = 0
            else:
                for c in CHARACTERS:
                    if c.isnumeric():
                        nd[c] = 1000
                    else:
                        nd[c] = 0

        return nd
##
class CreditCardDictionaryStrong(CharacterDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        length = len(prefix)
        nd = {}
        firsts = ["3", "4", "5", "6"]

        if length == 0:
            for c in CHARACTERS:
                if c in firsts:
                    nd[c] = 1000
                elif c.isnumeric():
                    nd[c] = 10
                else:
                    nd[c] = 0

        elif length == 1:
            for c in CHARACTERS:

                if prefix == "3":
                    if c in ["4", "7"]:
                        nd[c] = 1000
                    elif c.isnumeric():
                        nd[c] = 10
                    else:
                        nd[c] = 0
                elif prefix == "4":
                    if c in ["0", "1", "4", "8", "9"]:
                        nd[c] = "1000"
                    elif c.isnumeric():
                        nd[c] = 200
                    else:
                        nd[c] = 0
                elif prefix == "5":
                    if c in ["0", "1", "2", "3", "4", "5"]:
                        nd[c] = "1000"
                    elif c.isnumeric():
                        nd[c] = 200
                    else:
                        nd[c] = 0
                # continue making cc dict

        elif length > 1: # uniform
            for c in CHARACTERS:
                if c.isnumeric():
                    nd[c] = 1000
                else:
                    nd[c] = 0
                    
       

        return nd
##


class EnglishDictionary(CharacterDictionary):

    def __init__(self, max_depth: int):
        super().__init__()
        self._trie = Trie(max_depth=max_depth)
        self._is_built = False
        self._max_depth = max_depth

        self._single_char_counts: Counter = Counter()
        self._two_char_counts: Dict[str, Counter] = dict()

    @property
    def total_count(self) -> int:
        assert self._is_built, 'Must call build() first'
        return self._trie._root.count

    def build(self, path: str, min_count: int, has_counts: bool):
        if self._is_built:
            return

        # Read the input words
        if path.endswith('.json'):
            string_dictionary = read_json(path)
        elif path.endswith('.txt'):
            string_dictionary: Dict[str, int] = dict()

            with open(path, 'rb') as fin:
                io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

                for line in io_wrapper:
                    line = line.strip()

                    if not has_counts:
                        string_dictionary[line] = 1
                    else:
                        tokens = line.strip().split()
                        count = int(tokens[-1])
                        string = ' '.join(tokens[0:-1])

                        if count >= min_count:
                            string_dictionary[string] = count
        elif path.endswith('.gz'):
            string_dictionary: Dict[str, int] = dict()
            with gzip.open(path, 'rt', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if len(line) > 0:
                        string_dictionary[line] = 1
        else:
            raise ValueError('Unknown file type: {}'.format(path))

        # Build the trie
        for word, count in string_dictionary.items():
            count = max(count, 1)

            # Increment the single character counts
            for character in word:
                self._single_char_counts[character] += 1

            # Increment the bi-gram counts
            for idx in range(len(word) - 1):
                first_char = word[idx]
                second_char = word[idx + 1]

                if first_char not in self._two_char_counts:
                    self._two_char_counts[first_char] = Counter()

                self._two_char_counts[first_char][second_char] += 1

            self._trie.add_string(word, count=max(count, 1))

        self._is_built = True

    def iterate_words(self, path: str) -> Iterable[str]:
        assert path.endswith('txt'), 'Must provide a text file'

        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    yield line.split()[0]

    def get_words_for(self, prefixes: Iterable[str], max_num_results: int, min_length: Optional[int], max_count_per_prefix: Optional[int]) -> Iterable[Tuple[str, float]]:
        return self._trie.get_words_for(prefixes, max_num_results, min_length=min_length, max_count_per_prefix=max_count_per_prefix)

    def get_score_for_prefix(self, prefix: str, min_length: int) -> float:
        return self._trie.get_score_for_prefix(prefix=prefix, min_length=min_length)

    def get_score_for_string(self, string: str, should_aggregate: bool) -> float:
        return self._trie.get_score_for_string(string=string, should_aggregate=should_aggregate)

    def does_contain_string(self, string: str) -> bool:
        return self._trie.does_contain_string(string)

    def smooth_letter_counts(self, prefix: str, counts: Dict[str, int], min_count: int) -> Dict[str, Tuple[int, float]]:
        smoothed: Dict[str, int] = { key: (count, 1.0) for key, count in counts.items() }

        if prefix == 'ted l':
            print(counts)

        num_above_min_count = sum((int(count >= min_count) for count in counts.values()))

        if (num_above_min_count == 0) and (len(prefix) >= 1):
            bigram_suffix = prefix[-1]
            smoothed = {char: (count + 100, DISCOUNT_FACTOR) for char, count in self._two_char_counts.get(bigram_suffix, dict()).items()}

            if prefix == 'ted l':
                print(smoothed)

        if num_above_min_count == 0:
            smoothed = {char: (count + 100, DISCOUNT_FACTOR) for char, count in self._single_char_counts.items()}

        for c in self._characters:
            if c in smoothed:
                smoothed[c] = (smoothed[c][0] + 1, smoothed[c][1])
            else:
                smoothed[c] = (1, 1.0)

        return smoothed

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'

        # Get the prior counts of the next characters using the given prefix
        character_counts = self._trie.get_next_characters(prefix, length=length)

        # Convert any needed characters
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): count for char, count in character_counts.items()}

        return character_counts

    @classmethod
    def restore(cls, path: str):
        data_dict = read_pickle_gz(path)

        dictionary = cls(max_depth=1)
        dictionary._trie = data_dict['trie']
        dictionary._single_char_counts = data_dict['single_char_counts']
        dictionary._two_char_counts = data_dict['two_char_counts']
        dictionary._is_built = True
        dictionary._max_depth = dictionary._trie.max_depth
        return dictionary

    def save(self, path: str):
        data_dict = {
            'trie': self._trie,
            'single_char_counts': self._single_char_counts,
            'two_char_counts': self._two_char_counts
        }
        save_pickle_gz(data_dict, path)
