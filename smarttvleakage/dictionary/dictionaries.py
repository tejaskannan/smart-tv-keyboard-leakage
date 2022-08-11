import string
import os.path
import io
import gzip
from collections import Counter
from typing import Dict, List, Optional, Iterable, Tuple, Any

from smarttvleakage.utils.ngrams import create_ngrams
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
        self._single_char_counts: Counter = Counter()
        self._bigram_counts = Trie(max_depth=2)
        self._is_built = False
        self._max_depth = max_depth

    def parse_dictionary_file(self, path: str, min_count: int, has_counts: bool) -> Dict[str, int]:
        string_dictionary: Dict[str, int] = dict()

        if path.endswith('.json'):
            string_dictionary = read_json(path)
        elif path.endswith('.txt'):
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

        return string_dictionary

    def build(self, path: str, min_count: int, has_counts: bool):
        if self._is_built:
            return

        # Read the input words
        string_dictionary = self.parse_dictionary_file(path=path,
                                                       min_count=min_count,
                                                       has_counts=has_counts)

        print('Indexing {} Strings...'.format(len(string_dictionary)))

        # Build the trie
        for word, count in string_dictionary.items():
            count = max(count, 1)

            # Increment the single character counts
            for character in word:
                self._single_char_counts[character] += count

            for bigram in create_ngrams(word, 2):
                self._bigram_counts.add_string(bigram, count=count)

            self._trie.add_string(word, count=count)

        self._is_built = True

    def iterate_words(self, path: str) -> Iterable[str]:
        assert path.endswith('txt'), 'Must provide a text file'

        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    yield line.split()[0]

    def estimate_remaining_log_prob(self, prefix: str, length: int) -> float:
        return self._trie.get_max_log_prob(prefix=prefix,
                                           single_char_counts=self._single_char_counts,
                                           length=length)

    def get_words_for(self, prefixes: Iterable[str], max_num_results: int, min_length: Optional[int], max_count_per_prefix: Optional[int]) -> Iterable[Tuple[str, float]]:
        return self._trie.get_words_for(prefixes, max_num_results, min_length=min_length, max_count_per_prefix=max_count_per_prefix)

    def get_score_for_prefix(self, prefix: str, min_length: int) -> float:
        return self._trie.get_score_for_prefix(prefix=prefix, min_length=min_length)

    def get_score_for_string(self, string: str, should_aggregate: bool) -> float:
        return self._trie.get_score_for_string(string=string, should_aggregate=should_aggregate)

    def does_contain_prefix(self, prefix: str) -> bool:
        return self._trie.get_node_for(prefix) is not None

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'
        
        #if length is not None:
        #    length = min(length, self._max_depth)

        # Get the prior counts of the next characters using the given prefix
        character_counts = self._trie.get_next_characters(prefix, length=length)

        if (len(character_counts) == 0) and (len(prefix) >= 1):
            character_counts = self._bigram_counts.get_next_characters(prefix=prefix[-1], length=None)

        # If we still have no characters, then use the single-character counts
        if len(character_counts) == 0:
            character_counts = {key: count for key, count in self._single_char_counts.items()}

        # Re-map the character names
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): count for char, count in character_counts.items()}

        # Apply Laplace Smoothing
        for character in self._characters:
            character_counts[character] = character_counts.get(character, 0) + 1

        # Normalize the result
        total_count = sum(character_counts.values())
        return {char: (count / total_count) for char, count in character_counts.items()}

    @classmethod
    def restore(cls, serialized: Dict[str, Any]):
        dictionary = cls(max_depth=1)
        dictionary._trie = serialized['trie']
        dictionary._single_char_counts = serialized['single_char_counts']
        dictionary._bigram_counts = serialized['bigram_counts']
        dictionary._is_built = True
        dictionary._max_depth = dictionary._trie.max_depth
        return dictionary

    def save(self, path: str):
        data_dict = {
            'trie': self._trie,
            'single_char_counts': self._single_char_counts,
            'bigram_counts': self._bigram_counts,
            'dict_type': 'english'
        }
        save_pickle_gz(data_dict, path)


class NgramDictionary(EnglishDictionary):

    def __init__(self):
        super().__init__(max_depth=0)
        self._is_built = False

        self._single_char_counts: Counter = Counter()
        self._2gram_trie = Trie(max_depth=2)
        self._3gram_trie = Trie(max_depth=3)
        self._4gram_trie = Trie(max_depth=4)
        self._total_count = 0

    def build(self, path: str, min_count: int, has_counts: bool):
        if self._is_built:
            return

        # Read the input words
        string_dictionary = self.parse_dictionary_file(path=path,
                                                       min_count=min_count,
                                                       has_counts=has_counts)

        # Build the ngram tries
        for word, count in string_dictionary.items():
            count = max(count, 1)

            for character in create_ngrams(word, 1):
                self._single_char_counts[character] += count

            for two_gram in create_ngrams(word, 2):
                self._2gram_trie.add_string(two_gram, count=count)

            for three_gram in create_ngrams(word, 3):
                self._3gram_trie.add_string(three_gram, count=count)

            for four_gram in create_ngrams(word, 4):
                self._4gram_trie.add_string(four_gram, count=count)

            self._total_count += 1

            if (self._total_count % 10000) == 0:
                print('Completed {} Strings...'.format(self._total_count), end='\r')

        print()
        self._is_built = True

    def projected_remaining_prob(self, prefix: str, length: int) -> float:
        return 1.0

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'
        length = length if length is not None else len(prefix)

        if length == 1:
            character_counts = {char: count for char, count in self._single_char_counts.items()}
        elif length == 2:
            character_counts = self._2gram_trie.get_next_characters(prefix, length=None)
        elif length == 3:
            character_counts = self._3gram_trie.get_next_characters(prefix[-2:], length=None)
        else:
            character_counts = self._4gram_trie.get_next_characters(prefix[-3:], length=None) 

        # Apply Laplace Smoothing
        for character in self._characters:
            character_counts[character] = character_counts.get(character, 0) + 1

        # Convert any needed characters and normalize the result
        total_count = sum(character_counts.values())
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): (count / total_count) for char, count in character_counts.items()}

        return character_counts

    def does_contain_prefix(prefix: str) -> bool:
        return False

    @classmethod
    def restore(cls, serialized: Dict[str, Any]):
        dictionary = cls()
        dictionary._single_char_counts = serialized['1gram']
        dictionary._2gram_trie = serialized['2gram']
        dictionary._3gram_trie = serialized['3gram']
        dictionary._4gram_trie = serialized['4gram']
        dictionary._is_built = True
        return dictionary

    def save(self, path: str):
        data_dict = {
            '1gram': self._single_char_counts,
            '2gram': self._2gram_trie,
            '3gram': self._3gram_trie,
            '4gram': self._4gram_trie,
            'dict_type': 'ngram'
        }
        save_pickle_gz(data_dict, path)


def restore_dictionary(path: str) -> CharacterDictionary:
    if path == 'uniform':
        return UniformDictionary()
    else:
        data_dict = read_pickle_gz(path)
        dict_type = data_dict['dict_type']

        if dict_type == 'english':
            return EnglishDictionary.restore(data_dict)
        elif dict_type == 'ngram':
            return NgramDictionary.restore(data_dict)
        else:
            raise ValueError('Unknown dictionary type: {}'.format(dict_type))
