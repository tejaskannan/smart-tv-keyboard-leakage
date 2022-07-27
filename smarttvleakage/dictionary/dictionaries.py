import string
import os.path
import io
import gzip
from collections import Counter
from typing import Dict, List, Optional, Iterable, Tuple

from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, save_pickle_gz
from smarttvleakage.dictionary.trie import Trie

standard_graph = read_json(os.path.join(os.path.dirname(__file__), '..', 'graphs', 'samsung', 'samsung_keyboard.json'))['adjacency_list']
special_graph = read_json(os.path.join(os.path.dirname(__file__), '..', 'graphs', 'samsung', 'samsung_keyboard_special_1.json'))['adjacency_list']

CHARACTERS: List[str] = list(sorted(standard_graph.keys())) + list(sorted(special_graph.keys()))
UNPRINTED_CHARACTERS = { '<CHANGE>', '<RIGHT>', '<LEFT>', '<UP>', '<DOWN>', '<BACK>', '<CAPS>', '<NEXT>' }
SELECT_SOUND_KEYS = { '<CHANGE>', '<CAPS>', '<NEXT>', '<SPACE>' }


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
BACKSPACE = '<BACK>'
NEXT = '<NEXT>'
SPACE = '<SPACE>'


CHARACTER_TRANSLATION = {
    '<MULT>': '×',
    '<DIV>': '÷',
    '<SPACE>': ' ',
    '<WWW>': 'www',
    '<COM>': 'com'
}


class CharacterDictionary:

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        raise NotImplementedError()


class UniformDictionary(CharacterDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        return { c: 1 for c in CHARACTERS }


class EnglishDictionary(CharacterDictionary):

    def __init__(self, max_depth: int):
        self._trie = Trie(max_depth=max_depth)
        self._is_built = False
        self._max_depth = max_depth

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
                        print('tokens: ', tokens)
                        print('count: ', count)
                        string = ' '.join(tokens[0:-1])
                        print(string)

                        if count > min_count:
                            string_dictionary[tokens[0]] = count
        elif path.endswith('.gz'):
            string_dictionary: Dict[str, int] = dict()
            with gzip.open(path, 'rt') as fin:
                for line in fin:
                    line = line.strip()
                    if len(line) > 0:
                        string_dictionary[line] = 1
        else:
            raise ValueError('Unknown file type: {}'.format(path))

        # Build the trie
        for word, count in string_dictionary.items():
            self._trie.add_string(word, count=max(count, 1))

        self._is_built = True

    def iterate_words(self, path: str) -> Iterable[str]:
        assert path.endswith('txt'), 'Must provide a text file'

        with open(path, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    yield line.split()[0]

    def get_words_for(self, prefixes: Iterable[str], max_num_results: int, min_length: Optional[int], max_count_per_prefix: Optional[int]) -> Iterable[Tuple[str, float]]:
        return self._trie.get_words_for(prefixes, max_num_results, min_length=min_length, max_count_per_prefix=max_count_per_prefix)

    def get_score_for_string(self, string: str, should_aggregate: bool) -> float:
        return self._trie.get_score_for_string(string=string, should_aggregate=should_aggregate)

    def does_contain_string(self, string: str) -> bool:
        return self._trie.does_contain_string(string)

    def get_letter_counts(self, prefix: str, length: Optional[int], should_smooth: bool) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'

        # Get the prior counts of the next characters using the given prefix
        character_counts = self._trie.get_next_characters(prefix, length=length)

        # Use laplace smoothing
        if should_smooth:
            for c in CHARACTERS:
                character_counts[c] = character_counts.get(c, 0) + 1

        return character_counts

    @classmethod
    def restore(cls, path: str):
        dictionary = cls(max_depth=1)
        dictionary._trie = read_pickle_gz(path)
        dictionary._is_built = True
        dictionary._max_depth = dictionary._trie.max_depth
        return dictionary

    def save(self, path: str):
        save_pickle_gz(self._trie, path)
