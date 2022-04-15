import string
import os.path
import io
from collections import Counter
from typing import Dict, List

from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, save_pickle_gz
from smarttvleakage.dictionary.trie import Trie


standard_graph = read_json(os.path.join(os.path.dirname(__file__), '..', 'graphs', 'samsung_keyboard.json'))
special_graph = read_json(os.path.join(os.path.dirname(__file__), '..', 'graphs', 'samsung_keyboard_special_1.json'))

CHARACTERS: List[str] = list(sorted(standard_graph.keys())) + list(sorted(special_graph.keys()))
UNPRINTED_CHARACTERS = { '<CHANGE>', '<RIGHT>', '<LEFT>', '<UP>', '<DOWN>', '<WWW>', '<COM>', '<BACK>', '<CAPS>', '<NEXT>' }

CHARACTER_TRANSLATION = {
    '<MULT>': '×',
    '<DIV>': '÷'
}


class CharacterDictionary:

    def get_letter_counts(self, prefix: str) -> Dict[str, int]:
        raise NotImplementedError()


class UniformDictionary(CharacterDictionary):

    def get_letter_counts(self, prefix: str) -> Dict[str, int]:
        return { c: 1 for c in CHARACTERS }


class EnglishDictionary(CharacterDictionary):

    def __init__(self):
        self._trie = Trie()
        self._is_built = False

    def build(self, path: str):
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
                    if len(line) > 0:
                        string_dictionary[line] = 0

        # Build the trie
        for word in string_dictionary.keys():
            self._trie.add_string(word)

        self._is_built = True

    def get_letter_counts(self, prefix: str, should_smooth: bool) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'

        # Get the prior counts of the next characters using the given prefix
        character_counts = self._trie.get_next_characters(prefix)

        # Use laplace smoothing
        if should_smooth:
            for c in CHARACTERS:
                character_counts[c] = character_counts.get(c, 0) + 1

        return character_counts

    @classmethod
    def restore(cls, path: str):
        dictionary = cls()
        dictionary._trie = read_pickle_gz(path)
        dictionary._is_built = True
        return dictionary

    def save(self, path: str):
        save_pickle_gz(self._trie, path)
