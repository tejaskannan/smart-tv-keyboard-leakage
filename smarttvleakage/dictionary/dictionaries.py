import string
import os.path
import io
import gzip
import time
import sqlite3
import numpy as np
from collections import Counter, defaultdict, namedtuple, deque
from typing import Dict, List, Optional, Iterable, Tuple, Any

from smarttvleakage.utils.ngrams import create_ngrams, prepend_start_characters, split_ngram
from smarttvleakage.utils.constants import BIG_NUMBER, START_CHAR, END_CHAR, SMALL_NUMBER
from smarttvleakage.utils.credit_card_detection import validate_credit_card_number
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, save_pickle_gz
from smarttvleakage.dictionary.trie import Trie


CHANGE_KEYS = frozenset({ '<CHANGE>', '<ABC>', '<abc>', '<SPECIAL>' })
UNPRINTED_CHARACTERS = frozenset({ '<CHANGE>', '<RIGHT>', '<LEFT>', '<UP>', '<DOWN>', '<BACK>', '<CAPS>', '<NEXT>' })
SELECT_SOUND_KEYS = frozenset({ '<CHANGE>', '<CAPS>', '<NEXT>', '<SPACE>', '<LEFT>', '<RIGHT>', '<UP>', '<DOWN>', '<LANGUAGE>', '<DONE>', '<CANCEL>'})
DELETE_SOUND_KEYS = frozenset({ '<BACK>', '<DELETEALL>' })


CAPS = '<CAPS>'
CHANGE = '<CHANGE>'
BACKSPACE = '<BACK>'
NEXT = '<NEXT>'
SPACE = '<SPACE>'
DONE = '<DONE>'
DELETE_ALL = '<DELETEALL>'
SMOOTH_DELTA = 0.5


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
ProjectedState = namedtuple('ProjectedState', ['string', 'score', 'depth'])


def reverse_string(string: str) -> str:
    return ''.join(list(reversed(string)))


class CharacterDictionary:

    def __init__(self):
        self._characters: List[str] = []

    @property
    def characters(self) -> List[str]:
        return self._characters

    def is_valid(self, _: str) -> bool:
        return True

    def projected_remaining_log_prob(self, prefix: str, length: int) -> float:
        return 0.0

    def set_characters(self, characters: List[str]):
        self._characters = characters

    def smooth_character_counts(self, prefix: str, counts: Dict[str, int]) -> Dict[str, int]:
        return counts

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert len(self.characters) > 0, 'Must call set_characters() first'
        raise NotImplementedError()


class UniformDictionary(CharacterDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        return { REVERSE_CHARACTER_TRANSLATION.get(char, char): 1 for char in self.characters }


class NumericDictionary(CharacterDictionary):

    def is_valid(self, string: str) -> bool:
        try:
            int_val = int(string)
            return True
        except ValueError:
            return False

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        counts = {c: 1 for c in string.digits}
        counts[END_CHAR] = 1
        total_count = sum(counts.values())
        return {c: (count / total_count) for c, count in counts.items()}


class ExpDateDictionary(CharacterDictionary):

    def is_valid(self, string: str) -> bool:
        try:
            int_val = int(string)
            return True
        except ValueError:
            return False

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        months = list(map(lambda x : '0' + str(x), range(1, 10))) + ['10', '11', '12']
        years = list(map(str, range(22, 40)))
        dates = [m + y for m in months for y in years]

        counts = {}
        for c in string.digits:
            count = 0
            for date in dates:
                if date.startswith(prefix + c):
                    count += 1
            counts[c] = count

        counts[END_CHAR] = 1
        total_count = sum(counts.values())
        return {c: (count / total_count) for c, count in counts.items()}


class ExpYearDictionary(NumericDictionary):

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        if len(prefix) == 0:
            return { '2': 0.5, '3': 0.5 }
        elif (len(prefix) == 1) and prefix.startswith('2'):
            return {str(digit): (1.0 / 8.0) for digit in range(2, 10)}
        elif (len(prefix) == 1) and prefix.startswith('3'):
            return {str(digit): (1.0 / 4.0) for digit in range(0, 4)}
        else:
            return super().get_letter_counts(prefix, length)


class CVVDictionary(CharacterDictionary):

    def is_valid(self, string: str) -> bool:
        try:
            int_val = int(string)
            if len(string) in [3, 4]:
                return True
            return False
        except ValueError:
            return False

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        counts = {}
        if len(prefix) < 4:
            for c in string.digits:
                counts[c] = 10
        if len(prefix) > 2:
            counts[END_CHAR] = 10
        total_count = sum(counts.values())
        return {c: (count / total_count) for c, count in counts.items()}


class CreditCardDictionary(NumericDictionary):

    def __init__(self):
        super().__init__()

        self._single_counter: Counter = Counter()
        self._prefix_counter: DefaultDict[str, Counter] = defaultdict(Counter)

        dir_name = os.path.dirname(__file__)
        with open(os.path.join(dir_name, 'cc_bins.txt')) as fin:
            for line in fin:
                prefix = line.strip()

                if len(prefix) == 1:
                    self._single_counter[prefix] += 1
                elif len(prefix) > 1:
                    self._prefix_counter[prefix[0:-1]][prefix[-1]] += 1

    def is_valid(self, string: str) -> bool:
        is_valid = super().is_valid(string)
        return is_valid and validate_credit_card_number(string)

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        """
        Credit Card formats from here: https://www.tokenex.com/blog/ab-what-is-a-bin-bank-identification-number
        """
        length = len(prefix)

        if len(prefix) == 0:
            char_counts = self._single_counter
        elif prefix in self._prefix_counter:
            char_counts = self._prefix_counter[prefix]
        else:
            char_counts = {c: 1 for c in string.digits}

        total_count = sum(char_counts.values())
        return {c: (count / total_count) for c, count in char_counts.items()}


class ZipCodeDictionary(NumericDictionary):

    def __init__(self):
        super().__init__()
        self._trie = Trie(max_depth=6)
        self._is_built = False

    def build(self, path: str, min_count: int, has_counts: bool, should_reverse: bool):
        with open(path, 'r') as fin:
            for line in fin:
                tokens = line.strip().split()
                zip_code = tokens[0]
                population = int(tokens[1])

                if population >= min_count:
                    if should_reverse:
                        zip_code = reverse_string(zip_code)

                    self._trie.add_string(zip_code, count=population, should_index_prefixes=False)

        self._is_build = True

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, float]:
        # Get the prior counts of the next characters using the given prefix
        character_counts = self._trie.get_next_characters(prefix, length=5)  # All zip codes have length 5

        if len(character_counts) == 0:
            return {c: 0.1 for c in string.digits}

        total_count = sum(character_counts.values())
        return {c: (count / total_count) for c, count in character_counts.items()}

    def save(self, path: str):
        data_dict = {
            'trie': self._trie,
            'dict_type': 'zip_code'
        }
        save_pickle_gz(data_dict, path)

    @classmethod
    def restore(cls, serialized: Dict[str, Any]):
        dictionary = cls()
        dictionary._trie = serialized['trie']
        dictionary._is_built = True
        return dictionary


class EnglishDictionary(CharacterDictionary):

    def __init__(self, max_depth: int):
        super().__init__()
        self._trie = Trie(max_depth=max_depth)
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

    def build(self, path: str, min_count: int, has_counts: bool, should_reverse: bool):
        if self._is_built:
            return

        # Read the input words
        string_dictionary = self.parse_dictionary_file(path=path,
                                                       min_count=min_count,
                                                       has_counts=has_counts)

        print('Indexing {} Strings...'.format(len(string_dictionary)))

        # Build the trie
        elapsed = 0.0
        for idx, (word, count) in enumerate(string_dictionary.items()):
            count = max(count, 1)

            if should_reverse:
                word = reverse_string(word)

            start = time.time()
            self._trie.add_string(word, count=count, should_index_prefixes=False)
            end = time.time()

            elapsed += (end - start)

            if (idx + 1) % 100000 == 0:
                print('Completed {} strings. Avg Time / Insert {:.7f}ms'.format(idx + 1, 1000.0 * (elapsed / (idx + 1))), end='\r')

        print()
        self._is_built = True

    def get_words_for(self, prefixes: Iterable[str], max_num_results: int, min_length: Optional[int], max_count_per_prefix: Optional[int]) -> Iterable[Tuple[str, float]]:
        return self._trie.get_words_for(prefixes, max_num_results, min_length=min_length, max_count_per_prefix=max_count_per_prefix)

    def does_contain_prefix(self, prefix: str) -> bool:
        return self._trie.get_node_for(prefix) is not None

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'
        
        # Get the prior counts of the next characters using the given prefix
        character_counts = self._trie.get_next_characters(prefix, length=length)

        # Re-map the character names
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): count for char, count in character_counts.items()}

        # Apply Laplace Smoothing
        for character in self._characters:
            character_counts[character] = character_counts.get(character, 0) + SMOOTH_DELTA

        character_counts[END_CHAR] = character_counts.get(END_CHAR, 0) + SMOOTH_DELTA

        # Normalize the result
        total_count = sum(character_counts.values())
        return {char: (count / total_count) for char, count in character_counts.items()}

    @classmethod
    def restore(cls, serialized: Dict[str, Any]):
        dictionary = cls(max_depth=1)
        dictionary._trie = serialized['trie']
        dictionary._is_built = True
        dictionary._max_depth = dictionary._trie.max_depth
        return dictionary

    def save(self, path: str):
        data_dict = {
            'trie': self._trie,
            'dict_type': 'english'
        }
        save_pickle_gz(data_dict, path)


class NgramDictionary(EnglishDictionary):

    def __init__(self):
        super().__init__(max_depth=0)
        self._is_built = False

        self._counts_per_length: Dict[int, Dict[str, Counter]] = {
            0: defaultdict(Counter),
            1: defaultdict(Counter),
            2: defaultdict(Counter)
        }

        self._ngram_size = 5
        self._total_count = 0

    def build(self, path: str, min_count: int, has_counts: bool, should_reverse: bool):
        if self._is_built:
            return

        # Read the input words
        string_dictionary = self.parse_dictionary_file(path=path,
                                                       min_count=min_count,
                                                       has_counts=has_counts)

        # Build the ngram counters
        for word, count in string_dictionary.items():
            if len(word) == 0:
                continue

            count = max(count, 1)
            length_bucket = self.get_length_bucket(len(word))

            if should_reverse:
                word = reverse_string(word)

            for ngram in create_ngrams(word, self._ngram_size):
                ngram_prefix, ngram_suffix = split_ngram(ngram)
                self._counts_per_length[length_bucket][ngram_prefix][ngram_suffix] += 1

            self._total_count += 1

            if (self._total_count % 10000) == 0:
                print('Completed {} Strings...'.format(self._total_count), end='\r')

        print()
        self._is_built = True

    def get_length_bucket(self, length: int) -> int:
        if length is None: # added to fix error
            return 0
        if length < 6:
            return 0
        elif length < 10:
            return 1
        else:
            return 2

    def get_score_for_string(self, string: str, length: int) -> float:
        letter_freqs = self.get_letter_counts('', length=length)

        score = 0.0
        for idx, character in enumerate(string):
            if character not in letter_freqs:
                return BIG_NUMBER

            prob = letter_freqs[character]
            score -= np.log(prob)

            letter_freqs = self.get_letter_counts(string[0:idx+1], length=length)

        end_prob = self.get_letter_counts(string, length=length).get(END_CHAR, SMALL_NUMBER)
        return score - np.log(end_prob)

    def projected_remaining_log_prob(self, prefix: str, length: int) -> float:
        neg_log_prob = BIG_NUMBER
        max_depth = min(3, length - len(prefix))
        num_to_keep = 3

        frontier = deque()
        init_state = ProjectedState(string=prefix, score=0.0, depth=max_depth)
        frontier.append(init_state)

        while len(frontier) > 0:
            state = frontier.pop()

            if state.depth <= 0 or len(state.string) >= length:
                neg_log_prob = min(neg_log_prob, state.score)
            else:
                next_letter_freq = Counter({c: freq for c, freq in self.get_letter_counts(state.string, length=length).items() if c != END_CHAR})

                for character, freq in next_letter_freq.most_common(num_to_keep):
                    next_state = ProjectedState(string=state.string + character,
                                                score=state.score - np.log(freq),
                                                depth=state.depth - 1)
                    frontier.append(next_state)

        return neg_log_prob

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'

        length_bucket = self.get_length_bucket(length)

        # Look up the string in the counts dictionary
        if len(prefix) <= (self._ngram_size - 1):
            adjusted_prefix = prepend_start_characters(prefix, self._ngram_size - 1)
            character_counts = self._counts_per_length[length_bucket].get(adjusted_prefix, dict())
        else:
            character_counts = self._counts_per_length[length_bucket].get(prefix[-self._ngram_size + 1:], dict())

        # Convert any characters
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): count for char, count in character_counts.items()}

        # Apply Laplace Smoothing and include the end character
        for character in self._characters:
            character_counts[character] = character_counts.get(character, 0) + SMOOTH_DELTA

        character_counts[END_CHAR] = character_counts.get(END_CHAR, 0) + SMOOTH_DELTA

        # Normalize the result
        total_count = sum(character_counts.values())
        character_counts = {char: (count / total_count) for char, count in character_counts.items()}
        return character_counts

    def does_contain_prefix(prefix: str) -> bool:
        return False

    @classmethod
    def restore(cls, serialized: Dict[str, Any]):
        dictionary = cls()
        dictionary._counts_per_length = serialized['ngram']
        dictionary._is_built = True
        return dictionary

    def save(self, path: str):
        data_dict = {
            'ngram': self._counts_per_length,
            'dict_type': 'ngram'
        }
        save_pickle_gz(data_dict, path)


class NgramSQLDictionary(NgramDictionary):

    def __init__(self, db_file: str):
        self._is_built = True

        self._db_file = db_file
        self._conn = sqlite3.connect(db_file)

        self._ngram_size = 5

    def get_letter_counts(self, prefix: str, length: Optional[int]) -> Dict[str, int]:
        assert self._is_built, 'Must call build() first'

        length_bucket = self.get_length_bucket(length)

        # Look up the string in the SQL table
        if len(prefix) <= (self._ngram_size - 1):
            adjusted_prefix = prepend_start_characters(prefix, self._ngram_size - 1)
        else:
            adjusted_prefix = prefix[-self._ngram_size + 1:]

        cursor = self._conn.cursor()
        query_exec = cursor.execute('SELECT suffix, count FROM ngrams WHERE prefix=:prefix AND length_bucket=:length_bucket;', {'prefix': adjusted_prefix, 'length_bucket': length_bucket})
        query_results = query_exec.fetchall()
        character_counts = { char: count for (char, count) in query_results }

        # Convert any characters
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): count for char, count in character_counts.items()}

        # Apply Laplace Smoothing and include the end character
        for character in self._characters:
            character_counts[character] = character_counts.get(character, 0) + SMOOTH_DELTA

        character_counts[END_CHAR] = character_counts.get(END_CHAR, 0) + SMOOTH_DELTA

        # Normalize the result
        total_count = sum(character_counts.values())
        character_counts = {char: (count / total_count) for char, count in character_counts.items()}
        return character_counts

    @classmethod
    def restore(cls, db_file: str):
        dictionary = cls(db_file)
        return dictionary

    def save(self, path: str):
        pass


def restore_dictionary(path: str) -> CharacterDictionary:
    if path == 'uniform':
        return UniformDictionary()
    elif path == 'credit_card':
        return CreditCardDictionary()
    elif path == 'numeric':
        return NumericDictionary()
    elif path == 'exp_date':
        return ExpDateDictionary()
    elif path == 'exp_year':
        return ExpYearDictionary()
    elif path == 'cvv':
        return CVVDictionary()
    elif path.endswith('.db'):
        return NgramSQLDictionary.restore(path)
    else:
        data_dict = read_pickle_gz(path)
        dict_type = data_dict['dict_type']

        if dict_type == 'english':
            return EnglishDictionary.restore(data_dict)
        elif dict_type == 'ngram':
            return NgramDictionary.restore(data_dict)
        elif dict_type == 'zip_code':
            return ZipCodeDictionary.restore(data_dict)
        else:
            raise ValueError('Unknown dictionary type: {}'.format(dict_type))
