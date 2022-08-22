import unittest

from smarttvleakage.utils.transformations import capitalization_combinations, get_string_from_keys
from smarttvleakage.utils.ngrams import create_ngrams, split_ngram, prepend_start_characters


class Transformations(unittest.TestCase):

    def test_capitalizations(self):
        string = '5fGE'
        expected = { '5fGE', '5FGE', '5FgE', '5FGe', '5Fge', '5fGe', '5fgE', '5fge' }
        result = capitalization_combinations(string)

        self.assertEqual(expected, result)

    def test_get_string_from_keys_simple(self):
        keys = ['h', 'e', 'l', 'l', 'o']
        result = get_string_from_keys(keys)

        self.assertEqual(result, 'hello')

    def test_get_string_from_keys_backspace_end(self):
        keys = ['4', '9', 'e', 'r', 's', '<BACK>']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '49er')

    def test_get_string_from_keys_backspace_middle(self):
        keys = ['4', '9', 'e', 'r', 's', '<BACK>', '5']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '49er5')

    def test_get_string_from_keys_caps(self):
        keys = ['4', '9', 'e', 'r', '<CAPS>', 's', 's']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '49erSs')

    def test_get_string_from_keys_caps_lock(self):
        keys = ['4', '<CAPS>', '<CAPS>', '9', 'e', 'r', '<CAPS>', 's']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '49ERs')

    def test_get_string_from_keys_caps_lock_backspace(self):
        keys = ['4', '<CAPS>', '<CAPS>', '<BACK>', '9', 'e', 'r', '<CAPS>', 's']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '9ERs')

    def test_get_string_from_keys_caps_translation(self):
        keys = ['4', '<CAPS>', '<SPACE>', '9', 'e', 'r', '<CAPS>', 's']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '4 9erS')

    def test_get_string_from_keys_unprinted(self):
        keys = ['4', '<NEXT>', '9', 'e', 'r', 's']
        result = get_string_from_keys(keys)

        self.assertEqual(result, '49ers')


class TestNgrams(unittest.TestCase):

    def test_1grams(self):
        string = '49ers'
        self.assertEqual(list(create_ngrams(string, 1)), ['4', '9', 'e', 'r', 's'])

    def test_2grams(self):
        string = '49ers'
        self.assertEqual(list(create_ngrams(string, 2)), ['<S>4', '49', '9e', 'er', 'rs', 's<E>'])

    def test_3grams(self):
        string = 'lance'
        self.assertEqual(list(create_ngrams(string, 3)), ['<S><S>l', '<S>la', 'lan', 'anc', 'nce', 'ce<E>'])

    def test_4grams(self):
        string = 'lance'
        self.assertEqual(list(create_ngrams(string, 4)), ['<S><S><S>l', '<S><S>la', '<S>lan', 'lanc', 'ance', 'nce<E>'])

    def test_4grams_longer(self):
        string = 'the'
        self.assertEqual(list(create_ngrams(string, 4)), ['<S><S><S>t', '<S><S>th', '<S>the', 'the<E>'])

    def test_4grams_longer_2(self):
        string = 'a'
        self.assertEqual(list(create_ngrams(string, 4)), ['<S><S><S>a', '<S><S>a<E>'])

    def test_split_standard_1(self):
        prefix, suffix = split_ngram('the')
        self.assertEqual(prefix, 'th')
        self.assertEqual(suffix, 'e')

    def test_split_standard_2(self):
        prefix, suffix = split_ngram('49ers')
        self.assertEqual(prefix, '49er')
        self.assertEqual(suffix, 's')

    def test_split_start(self):
        prefix, suffix = split_ngram('<S>49ers')
        self.assertEqual(prefix, '<S>49er')
        self.assertEqual(suffix, 's')

    def test_split_end(self):
        prefix, suffix = split_ngram('49ers<E>')
        self.assertEqual(prefix, '49ers')
        self.assertEqual(suffix, '<E>')

    def test_split_start_end(self):
        prefix, suffix = split_ngram('<S><S>a<E>')
        self.assertEqual(prefix, '<S><S>a')
        self.assertEqual(suffix, '<E>')

    def test_prepend_none(self):
        self.assertEqual(prepend_start_characters('the', 3), 'the')
        self.assertEqual(prepend_start_characters('49ers', 4), '49ers')

    def test_prepend_single(self):
        self.assertEqual(prepend_start_characters('th', 3), '<S>th')
        self.assertEqual(prepend_start_characters('49er', 5), '<S>49er') 

    def test_prepend_many(self):
        self.assertEqual(prepend_start_characters('th', 5), '<S><S><S>th')
        self.assertEqual(prepend_start_characters('49er', 6), '<S><S>49er') 


if __name__ == '__main__':
    unittest.main()

