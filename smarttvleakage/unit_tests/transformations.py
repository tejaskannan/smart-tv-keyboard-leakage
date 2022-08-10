import unittest

from smarttvleakage.utils.transformations import capitalization_combinations, get_string_from_keys
from smarttvleakage.utils.ngrams import create_ngrams


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
        self.assertEqual(list(create_ngrams(string, 2)), ['49', '9e', 'er', 'rs'])

    def test_3grams(self):
        string = 'lance'
        self.assertEqual(list(create_ngrams(string, 3)), ['lan', 'anc', 'nce'])

    def test_4grams(self):
        string = 'lance'
        self.assertEqual(list(create_ngrams(string, 4)), ['lanc', 'ance'])

    def test_4grams_longer(self):
        string = 'the'
        self.assertEqual(list(create_ngrams(string, 4)), [])


if __name__ == '__main__':
    unittest.main()

