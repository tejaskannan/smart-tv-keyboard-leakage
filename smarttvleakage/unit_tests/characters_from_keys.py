import unittest
from smarttvleakage.graph_search import get_characters_from_keys


class CharactersFromKeys(unittest.TestCase):

    def test_single_caps(self):
        keys = ['b', '<CAPS>', 'e', 'd']
        expected = ['b', 'E', 'd']

        self.assertEqual(expected, get_characters_from_keys(keys))

    def test_single_caps_first(self):
        keys = ['<CAPS>', 'b', 'e', 'd']
        expected = ['B', 'e', 'd']

        self.assertEqual(expected, get_characters_from_keys(keys))

    def test_caps_lock(self):
        keys = ['h', 'o', '<CAPS>', '<CAPS>', 'u', 's', '<CAPS>', 'e']
        expected = ['h', 'o', 'U', 'S', 'e']

        self.assertEqual(expected, get_characters_from_keys(keys))

    def test_caps_lock_first(self):
        keys = ['<CAPS>', '<CAPS>', 'h', 'o', '<CAPS>', 'u', 's', 'e']
        expected = ['H', 'O', 'u', 's', 'e']

        self.assertEqual(expected, get_characters_from_keys(keys))

    def test_caps_lock_multiple(self):
        keys = ['r', 'a', '<CAPS>', '<CAPS>', 'c', 'e', '<CAPS>', 'c', '<CAPS>', '<CAPS>', 'a', 'r']
        expected = ['r', 'a', 'C', 'E', 'c', 'A', 'R']

        self.assertEqual(expected, get_characters_from_keys(keys))

    def test_caps_both(self):
        keys = ['r', 'a', '<CAPS>', '<CAPS>', 'c', 'e', '<CAPS>', 'c', '<CAPS>', 'a', 'r']
        expected = ['r', 'a', 'C', 'E', 'c', 'A', 'r']

        self.assertEqual(expected, get_characters_from_keys(keys))

    def test_caps_both_single_first(self):
        keys = ['<CAPS>', 'r', 'a', '<CAPS>', '<CAPS>', 'c', 'e', '<CAPS>', 'c', '<CAPS>', 'a', 'r']
        expected = ['R', 'a', 'C', 'E', 'c', 'A', 'r']

        self.assertEqual(expected, get_characters_from_keys(keys))


if __name__ == '__main__':
    unittest.main()
