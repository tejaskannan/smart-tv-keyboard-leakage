import unittest

from smarttvleakage.dictionary.trie import Trie


class TrieTests(unittest.TestCase):

    def test_get_empty(self):
        trie = Trie()

        self.assertEqual(trie.get_next_characters('test'), dict())

        trie.add_string('random')

        next_characters = trie.get_next_characters('other')
        self.assertEqual(next_characters, dict())

    def test_add_get_single(self):
        trie = Trie()

        trie.add_string('warriors')

        next_characters = trie.get_next_characters('warrior')
        self.assertEqual(next_characters, { 's': 1 })

    def test_add_get_two(self):
        trie = Trie()

        trie.add_string('this')
        trie.add_string('that')

        next_characters = trie.get_next_characters('th')
        self.assertEqual(next_characters, { 'a': 1, 'i': 1 })

        next_characters = trie.get_next_characters('thi')
        self.assertEqual(next_characters, { 's': 1 })

        next_characters = trie.get_next_characters('t')
        self.assertEqual(next_characters, { 'h': 2 })

        self.assertEqual(trie.get_num_nodes(), 6)

    def test_add_get_five(self):
        trie = Trie()

        trie.add_string('pistons')
        trie.add_string('pelicans')
        trie.add_string('deer')
        trie.add_string('desert')
        trie.add_string('dessert')

        next_characters = trie.get_next_characters('')
        self.assertEqual(next_characters, { 'p': 2, 'd': 3 })

        next_characters = trie.get_next_characters('p')
        self.assertEqual(next_characters, { 'i': 1, 'e': 1 })

        next_characters = trie.get_next_characters('de')
        self.assertEqual(next_characters, { 'e': 1, 's': 2 })


if __name__ == '__main__':
    unittest.main()
