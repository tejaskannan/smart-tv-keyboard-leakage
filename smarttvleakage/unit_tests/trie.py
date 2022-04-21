import os
import unittest

from smarttvleakage.dictionary.trie import Trie


class TrieTests(unittest.TestCase):

    def test_get_empty(self):
        trie = Trie(max_depth=20)

        self.assertEqual(trie.get_next_characters('test'), dict())

        trie.add_string('random')

        next_characters = trie.get_next_characters('other')
        self.assertEqual(len(next_characters), 0)

    def test_add_get_single(self):
        trie = Trie(max_depth=20)

        trie.add_string('warriors')

        next_characters = trie.get_next_characters('warrior')
        self.assertEqual(next_characters, { 's': 1 })

    def test_add_get_two(self):
        trie = Trie(max_depth=20)

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
        trie = Trie(max_depth=20)

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

    def test_add_get_max_depth(self):
        trie = Trie(max_depth=5)

        trie.add_string('deer')
        trie.add_string('desert')
        trie.add_string('deserts')

        next_characters = trie.get_next_characters('de')
        self.assertEqual(next_characters, { 'e': 1, 's': 2 })

        next_characters = trie.get_next_characters('deser')
        self.assertEqual(next_characters, { 't': 2, 's': 1})

        next_characters = trie.get_next_characters('desert')
        self.assertEqual(next_characters, { 't': 2, 's': 1 })

    #def test_save_restore_five(self):
    #    trie0 = Trie()

    #    trie0.add_string('pistons')
    #    trie0.add_string('pelicans')
    #    trie0.add_string('deer')
    #    trie0.add_string('desert')
    #    trie0.add_string('dessert')

    #    trie0.save('test.jsonl.gz')

    #    trie1 = Trie.restore('test.jsonl.gz')
    #    next_characters = trie1.get_next_characters('')
    #    self.assertEqual(next_characters, { 'p': 2, 'd': 3 })

    #    next_characters = trie1.get_next_characters('p')
    #    self.assertEqual(next_characters, { 'i': 1, 'e': 1 })

    #    next_characters = trie1.get_next_characters('de')
    #    self.assertEqual(next_characters, { 'e': 1, 's': 2 })

    #    os.remove('test.jsonl.gz')


if __name__ == '__main__':
    unittest.main()
