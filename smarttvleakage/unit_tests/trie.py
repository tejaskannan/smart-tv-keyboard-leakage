import os
import unittest

from smarttvleakage.dictionary.trie import Trie
from smarttvleakage.utils.file_utils import read_pickle_gz


class TrieTests(unittest.TestCase):

    def test_get_empty(self):
        trie = Trie(max_depth=20)

        self.assertEqual(trie.get_next_characters('test', length=None), dict())

        trie.add_string('random', count=1)

        next_characters = trie.get_next_characters('other', length=None)
        self.assertEqual(len(next_characters), 0)

    def test_add_get_single(self):
        trie = Trie(max_depth=20)

        trie.add_string('warriors', count=1)

        next_characters = trie.get_next_characters('warrior', length=None)
        self.assertEqual(next_characters, { 's': 1 })

    def test_add_get_two(self):
        trie = Trie(max_depth=20)

        trie.add_string('this', count=1)
        trie.add_string('that', count=2)

        next_characters = trie.get_next_characters('th', length=None)
        self.assertEqual(next_characters, { 'a': 2, 'i': 1 })

        next_characters = trie.get_next_characters('thi', length=None)
        self.assertEqual(next_characters, { 's': 1 })

        next_characters = trie.get_next_characters('t', length=None)
        self.assertEqual(next_characters, { 'h': 3 })

        self.assertEqual(trie.get_num_nodes(), 6)

    def test_add_get_five(self):
        trie = Trie(max_depth=20)

        trie.add_string('pistons', count=1)
        trie.add_string('pelicans', count=1)
        trie.add_string('deer', count=1)
        trie.add_string('desert', count=1)
        trie.add_string('dessert', count=1)

        next_characters = trie.get_next_characters('', length=None)
        self.assertEqual(next_characters, { 'p': 2, 'd': 3 })

        next_characters = trie.get_next_characters('p', length=None)
        self.assertEqual(next_characters, { 'i': 1, 'e': 1 })

        next_characters = trie.get_next_characters('de', length=None)
        self.assertEqual(next_characters, { 'e': 1, 's': 2 })

    def test_add_get_length(self):
        trie = Trie(max_depth=20)

        trie.add_string('pistons', count=1)
        trie.add_string('pelicans', count=1)
        trie.add_string('deer', count=1)
        trie.add_string('desert', count=1)
        trie.add_string('dessert', count=1)

        next_characters = trie.get_next_characters('', length=7)
        self.assertEqual(next_characters, { 'p': 1, 'd': 1 })

        next_characters = trie.get_next_characters('p', length=7)
        self.assertEqual(next_characters, { 'i': 1 })

        next_characters = trie.get_next_characters('de', length=6)
        self.assertEqual(next_characters, { 's': 1 })

        next_characters = trie.get_next_characters('de', length=4)
        self.assertEqual(next_characters, { 'e': 1 })

    def test_score(self):
        trie = Trie(max_depth=20)

        trie.add_string('pelican', count=1)
        trie.add_string('pelicans', count=1)
        trie.add_string('deer', count=1)
        trie.add_string('desert', count=2)
        trie.add_string('dessert', count=1)

        score = trie.get_score_for_string('deer', should_aggregate=True)
        self.assertAlmostEqual(score, (1.0 / 6.0))

        score = trie.get_score_for_string('de', should_aggregate=True)
        self.assertAlmostEqual(score, (4.0 / 6.0))

        score = trie.get_score_for_string('p', should_aggregate=True)
        self.assertAlmostEqual(score, (2.0 / 6.0))

        score = trie.get_score_for_string('', should_aggregate=True)
        self.assertAlmostEqual(score, 1.0)

        score = trie.get_score_for_string('de', should_aggregate=False)
        self.assertAlmostEqual(score, 0.0)

        score = trie.get_score_for_string('pelican', should_aggregate=False)
        self.assertAlmostEqual(score, (1.0 / 6.0))

        score = trie.get_score_for_string('pelican', should_aggregate=True)
        self.assertAlmostEqual(score, (2.0 / 6.0))

    def test_add_get_max_depth(self):
        trie = Trie(max_depth=5)

        trie.add_string('deer', count=1)
        trie.add_string('desert', count=1)
        trie.add_string('deserts', count=1)

        next_characters = trie.get_next_characters('de', length=None)
        self.assertEqual(next_characters, { 'e': 1, 's': 2 })

        next_characters = trie.get_next_characters('deser', length=None)
        self.assertEqual(next_characters, { 't': 2, 's': 1})

        next_characters = trie.get_next_characters('desert', length=None)
        self.assertEqual(next_characters, { 't': 2, 's': 1 })


    #def test_dictionary(self):
    #    trie = read_pickle_gz('/local/dictionaries/wikipedia.pkl.gz')

    #    next_characters = trie.get_next_characters('peopl', length=6)
    #    print(next_characters)


if __name__ == '__main__':
    unittest.main()
