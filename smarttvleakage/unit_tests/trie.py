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

    def test_add_get_max_depth_length(self):
        trie = Trie(max_depth=5)

        trie.add_string('deer', count=1)
        trie.add_string('desert', count=1)
        trie.add_string('deserts', count=1)

        next_characters = trie.get_next_characters('de', length=4)
        self.assertEqual(next_characters, { 'e': 1 })

        next_characters = trie.get_next_characters('deser', length=6)  # Beyond 5, we no longer know the lengths of the strings
        self.assertEqual(next_characters, { 't': 1, 's': 1 })

        next_characters = trie.get_next_characters('desert', length=7)
        self.assertEqual(next_characters, { 't': 1, 's': 1 })

    def test_get_words_for_prefixes(self):
        trie = Trie(max_depth=20)

        trie.add_string('pelican', count=1)
        trie.add_string('pelicans', count=3)
        trie.add_string('deer', count=1)
        trie.add_string('desert', count=4)
        trie.add_string('dessert', count=2)
        trie.add_string('warriors', count=10)
        trie.add_string('wash', count=6)
        trie.add_string('49ers', count=7)

        results = list(trie.get_words_for(prefixes=['pel', 'de'], max_num_results=6, min_length=None, max_count_per_prefix=None))
        words = list(map(lambda w: w[0], results))
        expected = ['desert', 'pelicans', 'dessert', 'deer', 'pelican']
        self.assertEqual(words, expected)

        results = list(trie.get_words_for(prefixes=['pel', 'de'], max_num_results=6, min_length=5, max_count_per_prefix=None))
        words = list(map(lambda w: w[0], results))
        expected = ['desert', 'pelicans', 'dessert', 'pelican']
        self.assertEqual(words, expected)

        results = list(trie.get_words_for(prefixes=['pel', 'de'], max_num_results=6, min_length=None, max_count_per_prefix=2))
        words = list(map(lambda w: w[0], results))
        expected = ['desert', 'pelicans', 'dessert', 'pelican']
        self.assertEqual(words, expected)

        results = list(trie.get_words_for(prefixes=['pel', 'de'], max_num_results=3, min_length=None, max_count_per_prefix=None))
        words = list(map(lambda w: w[0], results))
        expected = ['desert', 'pelicans', 'dessert']
        self.assertEqual(words, expected)

        results = list(trie.get_words_for(prefixes=['wa', 'de'], max_num_results=3, min_length=None, max_count_per_prefix=None))
        words = list(map(lambda w: w[0], results))
        expected = ['warriors', 'wash', 'desert']
        self.assertEqual(words, expected)

        results = list(trie.get_words_for(prefixes=['wa', 'de', 'pel'], max_num_results=10, min_length=5, max_count_per_prefix=1))
        words = list(map(lambda w: w[0], results))
        expected = ['warriors', 'desert', 'pelicans']
        self.assertEqual(words, expected)

    def test_add_get_spaces(self):
        trie = Trie(max_depth=12)

        trie.add_string('ted lasso', count=1)
        trie.add_string('ted 3', count=1)
        trie.add_string('tedx', count=1)
        trie.add_string('warriors', count=4)

        next_characters = trie.get_next_characters('ted', length=None)
        self.assertEqual(next_characters, { ' ': 2, 'x': 1 })

        next_characters = trie.get_next_characters('ted ', length=None)
        self.assertEqual(next_characters, { 'l': 1, '3': 1 })

    def test_add_get_unknown(self):
        trie = Trie(max_depth=12)

        trie.add_string('ted lasso', count=1)
        trie.add_string('ted 3', count=1)
        trie.add_string('tedx', count=1)
        trie.add_string('warriors', count=4)

        next_characters = trie.get_next_characters('ted', length=None)
        self.assertEqual(next_characters, { ' ': 2, 'x': 1 })

        next_characters = trie.get_next_characters('tedr', length=None)
        self.assertEqual(next_characters, { })

    def test_get_score_for_prefixes(self):
        trie = Trie(max_depth=12)

        trie.add_string('ted lasso', count=1)
        trie.add_string('ted 3', count=1)
        trie.add_string('tedx', count=1)
        trie.add_string('warriors', count=4)

        score = trie.get_score_for_prefix('ted', min_length=3)
        self.assertAlmostEqual(score, (3.0 / 7.0))

        score = trie.get_score_for_prefix('ted', min_length=5)
        self.assertAlmostEqual(score, (2.0 / 7.0))

        score = trie.get_score_for_prefix('ted', min_length=6)
        self.assertAlmostEqual(score, (1.0 / 7.0))

    #def test_dictionary(self):
    #    trie = read_pickle_gz('/local/dictionaries/wikipedia.pkl.gz')

    #    next_characters = trie.get_next_characters('peopl', length=6)
    #    print(next_characters)


if __name__ == '__main__':
    unittest.main()
