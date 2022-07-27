import unittest
from typing import List

from smarttvleakage.search_with_autocomplete import get_words_from_moves_autocomplete
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode
from smarttvleakage.dictionary import EnglishDictionary


print('Loading dictionary...')
dictionary = EnglishDictionary.restore(path='/local/dictionaries/wikipedia.pkl.gz')
graph = MultiKeyboardGraph()
print('Finished.')


class SearchWithAutocomplete(unittest.TestCase):

    def test_about(self):
        move_seq = [1, 6, 1, 0, 0]
        num_results = 4
        results = self.get_top_k_results(move_seq=move_seq, top_k=num_results)
        self.assertTrue('about' in results)

    def test_pillar(self):
        move_seq = [9, 3, 3, 0, 1, 0]
        num_results = 1
        results = self.get_top_k_results(move_seq=move_seq, top_k=num_results)
        self.assertTrue('pillar' in results)

    def test_other(self):
        move_seq = [8, 5, 1, 0, 0]
        num_results = 1
        results = self.get_top_k_results(move_seq=move_seq, top_k=num_results)
        self.assertTrue('other' in results)

    def test_people(self):
        move_seq = [9, 1, 0, 0, 0, 0]
        num_results = 4
        results = self.get_top_k_results(move_seq=move_seq, top_k=num_results)
        self.assertTrue('people' in results)

    def get_top_k_results(self, move_seq: List[int], top_k: int) -> List[str]:
        results: List[str] = []
        iterator = get_words_from_moves_autocomplete(num_moves=move_seq,
                                                     graph=graph,
                                                     dictionary=dictionary,
                                                     max_num_results=None)

        for idx, (string, score, _) in enumerate(iterator):
            if idx >= top_k:
                break

            results.append(string)

        return results


if __name__ == '__main__':
    unittest.main()
