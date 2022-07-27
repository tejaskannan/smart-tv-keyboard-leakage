import unittest
from typing import List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode


# Load the graph once globally
graph = MultiKeyboardGraph()


class GraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = graph.get_keys_for_moves_from(start_key='a',
                                                  num_moves=0,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=False)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = graph.get_keys_for_moves_from(start_key='a',
                                                  num_moves=4,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=False)
        expected = ['3', 'g', 'r', 'v']
        self.list_equal(neighbors, expected)

    def test_a_four_wraparound(self):
        neighbors = graph.get_keys_for_moves_from(start_key='a',
                                                  num_moves=4,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=True)
        expected = ['3', '<DELETEALL>', '*', '-', '<RIGHT>', '@', 'g', 'r', 'v']
        self.list_equal(neighbors, expected)

    def test_u_two(self):
        neighbors = graph.get_keys_for_moves_from(start_key='u',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=True)
        expected = ['6', '8', 'h', 'k', 'm', 'o', 't']
        self.list_equal(neighbors, expected)

    def test_space_one(self):
        neighbors = graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                  num_moves=1,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=True,
                                                  use_wraparound=False)
        expected = ['<SETTINGS>', '<WWW>', 'c']
        self.list_equal(neighbors, expected)

    def test_c_two(self):
        neighbors = graph.get_keys_for_moves_from(start_key='c',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=False)
        expected = ['z', 's', 'e', 'f', 'b', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_c_two_shortcuts(self):
        neighbors = graph.get_keys_for_moves_from(start_key='c',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=True,
                                                  use_wraparound=False)
        expected = ['z', 's', 'e', 'f', 'b', '<SETTINGS>', '<WWW>']
        self.list_equal(neighbors, expected)


    def test_z_two(self):
        neighbors = graph.get_keys_for_moves_from(start_key='z',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=False)
        expected = ['q', 's', 'c', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_z_two_shortcuts(self):
        neighbors = graph.get_keys_for_moves_from(start_key='z',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=True,
                                                  use_wraparound=False)
        expected = ['q', 's', 'c', '<CHANGE>', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_z_two_shortcuts_wraparound(self):
        neighbors = graph.get_keys_for_moves_from(start_key='z',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=True,
                                                  use_wraparound=True)
        expected = ['q', 's', 'c', '<CHANGE>', '<SPACE>', '<DONE>']
        self.list_equal(neighbors, expected)

    def test_dash_two(self):
        neighbors = graph.get_keys_for_moves_from(start_key='-',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=False,
                                                  use_wraparound=False)
        expected = ['?', '@', '*', '<DOWN>', '<DONE>', '<CANCEL>']
        self.list_equal(neighbors, expected)

    def test_dash_two_shortcuts(self):
        neighbors = graph.get_keys_for_moves_from(start_key='-',
                                                  num_moves=2,
                                                  mode=KeyboardMode.STANDARD,
                                                  use_shortcuts=True,
                                                  use_wraparound=False)
        expected = ['?', '@', '*', '<DOWN>', '<RETURN>', '<CANCEL>']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


if __name__ == '__main__':
    unittest.main()
