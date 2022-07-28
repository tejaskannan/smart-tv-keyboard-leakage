import unittest
from typing import List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE, APPLETV_ALPHABET
from smarttvleakage.utils.constants import SmartTVType


# Load the samsung_graphs once globally
samsung_graph = MultiKeyboardGraph(tv_type=SmartTVType.SAMSUNG)
appletv_graph = MultiKeyboardGraph(tv_type=SmartTVType.APPLE_TV)


class SamsungGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=0,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['3', 'g', 'r', 'v']
        self.list_equal(neighbors, expected)

    def test_a_four_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True)
        expected = ['3', '<DELETEALL>', '*', '-', '<RIGHT>', '@', 'g', 'r', 'v']
        self.list_equal(neighbors, expected)

    def test_u_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='u',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True)
        expected = ['6', '8', 'h', 'k', 'm', 'o', 't']
        self.list_equal(neighbors, expected)

    def test_space_one(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=1,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False)
        expected = ['<SETTINGS>', '<WWW>', 'c']
        self.list_equal(neighbors, expected)

    def test_c_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='c',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['z', 's', 'e', 'f', 'b', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_c_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='c',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False)
        expected = ['z', 's', 'e', 'f', 'b', '<SETTINGS>', '<WWW>']
        self.list_equal(neighbors, expected)


    def test_z_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['q', 's', 'c', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_z_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False)
        expected = ['q', 's', 'c', '<CHANGE>', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_z_two_shortcuts_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True)
        expected = ['q', 's', 'c', '<CHANGE>', '<SPACE>', '<DONE>']
        self.list_equal(neighbors, expected)

    def test_dash_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='-',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['?', '@', '*', '<DOWN>', '<DONE>', '<CANCEL>']
        self.list_equal(neighbors, expected)

    def test_dash_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='-',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False)
        expected = ['?', '@', '*', '<DOWN>', '<RETURN>', '<CANCEL>']
        self.list_equal(neighbors, expected)

    def test_change_special_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<CHANGE>',
                                                          num_moves=4,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['3', '$', ':', '+', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_change_special_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<CHANGE>',
                                                          num_moves=4,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=True)
        expected = ['3', '$', ':', '+', '<SETTINGS>', '0', ')', '<<', '<CENT>', '<RIGHT>']
        self.list_equal(neighbors, expected)

    def test_space_special_one(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=1,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['<MULT>']
        self.list_equal(neighbors, expected)

    def test_space_special_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=2,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['+', ':', '<DIV>']
        self.list_equal(neighbors, expected)

    def test_space_special_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=2,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=True,
                                                          use_wraparound=False)
        expected = ['+', ':', '<DIV>', '-', '\\', '<COM>', '<LANGUAGE>']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


class AppleTVGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=0,
                                                          mode=APPLETV_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_a_one(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=1,
                                                          mode=APPLETV_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['<SPACE>', 'b']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=APPLETV_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['e']
        self.list_equal(neighbors, expected)

    def test_m_three(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='m',
                                                          num_moves=3,
                                                          mode=APPLETV_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['p', 'j']
        self.list_equal(neighbors, expected)

    def test_z_one(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=1,
                                                          mode=APPLETV_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['<BACK>', 'y']
        self.list_equal(neighbors, expected)

    def test_z_two(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=APPLETV_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False)
        expected = ['x']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


if __name__ == '__main__':
    unittest.main()
