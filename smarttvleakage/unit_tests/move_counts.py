import unittest
import string
from typing import List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE, APPLETV_SEARCH_ALPHABET
from smarttvleakage.graphs.keyboard_graph import APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL, APPLETV_PASSWORD_CAPS
from smarttvleakage.utils.constants import KeyboardType, Direction

# Load the samsung_graphs once globally
samsung_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)
appletv_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.APPLE_TV_SEARCH)
appletv_password_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.APPLE_TV_PASSWORD)


class SamsungGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=0,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['3', 'g', 'r', 'v', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_a_four_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['3', '<DELETEALL>', '*', '-', '<RIGHT>', '@', 'g', 'r', 'v', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_u_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='u',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['6', '8', 'h', 'k', 'm', 'o', 't']
        self.list_equal(neighbors, expected)

    def test_space_one(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=1,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['<SETTINGS>', '<WWW>', 'c']
        self.list_equal(neighbors, expected)

    def test_c_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='c',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['z', 's', 'e', 'f', 'b', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_c_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='c',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['z', 's', 'e', 'f', 'b', '<SETTINGS>', '<WWW>']
        self.list_equal(neighbors, expected)

    def test_q_eight(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=8,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['8', 'o', 'k', 'm']
        self.list_equal(neighbors, expected)

    def test_q_eight_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=8,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['8', 'o', 'k', 'm', '.', '<LEFT>']
        self.list_equal(neighbors, expected)

    def test_z_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['q', 's', 'c']
        self.list_equal(neighbors, expected)

    def test_z_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['q', 's', 'c', '<CHANGE>', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_z_two_shortcuts_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['q', 's', 'c', '<CHANGE>', '<SPACE>', '<DONE>']
        self.list_equal(neighbors, expected)

    def test_dash_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='-',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['?', '@', '*', '<DOWN>', '<DONE>', '<CANCEL>']
        self.list_equal(neighbors, expected)

    def test_dash_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='-',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['?', '@', '*', '<DOWN>', '<DONE>', '<RETURN>', '<CANCEL>']
        self.list_equal(neighbors, expected)

    def test_change_special_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<CHANGE>',
                                                          num_moves=4,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['3', '$', ':', '+', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_change_special_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<CHANGE>',
                                                          num_moves=4,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['3', '$', ':', '+', '<SETTINGS>', '0', ')', '<<', '<CENT>', '<RIGHT>']
        self.list_equal(neighbors, expected)

    def test_space_special_one(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=1,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['<MULT>']
        self.list_equal(neighbors, expected)

    def test_space_special_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=2,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['+', ':', '<DIV>']
        self.list_equal(neighbors, expected)

    def test_space_special_two_shortcuts(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<SPACE>',
                                                          num_moves=2,
                                                          mode=SAMSUNG_SPECIAL_ONE,
                                                          use_shortcuts=True,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['+', ':', '<DIV>', '-', '\\', '<COM>', '<LANGUAGE>']
        self.list_equal(neighbors, expected)

    def test_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=3,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['*','<DELETEALL>','<DONE>','x','r','3','d', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_wraparound_1(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=5,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['p','5','y','g','v','@','-','0','<RIGHT>','<SPACE>']
        self.list_equal(neighbors, expected)

    def test_wraparound_2(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='p',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['i','<BACK>','9','l','?','@','*']
        self.list_equal(neighbors, expected)

    def test_3_numeric(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='3',
                                                          num_moves=6,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)

        neighbors = list(filter(lambda k: k in string.digits, neighbors))
        expected = ['0', '9']
        self.assertEqual(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


class SamsungDirectionMoveCounts(unittest.TestCase):

    def test_q_5h_1v(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=6,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=[Direction.HORIZONTAL] * 5 + [Direction.VERTICAL])
        expected = ['6', 'h']
        self.assertEqual(neighbors, expected)

    def test_q_4h_1v(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=5,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=[Direction.HORIZONTAL] * 4 + [Direction.VERTICAL])
        expected = ['5', 'g']
        self.assertEqual(neighbors, expected)

    def test_q_4h_2v(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=6,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=[Direction.HORIZONTAL] * 4 + [Direction.VERTICAL] * 2)
        expected = ['b']
        self.assertEqual(neighbors, expected)

    def test_q_4h_2v_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=6,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=[Direction.HORIZONTAL] * 4 + [Direction.VERTICAL] * 2)
        expected = ['<UP>', 'b']
        self.assertEqual(neighbors, expected)

    def test_a_3h_1v_wraparound(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=[Direction.HORIZONTAL] * 3 + [Direction.VERTICAL])
        expected = ['*', '-', 'r', 'v']
        self.assertEqual(neighbors, expected)

    def test_g_4h_1v(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='g',
                                                          num_moves=5,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=[Direction.HORIZONTAL] * 4 + [Direction.VERTICAL])
        expected = list(sorted(['.', 'o', 'q', 'z']))
        self.assertEqual(neighbors, expected)


class AppleTVGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=0,
                                                          mode=APPLETV_SEARCH_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_a_one(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=1,
                                                          mode=APPLETV_SEARCH_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['<SPACE>', 'b']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='a',
                                                          num_moves=4,
                                                          mode=APPLETV_SEARCH_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['e']
        self.list_equal(neighbors, expected)

    def test_m_three(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='m',
                                                          num_moves=3,
                                                          mode=APPLETV_SEARCH_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['p', 'j']
        self.list_equal(neighbors, expected)

    def test_z_one(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=1,
                                                          mode=APPLETV_SEARCH_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['<BACK>', 'y']
        self.list_equal(neighbors, expected)

    def test_z_two(self):
        neighbors = appletv_graph.get_keys_for_moves_from(start_key='z',
                                                          num_moves=2,
                                                          mode=APPLETV_SEARCH_ALPHABET,
                                                          use_shortcuts=False,
                                                          use_wraparound=False,
                                                          directions=Direction.ANY)
        expected = ['x']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


class AppleTVPasswordGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='a',
                                                                   num_moves=0,
                                                                   mode=APPLETV_PASSWORD_STANDARD,
                                                                   use_shortcuts=False,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_m_one(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='m',
                                                                   num_moves=1,
                                                                   mode=APPLETV_PASSWORD_STANDARD,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = ['n', 'l', '.']
        self.list_equal(neighbors, expected)

    def test_r_four(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='r',
                                                                   num_moves=4,
                                                                   mode=APPLETV_PASSWORD_STANDARD,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = ['v', 'n', '^', '_']
        self.list_equal(neighbors, expected)

    def test_R_four(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='R',
                                                                   num_moves=4,
                                                                   mode=APPLETV_PASSWORD_CAPS,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = ['V', 'N', '^', '_']
        self.list_equal(neighbors, expected)

    def test_equals_three(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='=',
                                                                   num_moves=3,
                                                                   mode=APPLETV_PASSWORD_SPECIAL,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = [';', '?', '0', '5']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


if __name__ == '__main__':
    unittest.main()
