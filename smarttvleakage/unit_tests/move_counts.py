import unittest
import string
from typing import List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE, APPLETV_SEARCH_ALPHABET
from smarttvleakage.graphs.keyboard_graph import APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL, APPLETV_PASSWORD_CAPS, SAMSUNG_SPECIAL_TWO
from smarttvleakage.graphs.keyboard_graph import ABC_STANDARD, ABC_CAPS, ABC_SPECIAL
from smarttvleakage.utils.constants import KeyboardType, Direction

# Load the samsung_graphs once globally
samsung_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)
appletv_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.APPLE_TV_SEARCH)
appletv_password_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.APPLE_TV_PASSWORD)
abc_graph = MultiKeyboardGraph(keyboard_type=KeyboardType.ABC)


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

    def test_done_one(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=1,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['<CANCEL>', '!', '<RETURN>', '<LANGUAGE>']
        self.list_equal(neighbors, expected)

    def test_done_two(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['<DELETEALL>', '*', '@', '-', 'a', '<CHANGE>', '<RIGHT>']
        self.list_equal(neighbors, expected)

    def test_done_three(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=3,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['<BACK>', '<CAPS>', '^', '~', '<UP>', '<DOWN>', 'q', 'z', 's']
        self.list_equal(neighbors, expected)

    def test_done_four(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=4,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['0', 'p', 'l', '?', '<LEFT>', '1', 'w', 'd', 'x', '<SETTINGS>']
        self.list_equal(neighbors, expected)

    def test_done_five(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=5,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['9', 'o', 'k', '.', '/', '2', 'e', 'f', 'c', '<SPACE>']
        self.list_equal(neighbors, expected)

    def test_done_six(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=6,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['8', 'i', 'j', ',', '<COM>', '3', 'r', 'g', 'v', '<SPACE>', '<WWW>']
        self.list_equal(neighbors, expected)

    def test_done_seven(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<DONE>',
                                                          num_moves=7,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['7', 'u', 'h', 'm', '4', 't', 'b', '<WWW>']
        self.list_equal(neighbors, expected)

    def test_done_seven_to(self):
        neighbors = samsung_graph.get_keys_for_moves_to(end_key='<DONE>',
                                                        num_moves=7,
                                                        mode=SAMSUNG_STANDARD,
                                                        use_shortcuts=True,
                                                        use_wraparound=True)
        expected = ['7', 'u', 'h', 'n', 'm', '4', 't', '<WWW>', '6', '5']  # TODO: Investigate `m` being here (I can't find a path that makes sense)
        self.list_equal(neighbors, expected)

    def test_6_seven(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='6',
                                                          num_moves=7,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['<RETURN>', '*', '@', '?', '/', '<CHANGE>', 'a', 'x', '<SPACE>', '<DONE>']
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

    def test_special_2(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='<NEXT>',
                                                          num_moves=3,
                                                          mode=SAMSUNG_SPECIAL_TWO,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=Direction.ANY)
        expected = ['_', '<BULLET>']
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

    def test_q_left_up(self):
        neighbors = samsung_graph.get_keys_for_moves_from(start_key='q',
                                                          num_moves=2,
                                                          mode=SAMSUNG_STANDARD,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=[Direction.LEFT, Direction.UP])
        expected = ['<CAPS>']
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



class ABCGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='a',
                                                      num_moves=0,
                                                      mode=ABC_STANDARD,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['a']
        self.list_equal(neighbors, expected)

    def test_a_one(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='a',
                                                      num_moves=1,
                                                      mode=ABC_STANDARD,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['b', 'h']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='a',
                                                      num_moves=4,
                                                      mode=ABC_STANDARD,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['<SPACE>', 'e', 'k', 'q', 'w']
        self.list_equal(neighbors, expected)

    def test_m_three(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='m',
                                                      num_moves=3,
                                                      mode=ABC_STANDARD,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['<BACK>', '<CAPS>', '\'', 'z', 'r', 'j', 'd']
        self.list_equal(neighbors, expected)

    def test_e_one(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='e',
                                                      num_moves=1,
                                                      mode=ABC_STANDARD,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['d', 'f', 'l']
        self.list_equal(neighbors, expected)

    def test_p_two(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='p',
                                                      num_moves=2,
                                                      mode=ABC_STANDARD,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['h', 'v', 'x', 'j', 'b', 'r']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


class ABCCapsGraphMoveCounts(unittest.TestCase):

    def test_a_zero(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='A',
                                                      num_moves=0,
                                                      mode=ABC_CAPS,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['A']
        self.list_equal(neighbors, expected)

    def test_a_one(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='A',
                                                      num_moves=1,
                                                      mode=ABC_CAPS,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['B', 'H']
        self.list_equal(neighbors, expected)

    def test_a_four(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='A',
                                                      num_moves=4,
                                                      mode=ABC_CAPS,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['<SPACE>', 'E', 'K', 'Q', 'W']
        self.list_equal(neighbors, expected)

    def test_m_three(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='M',
                                                      num_moves=3,
                                                      mode=ABC_CAPS,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['<BACK>', '<CAPS>', '\'', 'Z', 'R', 'J', 'D']
        self.list_equal(neighbors, expected)

    def test_e_one(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='E',
                                                      num_moves=1,
                                                      mode=ABC_CAPS,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['D', 'F', 'L']
        self.list_equal(neighbors, expected)

    def test_p_two(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='P',
                                                      num_moves=2,
                                                      mode=ABC_CAPS,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['H', 'V', 'X', 'J', 'B', 'R']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


class ABCSpecialGraphMoveCounts(unittest.TestCase):

    def test_1_zero(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='1',
                                                      num_moves=0,
                                                      mode=ABC_SPECIAL,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['1']
        self.list_equal(neighbors, expected)

    def test_1_one(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='1',
                                                      num_moves=1,
                                                      mode=ABC_SPECIAL,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['2', '4']
        self.list_equal(neighbors, expected)

    def test_1_four(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='1',
                                                      num_moves=4,
                                                      mode=ABC_SPECIAL,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['<SPACE>', '/', '9', '@', '#']
        self.list_equal(neighbors, expected)

    def test_colon_three(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key=':',
                                                      num_moves=3,
                                                      mode=ABC_SPECIAL,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['@', '#', '-', '[']
        self.list_equal(neighbors, expected)

    def test_hash_one(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='#',
                                                      num_moves=1,
                                                      mode=ABC_SPECIAL,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = ['&', '(', '!']
        self.list_equal(neighbors, expected)

    def test_change_two(self):
        neighbors = abc_graph.get_keys_for_moves_from(start_key='<CHANGE>',
                                                      num_moves=2,
                                                      mode=ABC_SPECIAL,
                                                      use_shortcuts=True,
                                                      use_wraparound=True,
                                                      directions=Direction.ANY)
        expected = [')', '\"', '?']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


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
        expected = ['v', 'n', '^', '_', '<ABC>']
        self.list_equal(neighbors, expected)

    def test_R_four(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='R',
                                                                   num_moves=4,
                                                                   mode=APPLETV_PASSWORD_CAPS,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = ['V', 'N', '^', '_', '<ABC>']
        self.list_equal(neighbors, expected)

    def test_equals_three(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='=',
                                                                   num_moves=3,
                                                                   mode=APPLETV_PASSWORD_SPECIAL,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = [';', '?', '0', '5', '<DONE>', '<abc>']
        self.list_equal(neighbors, expected)

    def test_z_nine(self):
        neighbors = appletv_password_graph.get_keys_for_moves_from(start_key='z',
                                                                   num_moves=9,
                                                                   mode=APPLETV_PASSWORD_STANDARD,
                                                                   use_shortcuts=True,
                                                                   use_wraparound=False,
                                                                   directions=Direction.ANY)
        expected = ['q', '_', 'p']
        self.list_equal(neighbors, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(sorted(observed)), list(sorted(expected)))


if __name__ == '__main__':
    unittest.main()
