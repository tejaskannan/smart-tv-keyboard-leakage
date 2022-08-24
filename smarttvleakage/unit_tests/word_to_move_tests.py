import unittest
from typing import List
from smarttvleakage.audio.move_extractor import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, Direction
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph

apple_pass = MultiKeyboardGraph(KeyboardType.APPLE_TV_PASSWORD)
samsung = MultiKeyboardGraph(KeyboardType.SAMSUNG)
apple_search = MultiKeyboardGraph(KeyboardType.APPLE_TV_SEARCH)


class GraphMoveCounts(unittest.TestCase):

    def test_same_letter(self):
        path = findPath('q', False, False, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_letter_apple(self):
        path = findPath('q', False, False, False, False, 0.0, 1.0, 0, apple_search)
        expected = [Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_letter_apple_pass(self):
        path = findPath('q', False, False, False, False, 0.0, 1.0, 0, apple_pass)
        expected = [Move(num_moves=16, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters(self):
        path = findPath('hello', False, False, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_apple(self):
        path = findPath('hello', False, False, False, False, 0.0, 1.0, 0, apple_search)
        expected = [Move(num_moves=12, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)
    
    def test_letters_apple_pass(self):
        path = findPath('hello', False, False, False, False, 0.0, 1.0, 0, apple_pass)
        expected = [Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_and_numbers(self):
        path = findPath('hello93', False, False, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_and_numbers_apple(self):
        path = findPath('hello93', False, False, False, False, 0.0, 1.0, 0, apple_search)
        expected = [Move(num_moves=12, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=2, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=6, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_and_numbers_apple_pass(self):
        path = findPath('hello93', True, False, False, False, 0.0, 1.0, 0, apple_pass)
        expected = [Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=6, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_switching(self):
        path = findPath('q"t]', False, False, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=12, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_switching_apple(self):
        path = findPath('q"t', False, False, False, False, 0.0, 1.0, 0, apple_search)
        expected = [Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=14, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=17, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_switching_apple_pass(self):
        path = findPath('q"t', True, False, False, False, 0.0, 1.0, 0, apple_pass)
        expected = [Move(num_moves=16, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=11, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=15, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_wraparound(self):
        path = findPath('qph:', False, True, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_shortcut(self):
        path = findPath('nc', True, False, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=2, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_both(self):
        path = findPath('qlnc', True, True, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=2, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_string_with_space(self):
        path = findPath('12 45', True, True, False, False, 0.0, 1.0, 0, samsung)
        expected = [Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=5, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_string_with_space_apple(self):
        path = findPath('ab cd', True, True, False, False, 0.0, 1.0, 0, apple_search)
        expected = [Move(num_moves=19, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=2, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_done_exists(self):
        path = findPath('q', False, True, True, False, 0.0, 0.1, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=13, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]

    def test_done_exists_ignored(self):
        path = findPath('q', False, True, False, False, 0.0, 0.1, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]

    def test_done_doesnt_exist(self):
        path = findPath('q', False, False, True, False, 0.0, 0.1, 0, samsung)
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]

    def test_direction(self):
        path = findPath('el', False, False, False, True, 0.0, 0.1, 0, samsung)
        expected = [Move(num_moves=2, end_sound=SAMSUNG_KEY_SELECT, directions=[Direction.ANY, Direction.ANY]), Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=[Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.HORIZONTAL, Direction.VERTICAL])]

if __name__ == '__main__':
    unittest.main()
