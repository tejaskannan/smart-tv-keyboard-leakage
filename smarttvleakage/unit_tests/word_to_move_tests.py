import unittest
from typing import List
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.audio.sounds import SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT, APPLETV_TOOLBAR_MOVE
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, Direction
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph

apple_pass = MultiKeyboardGraph(KeyboardType.APPLE_TV_PASSWORD)
samsung = MultiKeyboardGraph(KeyboardType.SAMSUNG)
apple_search = MultiKeyboardGraph(KeyboardType.APPLE_TV_SEARCH)
abc = MultiKeyboardGraph(KeyboardType.ABC)


class GraphMoveCounts(unittest.TestCase):

    def test_same_letter(self):
        path = findPath('q', False, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_letter_apple(self):
        path = findPath('q', False, False, False, 0.0, 1.0, 0, apple_search, 't')
        expected = [Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters(self):
        path = findPath('hello', False, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_apple(self):
        path = findPath('hello', False, False, False, 0.0, 1.0, 0, apple_search, 't')
        expected = [Move(num_moves=12, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)
    
    def test_letters_and_numbers(self):
        path = findPath('hello93', False, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=6, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_and_numbers_apple(self):
        path = findPath('hello93', False, False, False, 0.0, 1.0, 0, apple_search, 't')
        expected = [Move(num_moves=12, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=2, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=6, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_switching(self):
        path = findPath('q"t]', False, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=1, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=12, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_switching_apple(self):
        path = findPath('q"t', False, False, False, 0.0, 1.0, 0, apple_search, 't')
        expected = [Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=14, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=17, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_wraparound(self):
        path = findPath('qph:', False, True, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_shortcut(self):
        path = findPath('nc', True, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=2, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_both(self):
        path = findPath('qlnc', True, True, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY),Move(num_moves=2, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_string_with_space(self):
        path = findPath('12 45', True, True, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=5, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_string_with_space_apple(self):
        path = findPath('ab cd', True, True, False, 0.0, 1.0, 0, apple_search, 't')
        expected = [Move(num_moves=19, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=2, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_done_exists(self):
        path = findPath('q', False, False, True, 0.0, 0.1, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=13, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_done_exists_wraparound(self):
        path = findPath('q', True, True, True, 0.0, 0.1, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_done_exists_ignored(self):
        path = findPath('q', False, True, False, 0.0, 0.1, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_underscore(self):
        path = findPath('q_q', False, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=4, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_underscore_special(self):
        path = findPath('t_$a', False, False, False, 0.0, 1.0, 0, samsung, 'q')
        expected = [Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=5, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=1, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=5, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=4, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=2, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)


class ABCTests(unittest.TestCase):

    def test_one_letter(self):
        path = findPath('q', use_shortcuts=True, use_wraparound=True, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=abc, start_key='a')
        expected = [Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_letter(self):
        path = findPath('qq', use_shortcuts=True, use_wraparound=True, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=abc, start_key='a')
        expected = [Move(num_moves=4, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=0, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters(self):
        path = findPath('hello', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=abc, start_key='a')
        expected = [Move(num_moves=m, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for m in [1, 5, 1, 0, 5]] + [Move(num_moves=4, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_cases(self):
        path = findPath('heLlO', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=abc, start_key='a')
        expected = [Move(num_moves=m, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for m in [1, 5, 1, 0, 5]] + [Move(num_moves=4, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_special(self):
        path = findPath('49ers!', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=abc, start_key='a')
        expected = [Move(num_moves=8, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)] + [Move(num_moves=m, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for m in [7, 3]] + [Move(num_moves=6, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)] + [Move(num_moves=m, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for m in [4, 3, 1]] + [Move(num_moves=4, end_sound=SAMSUNG_SELECT, directions=Direction.ANY), Move(num_moves=3, end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY), Move(num_moves=7, end_sound=SAMSUNG_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)


class AppleTVPasswordTests(unittest.TestCase):

    def test_one_letter(self):
        path = findPath('q', use_shortcuts=False, use_wraparound=False, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=apple_pass, start_key='a')
        expected = [Move(num_moves=16, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_same_letter(self):
        path = findPath('qq', use_shortcuts=False, use_wraparound=False, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=apple_pass, start_key='a')
        expected = [Move(num_moves=16, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters(self):
        path = findPath('hello', use_shortcuts=False, use_wraparound=False, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=apple_pass, start_key='a')
        expected = [Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_and_numbers(self):
        path = findPath('hello93', use_shortcuts=True, use_wraparound=False, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=apple_pass, start_key='a')
        expected = [Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=0, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=3, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=5, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=6, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_switching(self):
        path = findPath('q"t', use_shortcuts=True, use_wraparound=False, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=apple_pass, start_key='a')
        expected = [Move(num_moves=16, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=11, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY),Move(num_moves=15, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY)]
        self.assertEqual(path, expected)

    def test_letters_and_numbers_done(self):
        path = findPath('h3', use_shortcuts=True, use_wraparound=False, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=apple_pass, start_key='a')
        expected = [Move(num_moves=7, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=4, end_sound=APPLETV_KEYBOARD_SELECT, directions=Direction.ANY), Move(num_moves=2, end_sound=APPLETV_TOOLBAR_MOVE, directions=Direction.ANY)]
        self.assertEqual(path, expected)


if __name__ == '__main__':
    unittest.main()
