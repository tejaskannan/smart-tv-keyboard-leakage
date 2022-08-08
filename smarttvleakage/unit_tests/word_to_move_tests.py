import unittest
from typing import List
from smarttvleakage.audio.move_extractor import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT
from smarttvleakage.keyboard_utils.word_to_move import findPath


class GraphMoveCounts(unittest.TestCase):

    def test_same_letter(self):
        path = findPath('q', False, False, 0.0, 1.0, 0)
        expected = [Move(num_moves=0.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def test_letters(self):
        path = findPath('hello', False, False, 0.0, 1.0, 0)
        expected = [Move(num_moves=6.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=4.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=7.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=0.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=1.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def test_letters_and_numbers(self):
        path = findPath('hello93', False, False, 0.0, 1.0, 0)
        expected = [Move(num_moves=6.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=4.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=7.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=0.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=1.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=1.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=6.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def test_switching(self):
        path = findPath('q"t]', False, False, 0.0, 1.0, 0)
        expected = [Move(num_moves=0.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=1.0, end_sound=SAMSUNG_SELECT),Move(num_moves=3.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=3.0, end_sound=SAMSUNG_SELECT),Move(num_moves=5.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=5.0, end_sound=SAMSUNG_SELECT),Move(num_moves=12.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def test_same_wraparound(self):
        path = findPath('qph:', False, True, 0.0, 1.0, 0)
        expected = [Move(num_moves=0.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=5.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=5.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=7.0, end_sound=SAMSUNG_SELECT),Move(num_moves=4.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def test_same_shortcut(self):
        path = findPath('nc', True, False, 0.0, 1.0, 0)
        expected = [Move(num_moves=7.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=2.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def test_same_both(self):
        path = findPath('qlnc', True, True, 0.0, 1.0, 0)
        expected = [Move(num_moves=0.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=7.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=4.0, end_sound=SAMSUNG_KEY_SELECT),Move(num_moves=2.0, end_sound=SAMSUNG_KEY_SELECT)]
        self.list_equal(path, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(observed), list(expected))


if __name__ == '__main__':
    unittest.main()
