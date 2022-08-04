import unittest
from typing import List
from smarttvleakage.audio.move_extractor import Move, Sound
from smarttvleakage.keyboard_utils.word_to_move import findPath

class GraphMoveCounts(unittest.TestCase):

    def test_same_letter(self):
        path = findPath('q', 0, False, False)
        expected = [Move(num_moves=0.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def test_letters(self):
        path = findPath('hello', 0, False, False)
        expected = [Move(num_moves=6.0, end_sound=Sound.KEY_SELECT),Move(num_moves=4.0, end_sound=Sound.KEY_SELECT),Move(num_moves=7.0, end_sound=Sound.KEY_SELECT),Move(num_moves=0.0, end_sound=Sound.KEY_SELECT),Move(num_moves=1.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def test_letters_and_numbers(self):
        path = findPath('hello93', 0, False, False)
        expected = [Move(num_moves=6.0, end_sound=Sound.KEY_SELECT),Move(num_moves=4.0, end_sound=Sound.KEY_SELECT),Move(num_moves=7.0, end_sound=Sound.KEY_SELECT),Move(num_moves=0.0, end_sound=Sound.KEY_SELECT),Move(num_moves=1.0, end_sound=Sound.KEY_SELECT),Move(num_moves=1.0, end_sound=Sound.KEY_SELECT),Move(num_moves=6.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def test_switching(self):
        path = findPath('q"t]', 0, False, False)
        print(path)
        expected = [Move(num_moves=0.0, end_sound=Sound.KEY_SELECT),Move(num_moves=1.0, end_sound=Sound.SELECT),Move(num_moves=3.0, end_sound=Sound.KEY_SELECT),Move(num_moves=3.0, end_sound=Sound.SELECT),Move(num_moves=5.0, end_sound=Sound.KEY_SELECT),Move(num_moves=5.0, end_sound=Sound.SELECT),Move(num_moves=12.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def test_same_wraparound(self):
        path = findPath('qph:', 0, False, True)
        expected = [Move(num_moves=0.0, end_sound=Sound.KEY_SELECT),Move(num_moves=5.0, end_sound=Sound.KEY_SELECT),Move(num_moves=5.0, end_sound=Sound.KEY_SELECT),Move(num_moves=7.0, end_sound=Sound.SELECT),Move(num_moves=4.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def test_same_shortcut(self):
        path = findPath('nc', 0, True, False)
        expected = [Move(num_moves=7.0, end_sound=Sound.KEY_SELECT),Move(num_moves=2.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def test_same_both(self):
        path = findPath('qlnc', 0, True, True)
        expected = [Move(num_moves=0.0, end_sound=Sound.KEY_SELECT),Move(num_moves=7.0, end_sound=Sound.KEY_SELECT),Move(num_moves=4.0, end_sound=Sound.KEY_SELECT),Move(num_moves=2.0, end_sound=Sound.KEY_SELECT)]
        self.list_equal(path, expected)

    def list_equal(self, observed: List[str], expected: List[str]):
        self.assertEqual(list(observed), list(expected))


if __name__ == '__main__':
    unittest.main()
