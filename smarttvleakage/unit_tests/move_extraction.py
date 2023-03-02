import unittest
import os.path
from typing import List

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio import SamsungMoveExtractor, AppleTVMoveExtractor, create_spectrogram
from smarttvleakage.audio.utils import get_directions_appletv
from smarttvleakage.utils.file_utils import read_pickle_gz
from smarttvleakage.utils.constants import Direction


class AppleTVDirections(unittest.TestCase):

    def test_move_directions(self):
        move_times = [10, 20, 20, 32, 32, 32, 32, 40, 40, 40, 56, 56, 56, 56, 56, 56]
        expected = [Direction.ANY] * 4 + [Direction.HORIZONTAL] * 3 + [Direction.ANY] * 4 + [Direction.HORIZONTAL] * 5
        observed = get_directions_appletv(move_times)
        self.assertEqual(expected, observed)


class SamsungMoveExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = SamsungMoveExtractor()

    def test_test(self):
        expected_num_moves = [4, 3, 10, 9, 2, 4, 9] 
        expected_sounds = [sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_DELETE, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_SELECT]

        self.run_test(recording_path=os.path.join('recordings', 'samsung', 'test.pkl.gz'),
                      expected_num_moves=expected_num_moves,
                      expected_sounds=expected_sounds)

    def test_hello900(self):
        expected_num_moves = [6, 4, 7, 0, 1, 1, 1, 0, 1] 
        expected_sounds = [sounds.SAMSUNG_KEY_SELECT] * len(expected_num_moves)

        self.run_test(recording_path=os.path.join('recordings', 'samsung', 'hello900.pkl.gz'),
                      expected_num_moves=expected_num_moves,
                      expected_sounds=expected_sounds)

    def test_password(self):
        expected_num_moves = [9, 9, 0, 1, 7, 3, 0, 0, 0, 12, 1, 0, 1, 7, 5, 2, 10] 
        expected_sounds = [sounds.SAMSUNG_KEY_SELECT] * 5 + [sounds.SAMSUNG_DELETE] * 4 + [sounds.SAMSUNG_KEY_SELECT] * 7 + [sounds.SAMSUNG_SELECT] 

        self.run_test(recording_path=os.path.join('recordings', 'samsung', 'password.pkl.gz'),
                      expected_num_moves=expected_num_moves,
                      expected_sounds=expected_sounds)

    def run_test(self, recording_path: str, expected_num_moves: List[int], expected_sounds: List[str]):
        # Read in the serialized audio (use only channel 0)
        audio = read_pickle_gz(recording_path)[:, 0]
        target_spectrogram = create_spectrogram(audio)

        observed_moves = self.extractor.extract_moves(target_spectrogram)
        
        self.assertEqual(expected_num_moves, list(map(lambda m: m.num_moves, observed_moves)))
        self.assertEqual(expected_sounds, list(map(lambda m: m.end_sound, observed_moves)))


class AppleTVMoveExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extractor = AppleTVMoveExtractor()

    def test_qwerty(self):
        expected_num_moves = [16, 6, 18, 13, 2, 5]
        expected_sounds = [sounds.APPLETV_KEYBOARD_SELECT] * len(expected_num_moves)
        
        self.run_test(recording_path=os.path.join('recordings', 'appletv', 'qwerty.pkl.gz'),
                      expected_num_moves=expected_num_moves,
                      expected_sounds=expected_sounds)

    def test_lakers(self):
        expected_num_moves = [11, 11, 10, 6, 13, 1]
        expected_sounds = [sounds.APPLETV_KEYBOARD_SELECT] * len(expected_num_moves)
        
        self.run_test(recording_path=os.path.join('recordings', 'appletv', 'lakers.pkl.gz'),
                      expected_num_moves=expected_num_moves,
                      expected_sounds=expected_sounds)

    def run_test(self, recording_path: str, expected_num_moves: List[int], expected_sounds: List[str]):
        # Read in the serialized audio (use only channel 0)
        audio = read_pickle_gz(recording_path)[:, 0]
        target_spectrogram = create_spectrogram(audio)

        observed_moves = self.extractor.extract_moves(target_spectrogram)

        self.assertEqual(expected_num_moves, list(map(lambda m: m.num_moves, observed_moves)))
        self.assertEqual(expected_sounds, list(map(lambda m: m.end_sound, observed_moves)))


if __name__ == '__main__':
    unittest.main()
