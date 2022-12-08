import unittest
import os.path
from typing import List

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.move_extractor import SamsungMoveExtractor, AppleTVMoveExtractor
from smarttvleakage.utils.file_utils import read_pickle_gz


class SamsungMoveExtraction(unittest.TestCase):

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

        move_extractor = SamsungMoveExtractor()
        observed_moves = move_extractor.extract_moves(audio)
        
        self.assertEqual(expected_num_moves, list(map(lambda m: m.num_moves, observed_moves)))
        self.assertEqual(expected_sounds, list(map(lambda m: m.end_sound, observed_moves)))



if __name__ == '__main__':
    unittest.main()
