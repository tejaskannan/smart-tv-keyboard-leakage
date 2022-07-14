import unittest
import time
from smarttvleakage.audio.move_extractor import MoveExtractor
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_pickle_gz


class AudioExtraction(unittest.TestCase):

    def test_bed(self):
        audio_signal = read_pickle_gz('sounds/bed.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [6, 4, 1])

    def test_elk(self):
        audio_signal = read_pickle_gz('sounds/elk.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [2, 7, 1])

    def test_dog(self):
        audio_signal = read_pickle_gz('sounds/dog.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [3, 7, 5])

    def test_good(self):
        audio_signal = read_pickle_gz('sounds/good.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [5, 5, 0, 7])

    def test_tree(self):
        audio_signal = read_pickle_gz('sounds/tree.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [4, 1, 1, 0])

    def test_soccer3(self):
        audio_signal = read_pickle_gz('sounds/soccer3.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [2, 8, 8, 0, 2, 1, 2])

    def test_full_interaction(self):
        audio_signal = read_pickle_gz('sounds/full-interaction.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [4, 2, 2, 4])

    def test_earth_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/earth.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [2, 4, 1, 0, 0])

    def test_add_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/add.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [1, 3, 0])

    def test_be_autocomplete(self):
        audio_signal = read_pickle_gz('sounds/be.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [6, 1])

    def test_vol_50(self):
        audio_signal = read_pickle_gz('sounds/half_volume.pkl.gz')

        extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
        move_seq, _ = extractor.extract_move_sequence(audio=audio_signal)

        self.assertEqual(move_seq, [4, 2, 2, 4])


if __name__ == '__main__':
    unittest.main()
