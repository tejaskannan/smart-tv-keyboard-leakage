import unittest
import os.path

from smarttvleakage.audio import make_move_extractor
from smarttvleakage.audio.move_extractor import extract_move_directions
from smarttvleakage.utils.constants import SmartTVType, Direction
from smarttvleakage.utils.file_utils import read_pickle_gz


class MoveDirectionTests(unittest.TestCase):

    def test_bed(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', 'bed.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        self.assertEqual(move_directions[0], Direction.ANY)
        self.assertEqual(move_directions[1], Direction.ANY)
        self.assertEqual(move_directions[2], Direction.ANY)

    def test_dog(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', 'dog.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        self.assertEqual(move_directions[0], Direction.ANY)
        self.assertEqual(move_directions[1], [Direction.HORIZONTAL] * 6 + [Direction.VERTICAL])
        self.assertEqual(move_directions[2], Direction.ANY)

    def test_good(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', 'good.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        self.assertEqual(move_directions[0], [Direction.HORIZONTAL] * 4 + [Direction.VERTICAL])
        self.assertEqual(move_directions[1], [Direction.HORIZONTAL] * 4 + [Direction.VERTICAL])
        self.assertEqual(move_directions[2], Direction.ANY)
        self.assertEqual(move_directions[3], [Direction.HORIZONTAL] * 6 + [Direction.VERTICAL])

    def test_bighead(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', 'bighead.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        self.assertEqual(move_directions[0], [Direction.HORIZONTAL] * 4 + [Direction.VERTICAL] * 2)

        for direction in move_directions[1:]:
            self.assertEqual(direction, Direction.ANY)

    def test_hamilton(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', 'hamilton.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        self.assertEqual(move_directions[0], [Direction.HORIZONTAL] * 5 + [Direction.VERTICAL])

        for direction in move_directions[1:]:
            self.assertEqual(direction, Direction.ANY)

    def test_heather1(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', 'heather1.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        self.assertEqual(move_directions[0], [Direction.HORIZONTAL] * 5 + [Direction.VERTICAL])
        self.assertEqual(move_directions[1], Direction.ANY)
        self.assertEqual(move_directions[2], Direction.ANY)
        self.assertEqual(move_directions[3], [Direction.HORIZONTAL] * 4 + [Direction.VERTICAL])

        for direction in move_directions[4:]:
            self.assertEqual(direction, Direction.ANY)

    def test_321654987(self):
        audio_signal = read_pickle_gz(os.path.join('sounds', 'samsung', '321654987.pkl.gz'))

        extractor = make_move_extractor(tv_type=SmartTVType.SAMSUNG)
        moves, _, _ = extractor.extract_move_sequence(audio=audio_signal)

        move_directions = list(map(extract_move_directions, moves))
        for direction in move_directions:
            self.assertEqual(direction, Direction.ANY)


if __name__ == '__main__':
    unittest.main()
