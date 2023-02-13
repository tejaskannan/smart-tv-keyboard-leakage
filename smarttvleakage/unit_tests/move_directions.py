import unittest
import os.path

from smarttvleakage.audio.move_extractor import extract_move_directions
from smarttvleakage.utils.constants import Direction


class MoveDirectionTests(unittest.TestCase):

    def test_scroll0(self):
        move_times = [670, 762, 784, 805, 826, 847, 865, 893, 921]
        expected = [Direction.ANY] + [Direction.HORIZONTAL] * 8
        result = extract_move_directions(move_times)
        self.assertEquals(result, expected)

    def test_scroll1(self):
        move_times = [1072, 1160, 1184, 1205, 1226, 1247, 1274, 1337, 1387]
        expected = [Direction.ANY] + [Direction.HORIZONTAL] * 6 + [Direction.ANY] * 2
        result = extract_move_directions(move_times)
        self.assertEquals(result, expected)

    def test_scroll2(self):
        move_times = [1752, 1839, 1862, 1884, 1908, 1954, 1996]
        expected = [Direction.ANY] + [Direction.HORIZONTAL] * 4 + [Direction.ANY] * 2
        result = extract_move_directions(move_times)
        self.assertEquals(result, expected)

    def test_scroll2(self):
        move_times = [1240, 1328, 1351, 1373, 1394, 1415, 1440, 1520, 1597, 1631]
        expected = [Direction.ANY] + [Direction.HORIZONTAL] * 6 + [Direction.ANY] * 3
        result = extract_move_directions(move_times)
        self.assertEquals(result, expected)

    def test_scroll_short(self):
        move_times = [1862, 1884, 1908, 1954, 1996]
        result = extract_move_directions(move_times)
        self.assertEquals(result, Direction.ANY)

    def test_short(self):
        move_times = [1862, 1884, 1908]
        result = extract_move_directions(move_times)
        self.assertEquals(result, Direction.ANY)


if __name__ == '__main__':
    unittest.main()
