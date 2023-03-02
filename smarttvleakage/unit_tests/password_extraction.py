import unittest
from typing import List
from smarttvleakage.utils.file_utils import read_json


FILE_FORMAT = '/local/smart-tv-user-study/{}/samsung_passwords.json'
APPLETV_FILE_FORMAT = '/local/smart-tv-user-study/{}/appletv_passwords.json'


class SubjectAPasswordTests(unittest.TestCase):

    def test_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][0]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [3, 4, 3, 3, 4, 4, 3, 4, 1, 5, 5, 6, 12]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][1]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [7, 6, 0, 4, 1, 6, 0, 0, 8]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][2]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [9, 6, 2, 5, 4, 6, 11, 3, 5, 7]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_four(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][3]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [4, 6, 4, 3, 4, 6, 7, 3, 4, 4, 5, 4, 1, 12]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_five(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][4]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [9, 8, 4, 11, 7, 17, 4, 5, 5]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_six(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][5]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [10, 8, 1, 4, 4, 4, 6, 9, 6]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_seven(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][6]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [8, 6, 5, 7, 2, 5, 6, 1, 9]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_eight(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][7]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [4, 2, 2, 2, 3, 5, 5, 7, 0, 8]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_nine(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][8]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [6, 5, 9, 4, 4, 0, 5, 10, 8, 4, 7, 2]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_ten(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_seq = observed['move_sequences'][9]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [1, 6, 2, 0, 2, 10, 7, 6, 11]
        self.assert_num_moves_within_one(expected, observed_moves)

    def assert_num_moves_within_one(self, expected: List[int], observed: List[int]):
        self.assertEqual(len(expected), len(observed))

        for expected_count, observed_count in zip(expected, observed):
            self.assertTrue(abs(expected_count - observed_count) <= 1)


class SubjectCPasswordTests(unittest.TestCase):

    def test_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][0]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [2, 1, 4, 1, 12, 3, 1, 3, 9]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][1]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [9, 6, 1, 2, 2, 5, 4, 1, 1, 5, 14, 3, 4, 8, 5, 8, 12]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_three(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][2]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [2, 2, 1, 12, 10, 0, 9, 7, 5, 3, 4, 4]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_four(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][3]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [9, 7, 5, 4, 4, 3, 2, 1, 4, 6, 2, 9]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_five(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][4]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [8, 6, 1, 8, 2, 1, 4, 6, 1, 6, 11, 12, 11]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_six(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][5]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [13, 7, 7, 8, 6, 1, 7, 8, 2]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_seven(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][6]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [3, 3, 5, 8, 3, 5, 0, 0, 8]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_eight(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][7]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [6, 3, 3, 3, 5, 9, 7, 7, 8, 5, 3, 4, 1, 10]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_nine(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][8]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [4, 8, 8, 8, 9, 2, 7, 7, 2, 8, 4, 4, 6, 5, 10]
        self.assert_num_moves_within_one(expected, observed_moves)

    def test_ten(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][9]
        observed_moves = [move['num_moves'] for move in observed_seq]

        expected = [1, 4, 2, 4, 6, 3, 1, 1, 4, 3, 2, 11]
        self.assert_num_moves_within_one(expected, observed_moves)

    def assert_num_moves_within_one(self, expected: List[int], observed: List[int]):
        self.assertEqual(len(expected), len(observed))

        for expected_count, observed_count in zip(expected, observed):
            self.assertTrue(abs(expected_count - observed_count) <= 1)


class AppleTVSubjectC(unittest.TestCase):

    def test_one(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][0]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [5, 17, 9, 4, 6, 3, 1, 3]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_two(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][1]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [15, 5, 14, 9, 9, 22, 15, 17, 9, 10, 1, 13, 5, 12, 13, 4, 3, 8]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_three(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][2]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [26, 6, 14, 6, 7, 7, 4, 4, 2]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_four(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][3]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [11, 7, 9, 6, 19, 20, 5, 17, 0, 0, 0, 0, 19, 9, 9, 7, 7, 6, 16, 25, 6, 9, 15]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_five(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][4]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [12, 39, 26, 11, 8, 3, 4, 29, 27, 17, 5, 3]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_six(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][5]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [21, 11, 12, 10, 5, 1, 9, 11]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_seven(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][6]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [3, 2, 9, 15, 23, 8, 0, 0, 2]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_eight(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][7]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [5, 18, 3, 7, 1, 4, 5, 9, 17, 17, 17, 19, 13]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_nine(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][8]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [2, 12, 16, 12, 14, 22, 22, 15, 3, 2, 3, 10]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_ten(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-c'))
        observed_seq = observed['move_sequences'][9]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [18, 18, 8, 14, 22, 19, 13, 13, 24, 1]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def assert_num_moves_within_buffer(self, expected: List[int], observed: List[int], scrolls: List[int]):
        self.assertEqual(len(expected), len(observed))

        for expected_count, observed_count, num_scrolls in zip(expected, observed, scrolls):
            tolerance = max(num_scrolls, int(observed_count >= 4))
            self.assertTrue(abs(expected_count - observed_count) <= tolerance)


class AppleTVSubjectD(unittest.TestCase):

    def test_one(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][0]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [3, 8, 7, 1, 6, 6, 1, 6, 3]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_two(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][1]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [16, 6, 20, 13, 2, 6, 23, 3]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_three(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][2]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [15, 15, 21, 7, 6, 9, 5, 5, 15, 12, 4, 3, 0]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_four(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][3]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [12, 12, 17, 4, 11, 5, 7, 1, 2]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_five(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][4]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [11, 11, 9, 16, 9, 10, 10, 18, 0, 25, 1, 16, 10]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_six(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][5]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [0, 19, 1, 1, 9, 12, 9, 1, 2]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_seven(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][6]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [20, 10, 17, 11, 5, 13, 3, 7, 4]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_eight(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][7]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [0, 0, 0, 23, 25, 1, 1, 1, 2]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_nine(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][8]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [24, 17, 17, 13, 7, 23, 5, 24]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def test_ten(self):
        observed = read_json(APPLETV_FILE_FORMAT.format('subject-d'))
        observed_seq = observed['move_sequences'][9]
        observed_moves = [move['num_moves'] for move in observed_seq]
        observed_scrolls = [move['num_moves'] for move in observed_seq]

        expected = [19, 5, 3, 1, 4, 16, 16, 12, 3, 3]
        self.assert_num_moves_within_buffer(expected, observed_moves, observed_scrolls)

    def assert_num_moves_within_buffer(self, expected: List[int], observed: List[int], scrolls: List[int]):
        self.assertEqual(len(expected), len(observed))

        for expected_count, observed_count, num_scrolls in zip(expected, observed, scrolls):
            tolerance = max(num_scrolls, int(observed_count >= 4))
            self.assertTrue(abs(expected_count - observed_count) <= tolerance)



if __name__ == '__main__':
    unittest.main()
