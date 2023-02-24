import unittest
from typing import List
from smarttvleakage.utils.file_utils import read_json


FILE_FORMAT = '/local/smart-tv-user-study/{}/samsung_passwords.json'


class SubjectCPasswordTests(unittest.TestCase):

    def test_subject_c_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][0]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [2, 1, 4, 1, 12, 3, 1, 3, 9]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][1]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [9, 6, 1, 2, 2, 5, 4, 1, 1, 5, 14, 3, 4, 8, 5, 8, 12]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_three(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][2]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [2, 2, 1, 12, 10, 0, 9, 7, 5, 3, 4, 4]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_four(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][3]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [9, 7, 5, 4, 4, 3, 2, 1, 4, 6, 2, 9]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_five(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][4]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [8, 6, 1, 8, 2, 1, 4, 6, 1, 6, 11, 12, 11]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_six(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][5]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [13, 7, 7, 8, 6, 1, 7, 8, 2]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_seven(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][6]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 3, 5, 8, 3, 5, 0, 0, 8]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_eight(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][7]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [6, 3, 3, 3, 5, 9, 7, 7, 8, 5, 3, 4, 1, 10]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_nine(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][8]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 8, 8, 8, 9, 2, 7, 7, 2, 8, 4, 4, 6, 5, 10]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_ten(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][9]
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [1, 4, 2, 4, 6, 3, 1, 1, 4, 3, 2, 11]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def assert_num_moves_within_one(self, expected: List[int], observed: List[int]):
        self.assertEqual(len(expected), len(observed))

        for expected_count, observed_count in zip(expected, observed):
            self.assertTrue(abs(expected_count - observed_count) <= 1)



if __name__ == '__main__':
    unittest.main()
