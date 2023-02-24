import unittest
from typing import List
from smarttvleakage.utils.file_utils import read_json


FILE_FORMAT = '/local/smart-tv-user-study/{}/credit_card_details.json'


class CreditCardMoveTests(unittest.TestCase):

    def test_subject_a_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [2, 0, 6, 6, 3, 0, 7, 6, 2, 2, 6, 3, 9, 4, 2, 4, 7]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_a_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [10, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_a_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 4, 8]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_a_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 6, 1, 11]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_a_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [1, 9, 11, 7, 5, 9]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_a_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 1, 8, 0, 3, 8, 4, 1, 2, 7, 3, 3, 4, 4, 2, 1, 7]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_a_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [5, 8, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_a_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 3, 9]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_a_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [6, 7, 6, 5]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_a_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [5, 5, 4, 4, 4, 9]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_a_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))

        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 4, 6, 5, 1, 9, 1, 7, 1, 7, 4, 2, 4, 3, 5, 11]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_a_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [6, 3, 8]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_a_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 6, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_a_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [1, 7, 2, 7, 13]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_a_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [6, 4, 2, 2, 6, 10]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_b_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 2, 5, 4, 5, 2, 6, 5, 4, 0, 6, 6, 3, 3, 2, 3, 5]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_b_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 0, 4]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_b_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_b_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 3, 1, 9]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_b_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [2, 1, 0, 3, 0, 13]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_b_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 2, 6, 5, 2, 5, 3, 0, 3, 5, 5, 0, 3, 4, 6, 5, 5]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_b_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 1, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_b_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 7, 5]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_b_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 5, 8, 5]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_b_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [3, 0, 7, 4, 1, 10]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_b_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 1, 3, 2, 6, 6, 5, 0, 6, 0, 3, 5, 0, 4, 8, 6, 0, 7]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_b_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 0, 4]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_b_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_b_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [2, 4, 4, 7, 5]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_b_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [3, 7, 4, 6, 3, 8]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_c_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 3, 4, 0, 7, 4, 4, 2, 6, 0, 1, 6, 4, 4, 0, 7, 5]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [7, 5, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_c_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 7, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_c_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [3, 0, 0, 6]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_c_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [3, 7, 4, 2, 2, 5]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_c_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 3, 4, 1, 3, 7, 0, 7, 0, 8, 3, 2, 2, 7, 11]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_c_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [5, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_c_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 4, 9]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_c_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [2, 2, 3, 0, 7]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_c_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [6, 3, 6, 7, 2, 4]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    #def test_subject_c_three(self):
    #    observed = read_json(FILE_FORMAT.format('subject-c'))
    #    observed_ccn_seq = observed['move_sequences'][2]['credit_card']
    #    observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

    #    expected = [3, 1, 2, 4, 5, 1, 5, 2, 1, 3, 7, 0, 2, 3, 4, 6]
    #    self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_d_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 1, 6, 7, 1, 1, 0, 5, 3, 1, 6, 1, 6, 4, 4, 4]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_d_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [10, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_d_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 5, 8]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_d_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 3, 0, 0, 7]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_d_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [9, 8, 4, 5, 5, 10]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_d_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 4, 1, 5, 3, 3, 1, 2, 7, 8, 6, 5, 4, 3, 1, 5]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_d_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 1, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_d_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 6, 6]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_d_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 3, 7, 6, 8]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_d_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [6, 4, 4, 2, 0, 6]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_d_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 3, 6, 9, 4, 0, 4, 5, 1, 0, 2, 0, 5, 0, 8, 2, 7, 1, 12]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_d_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [10, 6, 11]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_d_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_d_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [3, 2, 3, 5]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_d_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [8, 2, 0, 6, 2, 6]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_f_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [2, 0, 6, 4, 3, 0, 6, 6, 0, 2, 7, 3, 10, 4, 2, 4, 7]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_f_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [5, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_f_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 4, 10]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_f_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 6, 1, 6]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_f_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [1, 4, 6, 6, 8, 8]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_f_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 1, 8, 0, 1, 5, 4, 1, 6, 5, 3, 3, 2, 4, 2, 1, 7]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_f_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [5, 5, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_f_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 3, 8]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_f_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [9, 6, 6, 4]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_f_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [6, 5, 6, 5, 4, 4, 4, 8]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_f_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 6, 6, 8, 1, 4, 1, 8, 1, 6, 4, 2, 0, 3, 8, 9]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_f_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [5, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_f_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 6, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_f_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [1, 9, 2, 6, 9]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_f_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [6, 4, 2, 2, 9, 7]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_g_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 2, 3, 4, 5, 2, 6, 5, 4, 0, 7, 6, 3, 3, 2, 3, 4]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_g_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 0, 4]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_g_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_g_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [10, 3, 1, 8]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_g_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [2, 1, 0, 3, 0, 7]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_g_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 2, 6, 5, 2, 5, 3, 0, 1, 5, 5, 0, 3, 4, 6, 5, 12]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_g_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 1, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_g_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 7, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_g_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 3, 8, 4]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_g_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [3, 0, 7, 4, 1, 9]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_g_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 3, 2, 6, 7, 8, 0, 6, 2, 0, 4, 8, 6, 0, 6]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_g_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 0, 4]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_g_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_g_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [2, 4, 4, 7, 5]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_g_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [0, 12, 10, 7, 4, 6, 3, 7]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_h_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [2, 3, 4, 0, 7, 2, 4, 2, 6, 0, 1, 6, 4, 4, 0, 7, 14]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_h_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [7, 7, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_h_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 6, 5]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_h_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [3, 0, 0, 6]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_h_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [3, 6, 4, 2, 2, 5]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_h_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 1, 5, 4, 3, 3, 7, 0, 8, 0, 5, 3, 2, 2, 7, 12]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_h_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [7, 3, 18]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_h_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 4, 9]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_h_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [2, 2, 3, 0, 8]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_h_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [8, 3, 6, 7, 2, 14]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_h_three(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 2, 4, 7, 1, 5, 2, 1, 3, 7, 0, 2, 3, 4, 8]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_h_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [5, 2, 15]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_h_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 8, 5]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_h_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [1, 4, 4, 3, 7]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_h_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [7, 3, 3, 3, 1, 9]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_i_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 1, 6, 7, 1, 1, 0, 5, 3, 1, 6, 1, 6, 4, 4, 14]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_i_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [10, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_i_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 5, 7]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_i_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 3, 0, 0, 7]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_i_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [9, 8, 4, 5, 5, 10]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_i_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [3, 4, 1, 5, 1, 3, 1, 2, 7, 8, 6, 5, 4, 3, 1, 5]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_i_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [1, 1, 13]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_i_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 6, 7]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_i_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [5, 3, 7, 6, 8]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_i_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [6, 4, 9, 2, 0, 12]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_i_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 3, 6, 9, 4, 0, 4, 5, 1, 0, 2, 0, 5, 6, 7, 1, 12]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_i_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [10, 6, 11]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_i_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 14]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_i_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [3, 2, 3, 13]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_i_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [8, 2, 0, 9, 2, 12]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_j_ccn_one(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [5, 4, 5, 2, 5, 6, 7, 1, 6, 2, 5, 3, 2, 2, 2, 1, 10]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_j_month_one(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_month_seq = observed['move_sequences'][0]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [6, 1, 5]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_j_year_one(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_year_seq = observed['move_sequences'][0]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [3, 2, 4]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_j_cvv_one(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_cvv_seq = observed['move_sequences'][0]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [4, 6, 9, 4]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_j_zip_one(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_zip_seq = observed['move_sequences'][0]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [4, 2, 1, 0, 5, 5]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_j_ccn_two(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 6, 0, 5, 0, 0, 4, 5, 3, 4, 3, 4, 3, 4, 8, 5, 14]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_j_month_two(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_month_seq = observed['move_sequences'][1]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [8, 6, 12]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_j_year_two(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_year_seq = observed['move_sequences'][1]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 2, 11]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_j_cvv_two(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_cvv_seq = observed['move_sequences'][1]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [6, 2, 1, 8]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_j_zip_two(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_zip_seq = observed['move_sequences'][1]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [2, 7, 3, 4, 9, 4]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def test_subject_j_ccn_three(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]
        expected = [4, 1, 3, 2, 2, 2, 8, 4, 5, 2, 2, 3, 2, 7, 0, 1, 10]
        self.assert_num_moves_within_one(expected, observed_ccn_moves)

    def test_subject_j_month_three(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_month_seq = observed['move_sequences'][2]['exp_month']
        observed_month_moves = [move['num_moves'] for move in observed_month_seq]
        expected = [10, 3, 7]
        self.assert_num_moves_within_one(expected, observed_month_moves)

    def test_subject_j_year_three(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_year_seq = observed['move_sequences'][2]['exp_year']
        observed_year_moves = [move['num_moves'] for move in observed_year_seq]
        expected = [2, 6, 7]
        self.assert_num_moves_within_one(expected, observed_year_moves)

    def test_subject_j_cvv_three(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_cvv_seq = observed['move_sequences'][2]['security_code']
        observed_cvv_moves = [move['num_moves'] for move in observed_cvv_seq]
        expected = [4, 4, 6, 8]
        self.assert_num_moves_within_one(expected, observed_cvv_moves)

    def test_subject_j_zip_three(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_zip_seq = observed['move_sequences'][2]['zip_code']
        observed_zip_moves = [move['num_moves'] for move in observed_zip_seq]
        expected = [4, 1, 4, 2, 7, 5]
        self.assert_num_moves_within_one(expected, observed_zip_moves)

    def assert_num_moves_within_one(self, expected: List[int], observed: List[int]):
        self.assertEqual(len(expected), len(observed))

        for expected_count, observed_count in zip(expected, observed):
            self.assertTrue(abs(expected_count - observed_count) <= 1)

if __name__ == '__main__':
    unittest.main()
