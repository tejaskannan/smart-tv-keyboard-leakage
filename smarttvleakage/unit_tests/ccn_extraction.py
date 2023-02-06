import unittest

from smarttvleakage.utils.file_utils import read_json


FILE_FORMAT = '/local/smart-tv-user-study/{}/credit_card_details.json'


class CreditCardMoveTests(unittest.TestCase):

    def test_subject_a_one(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [2, 0, 6, 6, 3, 0, 7, 6, 2, 2, 6, 3, 9, 4, 2, 4, 7]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_a_two(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 1, 8, 0, 3, 8, 4, 1, 2, 7, 3, 3, 4, 4, 2, 1, 7]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_a_three(self):
        observed = read_json(FILE_FORMAT.format('subject-a'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 4, 6, 5, 1, 9, 1, 7, 1, 7, 4, 2, 4, 3, 5, 11]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_b_one(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 2, 5, 4, 5, 2, 6, 5, 4, 0, 6, 6, 3, 3, 2, 3, 5]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_b_two(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 2, 6, 5, 2, 5, 3, 0, 3, 5, 5, 0, 3, 4, 6, 5, 5]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_b_three(self):
        observed = read_json(FILE_FORMAT.format('subject-b'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 3, 2, 6, 6, 5, 0, 6, 0, 3, 5, 0, 4, 8, 6, 0, 7]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_c_one(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 3, 4, 0, 7, 4, 4, 2, 6, 0, 1, 6, 4, 4, 0, 7, 5]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_c_two(self):
        observed = read_json(FILE_FORMAT.format('subject-c'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 3, 4, 1, 3, 7, 0, 7, 0, 8, 3, 2, 2, 7, 11]
        self.assertEqual(expected, observed_ccn_moves)

    #def test_subject_c_three(self):
    #    observed = read_json(FILE_FORMAT.format('subject-c'))
    #    observed_ccn_seq = observed['move_sequences'][2]['credit_card']
    #    observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

    #    expected = [3, 1, 2, 4, 5, 1, 5, 2, 1, 3, 7, 0, 2, 3, 4, 6]
    #    self.assertEqual(expected, observed_ccn_moves)

    def test_subject_d_one(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 6, 7, 1, 1, 0, 5, 3, 1, 6, 1, 6, 4, 4, 4]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_d_two(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 4, 1, 5, 3, 3, 1, 2, 7, 8, 6, 5, 4, 3, 1, 5]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_d_three(self):
        observed = read_json(FILE_FORMAT.format('subject-d'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 3, 6, 9, 4, 0, 4, 5, 1, 0, 2, 0, 5, 0, 8, 2, 7, 1, 12]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_f_one(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [2, 0, 6, 4, 3, 0, 6, 6, 0, 2, 7, 3, 10, 4, 2, 4, 7]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_f_two(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 1, 8, 0, 1, 5, 4, 1, 6, 5, 3, 3, 2, 4, 2, 1, 7]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_f_three(self):
        observed = read_json(FILE_FORMAT.format('subject-f'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 6, 6, 8, 1, 4, 1, 8, 1, 6, 4, 2, 0, 3, 8, 9]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_g_one(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 2, 3, 4, 5, 2, 6, 5, 4, 0, 7, 6, 3, 3, 2, 3, 4]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_g_two(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 2, 6, 5, 2, 5, 3, 0, 1, 5, 5, 0, 3, 4, 6, 5, 12]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_g_three(self):
        observed = read_json(FILE_FORMAT.format('subject-g'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 3, 2, 6, 7, 8, 0, 6, 2, 0, 4, 8, 6, 0, 6]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_h_one(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [2, 3, 4, 0, 7, 2, 4, 2, 6, 0, 1, 6, 4, 4, 0, 7, 14]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_h_two(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 5, 4, 3, 3, 7, 0, 8, 0, 5, 3, 2, 2, 7, 12]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_h_three(self):
        observed = read_json(FILE_FORMAT.format('subject-h'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 2, 4, 7, 1, 5, 2, 1, 3, 7, 0, 2, 3, 4, 8]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_i_one(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 1, 6, 7, 1, 1, 0, 5, 3, 1, 6, 1, 6, 4, 4, 14]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_i_two(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [3, 4, 1, 5, 1, 3, 1, 2, 7, 8, 6, 5, 4, 3, 1, 5]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_i_three(self):
        observed = read_json(FILE_FORMAT.format('subject-i'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 3, 6, 9, 4, 0, 4, 5, 1, 0, 2, 0, 5, 6, 7, 1, 12]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_j_one(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_ccn_seq = observed['move_sequences'][0]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [5, 4, 5, 2, 5, 6, 7, 1, 6, 2, 5, 3, 2, 2, 2, 1, 10]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_j_two(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_ccn_seq = observed['move_sequences'][1]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 6, 0, 5, 0, 0, 4, 5, 3, 4, 3, 4, 3, 4, 8, 5, 14]
        self.assertEqual(expected, observed_ccn_moves)

    def test_subject_j_three(self):
        observed = read_json(FILE_FORMAT.format('subject-j'))
        observed_ccn_seq = observed['move_sequences'][2]['credit_card']
        observed_ccn_moves = [move['num_moves'] for move in observed_ccn_seq]

        expected = [4, 1, 3, 2, 2, 2, 8, 4, 5, 2, 2, 3, 2, 7, 0, 1, 10]
        self.assertEqual(expected, observed_ccn_moves)


if __name__ == '__main__':
    unittest.main()
