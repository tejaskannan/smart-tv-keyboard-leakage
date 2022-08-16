import unittest

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.audio import Move, SAMSUNG_SELECT
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence, validate_credit_card_number


graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)


class CreditCardDetectionTests(unittest.TestCase):

    def test_detection_16(self):
        credit_card_seq = findPath('4044123456781934', True, True, 0.0, 1.0, 0, graph)
        zip_code_seq = findPath('60615', True, True, 0.0, 1.0, 0, graph)
        expiration_month = findPath('10', True, True, 0.0, 1.0, 0, graph)
        expiration_year = findPath('22', True, True, 0.0, 1.0, 0, graph)
        security_code = findPath('814', True, True, 0.0, 1.0, 0, graph)

        move_sequence = credit_card_seq + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)] + zip_code_seq + [Move(num_moves=9, end_sound=SAMSUNG_SELECT)] + expiration_month + [Move(num_moves=5, end_sound=SAMSUNG_SELECT)] + expiration_year + [Move(num_moves=12, end_sound=SAMSUNG_SELECT)] + security_code + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)]

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)

    def test_detection_15(self):
        credit_card_seq = findPath('370090174847295', True, True, 0.0, 1.0, 0, graph)
        zip_code_seq = findPath('10254', True, True, 0.0, 1.0, 0, graph)
        expiration_month = findPath('11', True, True, 0.0, 1.0, 0, graph)
        expiration_year = findPath('25', True, True, 0.0, 1.0, 0, graph)
        security_code = findPath('7491', True, True, 0.0, 1.0, 0, graph)

        move_sequence = credit_card_seq + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)] + zip_code_seq + [Move(num_moves=9, end_sound=SAMSUNG_SELECT)] + expiration_month + [Move(num_moves=5, end_sound=SAMSUNG_SELECT)] + expiration_year + [Move(num_moves=12, end_sound=SAMSUNG_SELECT)] + security_code + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)]

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)

    def test_detection_15_year_4(self):
        credit_card_seq = findPath('370090174847295', True, True, 0.0, 1.0, 0, graph)
        zip_code_seq = findPath('10254', True, True, 0.0, 1.0, 0, graph)
        expiration_month = findPath('11', True, True, 0.0, 1.0, 0, graph)
        expiration_year = findPath('2025', True, True, 0.0, 1.0, 0, graph)
        security_code = findPath('7491', True, True, 0.0, 1.0, 0, graph)

        move_sequence = credit_card_seq + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)] + zip_code_seq + [Move(num_moves=9, end_sound=SAMSUNG_SELECT)] + expiration_month + [Move(num_moves=5, end_sound=SAMSUNG_SELECT)] + expiration_year + [Move(num_moves=12, end_sound=SAMSUNG_SELECT)] + security_code + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)]

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)


class CreditCardValidation(unittest.TestCase):

    def test_valid(self):
        self.assertTrue(validate_credit_card_number('79927398713'))
        self.assertTrue(validate_credit_card_number('3716820019271998'))
        self.assertTrue(validate_credit_card_number('6823119834248189'))

    def test_invalid(self):
        self.assertTrue(not validate_credit_card_number('5190990281925290'))
        self.assertTrue(not validate_credit_card_number('37168200192719989'))
        self.assertTrue(not validate_credit_card_number('8102966371298364'))

    def test_amex(self):
        self.assertTrue(validate_credit_card_number('379097448806314'))

    def test_visa(self):
        self.assertTrue(validate_credit_card_number('4716278591454565'))

    def test_discover(self):
        self.assertTrue(validate_credit_card_number('6221261579464318446'))


if __name__ == '__main__':
    unittest.main()
