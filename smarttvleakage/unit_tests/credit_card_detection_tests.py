import unittest

from smarttvleakage.audio import Move, SAMSUNG_SELECT
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence


class CreditCardDetectionTests(unittest.TestCase):

    def test_detection_16(self):
        credit_card_seq = findPath('4044123456781934', True, True, 0.0, 1.0, 0)
        zip_code_seq = findPath('60615', True, True, 0.0, 1.0, 0)
        expiration_month = findPath('10', True, True, 0.0, 1.0, 0)
        expiration_year = findPath('22', True, True, 0.0, 1.0, 0)
        security_code = findPath('814', True, True, 0.0, 1.0, 0)

        move_sequence = credit_card_seq + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)] + zip_code_seq + [Move(num_moves=9, end_sound=SAMSUNG_SELECT)] + expiration_month + [Move(num_moves=5, end_sound=SAMSUNG_SELECT)] + expiration_year + [Move(num_moves=12, end_sound=SAMSUNG_SELECT)] + security_code + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)]

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)

    def test_detection_15(self):
        credit_card_seq = findPath('370090174847295', True, True, 0.0, 1.0, 0)
        zip_code_seq = findPath('10254', True, True, 0.0, 1.0, 0)
        expiration_month = findPath('11', True, True, 0.0, 1.0, 0)
        expiration_year = findPath('25', True, True, 0.0, 1.0, 0)
        security_code = findPath('7491', True, True, 0.0, 1.0, 0)

        move_sequence = credit_card_seq + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)] + zip_code_seq + [Move(num_moves=9, end_sound=SAMSUNG_SELECT)] + expiration_month + [Move(num_moves=5, end_sound=SAMSUNG_SELECT)] + expiration_year + [Move(num_moves=12, end_sound=SAMSUNG_SELECT)] + security_code + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)]

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)

    def test_detection_15_year_4(self):
        credit_card_seq = findPath('370090174847295', True, True, 0.0, 1.0, 0)
        zip_code_seq = findPath('10254', True, True, 0.0, 1.0, 0)
        expiration_month = findPath('11', True, True, 0.0, 1.0, 0)
        expiration_year = findPath('2025', True, True, 0.0, 1.0, 0)
        security_code = findPath('7491', True, True, 0.0, 1.0, 0)

        move_sequence = credit_card_seq + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)] + zip_code_seq + [Move(num_moves=9, end_sound=SAMSUNG_SELECT)] + expiration_month + [Move(num_moves=5, end_sound=SAMSUNG_SELECT)] + expiration_year + [Move(num_moves=12, end_sound=SAMSUNG_SELECT)] + security_code + [Move(num_moves=10, end_sound=SAMSUNG_SELECT)]

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)


if __name__ == '__main__':
    unittest.main()
