import unittest

from smarttvleakage.evaluate_credit_card_recovery import compute_recovery_rank
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.audio import Move, SAMSUNG_SELECT
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence, validate_credit_card_number


graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)


class CreditCardRankTests(unittest.TestCase):

    def test_top_ranked(self):
        total_rank = compute_recovery_rank(ccn_rank=1, month_rank=1, year_rank=1, cvv_rank=1, zip_rank=1)
        self.assertEqual(total_rank, 1)

    def test_ccn_4(self):
        total_rank = compute_recovery_rank(ccn_rank=4, month_rank=1, year_rank=1, cvv_rank=1, zip_rank=1)
        self.assertEqual(total_rank, 4)

    def test_cvv_2(self):
        total_rank = compute_recovery_rank(ccn_rank=1, month_rank=1, year_rank=1, cvv_rank=2, zip_rank=1)
        self.assertEqual(total_rank, 6)

    def test_ccn_3_cvv_2(self):
        total_rank = compute_recovery_rank(ccn_rank=3, month_rank=1, year_rank=1, cvv_rank=2, zip_rank=1)
        self.assertEqual(total_rank, 8)

    def test_zip_2(self):
        total_rank = compute_recovery_rank(ccn_rank=1, month_rank=1, year_rank=1, cvv_rank=1, zip_rank=2)
        self.assertEqual(total_rank, 26)

    def test_cvv_2_zip_2(self):
        total_rank = compute_recovery_rank(ccn_rank=1, month_rank=1, year_rank=1, cvv_rank=2, zip_rank=2)
        self.assertEqual(total_rank, 31)


class CreditCardDetectionTests(unittest.TestCase):

    def test_detection_16(self):
        credit_card_seq = findPath('4044123456781934', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        zip_code_seq = findPath('60615', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        expiration_month = findPath('10', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        expiration_year = findPath('22', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        security_code = findPath('814', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)

        move_sequence = credit_card_seq + zip_code_seq + expiration_month + expiration_year + security_code

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)

    def test_detection_15(self):
        credit_card_seq = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        expiration_year = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        security_code = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)

        move_sequence = credit_card_seq + zip_code_seq + expiration_month + expiration_year + security_code

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)

        self.assertEqual(credit_card_fields.credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields.zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields.expiration_month, expiration_month)
        self.assertEqual(credit_card_fields.expiration_year, expiration_year)
        self.assertEqual(credit_card_fields.security_code, security_code)

    def test_detection_15_year_4(self):
        credit_card_seq = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        expiration_year = findPath('2025', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)
        security_code = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph)

        move_sequence = credit_card_seq + zip_code_seq + expiration_month + expiration_year + security_code

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
