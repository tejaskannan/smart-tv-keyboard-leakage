import unittest

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.evaluate_credit_card_recovery import compute_recovery_rank
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.audio import Move
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
        credit_card_seq = findPath('4044123456781934', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('60615', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('10', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('22', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('814', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = credit_card_seq + expiration_month + expiration_year + security_code + zip_code_seq

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_15(self):
        credit_card_seq = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = credit_card_seq + security_code + expiration_month + expiration_year + zip_code_seq

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_15_year_4(self):
        credit_card_seq = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('2025', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = credit_card_seq + expiration_month + expiration_year + security_code + zip_code_seq

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_15_extra(self):
        credit_card_seq = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        extra_start = findPath('prefix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        extra_end = findPath('suffix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = extra_start + credit_card_seq + expiration_month + expiration_year + security_code + zip_code_seq + extra_end

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_16_extra(self):
        credit_card_seq = findPath('4044123456781934', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('749', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        extra_start = findPath('prefix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        extra_end = findPath('suffix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = extra_start + credit_card_seq + expiration_month + expiration_year + security_code + zip_code_seq + extra_end

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_15_deletes(self):
        credit_card_seq = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        delete_moves = [Move(num_moves=4, end_sound=sounds.SAMSUNG_KEY_SELECT, directions=[]), Move(num_moves=6, end_sound=sounds.SAMSUNG_DELETE, directions=[])]
        credit_card_seq = credit_card_seq[0:4] + delete_moves + credit_card_seq[4:]

        extra_start = findPath('prefix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        extra_end = findPath('suffix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = extra_start + credit_card_seq + expiration_month + expiration_year + security_code + zip_code_seq + extra_end

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_16_deletes(self):
        credit_card_seq = findPath('4044123456781934', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq = findPath('12254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month = findPath('05', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year = findPath('24', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code = findPath('491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        delete_moves = [Move(num_moves=4, end_sound=sounds.SAMSUNG_KEY_SELECT, directions=[]), Move(num_moves=6, end_sound=sounds.SAMSUNG_DELETE, directions=[])]
        credit_card_seq = credit_card_seq[0:5] + delete_moves + credit_card_seq[5:]

        extra_start = findPath('prefix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        extra_end = findPath('suffix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = extra_start + credit_card_seq + expiration_month + expiration_year + security_code + zip_code_seq + extra_end

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 1)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year)
        self.assertEqual(credit_card_fields[0].security_code, security_code)

    def test_detection_multiple(self):
        credit_card_seq_0 = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq_0 = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month_0 = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year_0 = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code_0 = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        credit_card_seq_1 = findPath('4044123456781934', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq_1 = findPath('12254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month_1 = findPath('05', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year_1 = findPath('24', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code_1 = findPath('491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = credit_card_seq_0 + expiration_month_0 + expiration_year_0 + security_code_0 + zip_code_seq_0
        move_sequence += credit_card_seq_1 + expiration_month_1 + expiration_year_1 + security_code_1 + zip_code_seq_1

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 2)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq_0)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq_0)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month_0)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year_0)
        self.assertEqual(credit_card_fields[0].security_code, security_code_0)

        self.assertEqual(credit_card_fields[1].credit_card, credit_card_seq_1)
        self.assertEqual(credit_card_fields[1].zip_code, zip_code_seq_1)
        self.assertEqual(credit_card_fields[1].expiration_month, expiration_month_1)
        self.assertEqual(credit_card_fields[1].expiration_year, expiration_year_1)
        self.assertEqual(credit_card_fields[1].security_code, security_code_1)

    def test_detection_multiple_with_extra(self):
        credit_card_seq_0 = findPath('370090174847295', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq_0 = findPath('10254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month_0 = findPath('11', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year_0 = findPath('25', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code_0 = findPath('7491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        credit_card_seq_1 = findPath('4044123456781934', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        zip_code_seq_1 = findPath('12254', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_month_1 = findPath('05', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        expiration_year_1 = findPath('24', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        security_code_1 = findPath('491', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        extra_start = findPath('prefix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        extra_middle = findPath('middlesection', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')
        extra_end = findPath('suffix', use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key='q')

        move_sequence = extra_start + credit_card_seq_0 + expiration_month_0 + expiration_year_0 + security_code_0 + zip_code_seq_0
        move_sequence += extra_middle + credit_card_seq_1 + expiration_month_1 + expiration_year_1 + security_code_1 + zip_code_seq_1 + extra_end

        credit_card_fields = extract_credit_card_sequence(move_sequence)
        self.assertTrue(credit_card_fields is not None)
        self.assertEqual(len(credit_card_fields), 2)

        self.assertEqual(credit_card_fields[0].credit_card, credit_card_seq_0)
        self.assertEqual(credit_card_fields[0].zip_code, zip_code_seq_0)
        self.assertEqual(credit_card_fields[0].expiration_month, expiration_month_0)
        self.assertEqual(credit_card_fields[0].expiration_year, expiration_year_0)
        self.assertEqual(credit_card_fields[0].security_code, security_code_0)

        self.assertEqual(credit_card_fields[1].credit_card, credit_card_seq_1)
        self.assertEqual(credit_card_fields[1].zip_code, zip_code_seq_1)
        self.assertEqual(credit_card_fields[1].expiration_month, expiration_month_1)
        self.assertEqual(credit_card_fields[1].expiration_year, expiration_year_1)
        self.assertEqual(credit_card_fields[1].security_code, security_code_1)


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
