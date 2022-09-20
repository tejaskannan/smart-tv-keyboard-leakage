from collections import namedtuple
from typing import List, Optional, Tuple

from smarttvleakage.audio import SAMSUNG_KEY_SELECT, SAMSUNG_SELECT, SAMSUNG_DELETE, Move


CreditCardSequence = namedtuple('CreditCardSequence', ['credit_card', 'zip_code', 'expiration_month', 'expiration_year', 'security_code'])


NUM_CREDIT_CARD_FIELDS = 5
CREDIT_CARD_LENGTH = (15, 16)
ZIP_CODE_LENGTH = (5, 5)
MONTH_LENGTH = (1, 2)
YEAR_LENGTH = (2, 4)
SECURITY_CODE_LENGTH = (3, 4)


def validate_credit_card_number(credit_card_number: str) -> bool:
    """
    Validates the credit card using the Luhn algorithm.
    """
    total_sum = 0
    should_double = False

    for idx in reversed(range(len(credit_card_number))):
        char = credit_card_number[idx]
        digit = int(char)

        if should_double:
            digit *= 2

        total_sum += int(digit / 10) + int(digit % 10)
        should_double = (not should_double)

    return (total_sum % 10) == 0


def get_field_with_bounds(seq_lengths: List[int], bounds: Tuple[int, int]) -> int:
    target_idx = -1
    for idx, length in enumerate(seq_lengths):
        if (length >= bounds[0]) and (length <= bounds[1]):
            target_idx = idx
            break

    return target_idx


def extract_credit_card_sequence(move_sequence: List[Move]) -> Optional[CreditCardSequence]:
    # Split the move sequence based on selects
    split_move_sequence: List[List[Move]] = []
    split_sequence_lengths: List[int] = []

    current_sequence: List[Move] = []
    current_length = 0

    for move in move_sequence:
        current_sequence.append(move)

        if move.end_sound == SAMSUNG_SELECT:
            split_move_sequence.append(current_sequence)
            split_sequence_lengths.append(current_length)  # Don't count the final select sound

            current_sequence = []
            current_length = 0
        elif move.end_sound == SAMSUNG_DELETE:
            current_length -= 1
        else:
            current_length += 1

    print(split_sequence_lengths)

    # Get the credit card number
    credit_card_idx = get_field_with_bounds(split_sequence_lengths, bounds=CREDIT_CARD_LENGTH)
    if credit_card_idx == -1:
        return None

    # Get the next few entries, as these constitute the remaining pieces of credit card information
    credit_card_move_seq = split_move_sequence[credit_card_idx:(credit_card_idx + NUM_CREDIT_CARD_FIELDS)]
    credit_card_move_seq_lengths = split_sequence_lengths[credit_card_idx:(credit_card_idx + NUM_CREDIT_CARD_FIELDS)]

    # Match the zip code
    zip_code_idx = get_field_with_bounds(credit_card_move_seq_lengths, bounds=ZIP_CODE_LENGTH)
    if zip_code_idx == -1:
        return None

    # Match the month and year
    month_idx = get_field_with_bounds(credit_card_move_seq_lengths, bounds=MONTH_LENGTH)
    if month_idx == -1:
        return None

    year_idx = month_idx + 1
    if year_idx >= len(credit_card_move_seq_lengths):
        return None

    # Match the security code
    security_code_idx = -1
    for idx, length in enumerate(credit_card_move_seq_lengths):
        if (length >= SECURITY_CODE_LENGTH[0]) and (length <= SECURITY_CODE_LENGTH[1]) and (idx != year_idx):
            security_code_idx = idx
            break

    if security_code_idx == -1:
        return None

    # Return the split sequence
    return CreditCardSequence(credit_card=credit_card_move_seq[0],
                              zip_code=credit_card_move_seq[zip_code_idx],
                              expiration_month=credit_card_move_seq[month_idx],
                              expiration_year=credit_card_move_seq[year_idx],
                              security_code=credit_card_move_seq[security_code_idx])



if __name__ == '__main__':
    credit_card = [Move(num_moves=4, end_sound='key_select'), Move(num_moves=6, end_sound='key_select'), Move(num_moves=0, end_sound='key_select'), Move(num_moves=0, end_sound='key_select'), Move(num_moves=9, end_sound='key_select'), Move(num_moves=1, end_sound='key_select'), Move(num_moves=1, end_sound='key_select'), Move(num_moves=1, end_sound='key_select'), Move(num_moves=2, end_sound='key_select'), Move(num_moves=2, end_sound='key_select'), Move(num_moves=3, end_sound='key_select'), Move(num_moves=5, end_sound='key_select'), Move(num_moves=1, end_sound='key_select'), Move(num_moves=8, end_sound='key_select'), Move(num_moves=5, end_sound='key_select'), Move(num_moves=2, end_sound='key_select'), Move(num_moves=3, end_sound='select')]
    zip_code = [Move(num_moves=6, end_sound='key_select'), Move(num_moves=4, end_sound='key_select'), Move(num_moves=4, end_sound='key_select'), Move(num_moves=5, end_sound='key_select'), Move(num_moves=4, end_sound='key_select'), Move(num_moves=4, end_sound='key_select'), Move(num_moves=3, end_sound='select')]
    expiration_month = [Move(num_moves=1, end_sound='key_select'), Move(num_moves=9, end_sound='key_select'), Move(num_moves=3, end_sound='select')]
    expiration_year = [Move(num_moves=2, end_sound='key_select'), Move(num_moves=5, end_sound='key_select'), Move(num_moves=3, end_sound='select')]
    security_code = [Move(num_moves=7, end_sound='key_select'), Move(num_moves=6, end_sound='key_select'), Move(num_moves=8, end_sound='key_select'), Move(num_moves=8, end_sound='select')]

    move_seq = credit_card + zip_code + expiration_month + expiration_year + security_code

    credit_card_sequence = extract_credit_card_sequence(move_seq)
    print(credit_card_sequence)
