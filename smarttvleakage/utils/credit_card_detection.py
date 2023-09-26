from collections import namedtuple
from enum import Enum, auto
from typing import List, Optional, Tuple

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.data_types import Move


CreditCardSequence = namedtuple('CreditCardSequence', ['credit_card', 'zip_code', 'expiration_month', 'expiration_year', 'security_code'])
CreditCardOrder = namedtuple('CreditCardOrder', ['lengths', 'fields'])


class CreditCardField(Enum):
    CCN = auto()
    CVV = auto()
    ZIP = auto()
    MONTH = auto()
    YEAR = auto()


NUM_CREDIT_CARD_FIELDS = 5
CREDIT_CARD_LENGTH = (15, 16)
ZIP_CODE_LENGTH = (5, 5)
MONTH_LENGTH = (1, 2)
YEAR_LENGTH = (2, 4)
SECURITY_CODE_LENGTH = (3, 4)

CREDIT_CARD_FIELD_LENGTHS = [
    CreditCardOrder(lengths=[3, 2, 4, 5], fields=[CreditCardField.CVV, CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[4, 2, 4, 5], fields=[CreditCardField.CVV, CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[3, 2, 2, 5], fields=[CreditCardField.CVV, CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[4, 2, 2, 5], fields=[CreditCardField.CVV, CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[2, 4, 3, 5], fields=[CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.CVV, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[2, 4, 4, 5], fields=[CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.CVV, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[2, 2, 3, 5], fields=[CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.CVV, CreditCardField.ZIP]),
    CreditCardOrder(lengths=[2, 2, 4, 5], fields=[CreditCardField.MONTH, CreditCardField.YEAR, CreditCardField.CVV, CreditCardField.ZIP])
]


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


def match_field_lengths(form_lengths: List[int], target_lengths: List[int]) -> Optional[int]:
    if len(target_lengths) > len(form_lengths):
        return None
    
    target_size = len(target_lengths)

    for start_idx in range(len(form_lengths) - target_size + 1):
        form_split = form_lengths[start_idx:(start_idx + target_size)]

        does_match = True
        for form, target in zip(form_split, target_lengths):
            if form != target:
                does_match = False
                break

        if does_match:
            return start_idx

    return None


def get_credit_card_indices(move_seq_lengths: List[int]) -> List[CreditCardSequence]:
    result: List[CreditCardSequence] = []

    for start_idx in range(1, len(move_seq_lengths) - 3):
        end_idx = start_idx + 4  # The number of Credit Card Fields to validate
        move_seq_split = move_seq_lengths[start_idx:end_idx]
        assert len(move_seq_split) == 4, 'Found a split of length {}'.format(len(move_seq_split))

        for field_order in CREDIT_CARD_FIELD_LENGTHS:
            does_match = all((field_length == target_length for field_length, target_length in zip(field_order.lengths, move_seq_split)))

            if does_match:
                cc_indices = CreditCardSequence(credit_card=(start_idx - 1),
                                                zip_code=(start_idx + field_order.fields.index(CreditCardField.ZIP)),
                                                expiration_month=(start_idx + field_order.fields.index(CreditCardField.MONTH)),
                                                expiration_year=(start_idx + field_order.fields.index(CreditCardField.YEAR)),
                                                security_code=(start_idx + field_order.fields.index(CreditCardField.CVV)))
                result.append(cc_indices)
                break

    return result


def extract_credit_card_number(moves: List[Move], ccn_length: int) -> List[Move]:
    result: List[Move] = []

    ccn_tracked_length = 0
    for ccn_move in reversed(moves):
        if ccn_move.end_sound == sounds.SAMSUNG_KEY_SELECT:
            ccn_tracked_length += 1
        elif ccn_move.end_sound == sounds.SAMSUNG_DELETE:
            ccn_tracked_length -= 1
        elif ccn_move.end_sound != sounds.SAMSUNG_SELECT:
            raise ValueError('Unknown sound: {}'.format(ccn_move.end_sound))

        result.append(ccn_move)
    
        if ccn_tracked_length == ccn_length:
            break

    # Reverse the move seq, as we build it backwards
    return list(reversed(result))


def extract_credit_card_sequence(move_sequence: List[Move], min_seq_length: int) -> Optional[List[CreditCardSequence]]:
    # Split the move sequence based on selects
    split_move_sequence: List[List[Move]] = []
    split_sequence_lengths: List[int] = []

    current_sequence: List[Move] = []
    current_length = 0

    for move in move_sequence:
        current_sequence.append(move)

        if move.end_sound == sounds.SAMSUNG_SELECT:
            if current_length >= min_seq_length:
                split_move_sequence.append(current_sequence)
                split_sequence_lengths.append(current_length)  # Don't count the final select sound

            current_sequence = []
            current_length = 0
        elif move.end_sound == sounds.SAMSUNG_DELETE:
            current_length = max(current_length - 1, 0)
        else:
            current_length += 1

    # Get the indices in the split move sequence that correspond the credit card information
    credit_card_indices = get_credit_card_indices(split_sequence_lengths)

    if len(credit_card_indices) == 0:
        return None

    result: List[CreditCardSequence] = []

    for cc_record in credit_card_indices:
        # Get the credit card number length
        ccn_length = 0
        cvv_length = split_sequence_lengths[cc_record.security_code]

        if cvv_length == 3:
            ccn_length = 16
        elif cvv_length == 4:
            ccn_length = 15
        else:
            raise ValueError('Invalid CVV length {}'.format(cvv_length))

        ccn_seq: List[Move] = extract_credit_card_number(split_move_sequence[cc_record.credit_card], ccn_length=ccn_length)
        cvv_seq: List[Move] = split_move_sequence[cc_record.security_code]
        month_seq: List[Move] = split_move_sequence[cc_record.expiration_month]
        year_seq: List[Move] = split_move_sequence[cc_record.expiration_year]
        zip_seq: List[Move] = split_move_sequence[cc_record.zip_code]

        credit_card_seq = CreditCardSequence(credit_card=ccn_seq,
                                             zip_code=zip_seq,
                                             security_code=cvv_seq,
                                             expiration_month=month_seq,
                                             expiration_year=year_seq)
        result.append(credit_card_seq)

    return result

    # Get the order of the fields. Note: We only look at the fields other than the CCN because the first entry
    # may not be separated with a 'select' from any previous entries in the overall form
    #ccn_seq: List[Move] = []
    #cvv_seq: List[Move] = []
    #month_seq: List[Move] = []
    #year_seq: List[Move] = []
    #zip_seq: List[Move] = []
    #did_match = False
    #start_idx = None
    #cvv_length = -1

    #for field_order in CREDIT_CARD_FIELD_LENGTHS:
    #    start_idx = match_field_lengths(split_sequence_lengths, target_lengths=field_order.lengths)

    #    if start_idx is not None:  # This means we have a match
    #        offset_idx = start_idx
    #        cc_splits: Dict[CreditCardField, List[Move]] = dict()

    #        for field in field_order.fields:
    #            cc_splits[field] = split_move_sequence[offset_idx]

    #            if field == CreditCardField.CVV:
    #                cvv_length = split_sequence_lengths[offset_idx]

    #            offset_idx += 1

    #        cvv_seq = cc_splits[CreditCardField.CVV]
    #        month_seq = cc_splits[CreditCardField.MONTH]
    #        year_seq = cc_splits[CreditCardField.YEAR]
    #        zip_seq = cc_splits[CreditCardField.ZIP]

    #        did_match = True
    #        break

    #if not did_match:
    #    return None

    #assert start_idx is not None, 'No start index for credit card number extraction.'

    ## Get the index of the move seq which contains the credit card number
    #ccn_idx = start_idx - 1
    #assert ccn_idx >= 0, 'Negative credit card index in the move sequence.'

    ## Get the credit card number length
    #ccn_length = 0
    #if cvv_length == 3:
    #    ccn_length = 16
    #elif cvv_length == 4:
    #    ccn_length = 15
    #else:
    #    raise ValueError('Invalid CVV length {}'.format(cvv_length))

    ## Extract the credit card sequence. We have to take care and account
    ## for possible 'delete' keys, so it is not as simple as just
    ## extracting the last 15 or 16 moves
    #ccn_tracked_length = 0
    #ccn_contained_move_seq = split_move_sequence[ccn_idx]

    #for ccn_move in reversed(ccn_contained_move_seq):
    #    if ccn_move.end_sound == sounds.SAMSUNG_KEY_SELECT:
    #        ccn_tracked_length += 1
    #    elif ccn_move.end_sound == sounds.SAMSUNG_DELETE:
    #        ccn_tracked_length -= 1
    #    elif ccn_move.end_sound != sounds.SAMSUNG_SELECT:
    #        raise ValueError('Unknown sound: {}'.format(ccn_move.end_sound))

    #    ccn_seq.append(ccn_move)
    #
    #    if ccn_tracked_length == ccn_length:
    #        break

    ## Reverse the move seq, as we build it backwards
    #ccn_seq = list(reversed(ccn_seq))

    #return CreditCardSequence(credit_card=ccn_seq,
    #                          zip_code=zip_seq,
    #                          expiration_month=month_seq,
    #                          expiration_year=year_seq,
    #                          security_code=cvv_seq)

    ## Get the credit card number
    #credit_card_idx = get_field_with_bounds(split_sequence_lengths, bounds=CREDIT_CARD_LENGTH)
    #if credit_card_idx == -1:
    #    return None

    ## Get the next few entries, as these constitute the remaining pieces of credit card information
    #credit_card_move_seq = split_move_sequence[credit_card_idx:(credit_card_idx + NUM_CREDIT_CARD_FIELDS)]
    #credit_card_move_seq_lengths = split_sequence_lengths[credit_card_idx:(credit_card_idx + NUM_CREDIT_CARD_FIELDS)]

    ## Match the zip code
    #zip_code_idx = get_field_with_bounds(credit_card_move_seq_lengths, bounds=ZIP_CODE_LENGTH)
    #if zip_code_idx == -1:
    #    return None

    ## Match the month and year
    #month_idx = get_field_with_bounds(credit_card_move_seq_lengths, bounds=MONTH_LENGTH)
    #if month_idx == -1:
    #    return None

    #year_idx = month_idx + 1  # The month always comes right before the year
    #if year_idx >= len(credit_card_move_seq_lengths):
    #    return None

    ## Match the security code
    #security_code_idx = -1
    #for idx, length in enumerate(credit_card_move_seq_lengths):
    #    if (length >= SECURITY_CODE_LENGTH[0]) and (length <= SECURITY_CODE_LENGTH[1]) and (idx != year_idx):
    #        security_code_idx = idx
    #        break

    #if security_code_idx == -1:
    #    return None

    ## Return the split sequence
    #return CreditCardSequence(credit_card=credit_card_move_seq[0],
    #                          zip_code=credit_card_move_seq[zip_code_idx],
    #                          expiration_month=credit_card_move_seq[month_idx],
    #                          expiration_year=credit_card_move_seq[year_idx],
    #                          security_code=credit_card_move_seq[security_code_idx])



if __name__ == '__main__':
    credit_card = [Move(num_moves=4, end_sound='key_select', directions=[]), Move(num_moves=6, end_sound='key_select', directions=[]), Move(num_moves=0, end_sound='key_select', directions=[]), Move(num_moves=0, end_sound='key_select', directions=[]), Move(num_moves=9, end_sound='key_select', directions=[]), Move(num_moves=1, end_sound='key_select', directions=[]), Move(num_moves=1, end_sound='key_select', directions=[]), Move(num_moves=1, end_sound='key_select', directions=[]), Move(num_moves=2, end_sound='key_select', directions=[]), Move(num_moves=2, end_sound='key_select', directions=[]), Move(num_moves=3, end_sound='key_select', directions=[]), Move(num_moves=5, end_sound='key_select', directions=[]), Move(num_moves=1, end_sound='key_select', directions=[]), Move(num_moves=8, end_sound='key_select', directions=[]), Move(num_moves=5, end_sound='key_select', directions=[]), Move(num_moves=2, end_sound='key_select', directions=[]), Move(num_moves=3, end_sound='select', directions=[])]
    zip_code = [Move(num_moves=5, end_sound='key_select', directions=[]), Move(num_moves=4, end_sound='key_select', directions=[]), Move(num_moves=4, end_sound='key_select', directions=[]), Move(num_moves=5, end_sound='key_select', directions=[]), Move(num_moves=4, end_sound='key_select', directions=[]), Move(num_moves=4, end_sound='select', directions=[])]
    expiration_month = [Move(num_moves=1, end_sound='key_select', directions=[]), Move(num_moves=9, end_sound='key_select', directions=[]), Move(num_moves=3, end_sound='select', directions=[])]
    expiration_year = [Move(num_moves=2, end_sound='key_select', directions=[]), Move(num_moves=5, end_sound='key_select', directions=[]), Move(num_moves=3, end_sound='select', directions=[])]
    security_code = [Move(num_moves=7, end_sound='key_select', directions=[]), Move(num_moves=6, end_sound='key_select', directions=[]), Move(num_moves=8, end_sound='key_select', directions=[]), Move(num_moves=8, end_sound='select', directions=[])]

    move_seq = credit_card + expiration_month + expiration_year + security_code + zip_code

    credit_card_sequence = extract_credit_card_sequence(move_seq)
    print(credit_card_sequence)
