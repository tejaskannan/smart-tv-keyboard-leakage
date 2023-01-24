import numpy as np
import os.path
import sys
from argparse import ArgumentParser
from enum import Enum, auto
from typing import List, Optional, Any, Tuple, Dict

from smarttvleakage.audio import make_move_extractor, Move
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence, CreditCardSequence
from smarttvleakage.utils.file_utils import read_pickle_gz, save_json


class SequenceType(Enum):
    STANDARD = auto()
    CREDIT_CARD = auto()


def split_into_instances(move_sequence: List[Move], min_num_selections: int) -> Tuple[SequenceType, List[Any]]:
    if len(move_sequence) == 0:
        return SequenceType.STANDARD, []

    # Handle Credit Card Entries based on counting instead of timing. Note that when we detect credit cards,
    # we only extract the information relevant to credit cards even if there are other typing instance. This
    # feature is purely out of convenience during experimentation. We can use the same timing principles as below
    # to extract other fields if desired.
    credit_card_splits: Optional[List[CreditCardSequence]] = extract_credit_card_sequence(move_sequence, min_seq_length=min_num_selections)

    if credit_card_splits is not None:
        print('Found {} Credit Card Info'.format(len(credit_card_splits)))
        return SequenceType.CREDIT_CARD, credit_card_splits

    # Get the time differences between consecutive moves
    time_diffs: List[int] = []

    for idx in range(1, len(move_sequence)):
        prev_move = move_sequence[idx - 1]
        curr_move = move_sequence[idx]
        diff = curr_move.start_time - prev_move.end_time
        time_diffs.append(diff)

    # Get the cutoff time between moves
    iqr_time = np.percentile(time_diffs, 75) - np.percentile(time_diffs, 25)
    median_time = np.median(time_diffs)
    cutoff_time = median_time + 4.0 * iqr_time

    split_sequence: List[List[Move]] = []
    current_split: List[Move] = [move_sequence[0]]

    for idx in range(1, len(move_sequence)):
        time_diff = time_diffs[idx - 1]
    
        if time_diff >= cutoff_time:
            if len(current_split) >= min_num_selections:
                split_sequence.append(current_split)

            current_split = []
        
        current_split.append(move_sequence[idx])

    if (len(current_split) > 0) and (len(current_split) >= min_num_selections):
        split_sequence.append(current_split)

    return SequenceType.STANDARD, split_sequence


def serialize_splits(split_seq: List[Any], tv_type: SmartTVType, seq_type: SequenceType, output_path: str):
    dictionary_splits: List[Any] = []

    def moves_to_dict(moves: List[Move]) -> List[Dict[str, Any]]:
        return list(map(lambda m: m.to_dict(), moves))

    for split in split_seq:
        if isinstance(split, list):
            dictionary_splits.append(moves_to_dict(split))
        elif isinstance(split, CreditCardSequence):
            cc_fields = {
                'credit_card': moves_to_dict(split.credit_card),
                'zip_code': moves_to_dict(split.zip_code),
                'exp_month': moves_to_dict(split.expiration_month),
                'exp_year': moves_to_dict(split.expiration_year),
                'security_code': moves_to_dict(split.security_code)
            }
            dictionary_splits.append(cc_fields)
        else:
            raise ValueError('Split of unknown type: {}'.format(type(split)))

    result = {
        'tv_type': tv_type.name.lower(),
        'seq_type': seq_type.name.lower(),
        'move_sequences': dictionary_splits
    }
    save_json(result, output_path)


if __name__ == '__main__':
    parser = ArgumentParser('Splits the Smart TV audio file into keyboard instances and extracts the move sequence for each instance.')
    parser.add_argument('--spectrogram-path', type=str, required=True, help='The path to the spectrogram file (pkl.gz).')
    args = parser.parse_args()

    assert args.spectrogram_path.endswith('.pkl.gz'), 'Must provide a pickle file containing the spectrogram.'

    # Make the output file (implicit) and get user confirmation if it already exists.
    data_folder, file_name = os.path.split(args.spectrogram_path)
    output_path = os.path.join(data_folder, file_name.replace('.pkl.gz', '.json'))

    if os.path.exists(output_path):
        print('This operation will overwrite the file {}. Is this okay? [Y/N]'.format(output_path), end=' ')
        user_decision = input()

        if user_decision.lower() not in ('y', 'yes'):
            print('Quitting')
            sys.exit(0)

    # Get the spectrogram
    target_spectrogram = read_pickle_gz(args.spectrogram_path)

    # TODO: Infer the Smart TV Type
    tv_type = SmartTVType.SAMSUNG
    move_extractor = make_move_extractor(tv_type=tv_type)

    # Get the move sequence independent of the keyboard instances
    move_seq = move_extractor.extract_moves(target_spectrogram)

    # Split the move sequence into keyboard instances
    seq_type, split_seq = split_into_instances(move_sequence=move_seq, min_num_selections=1)

    print('Number of splits: {}'.format(len(split_seq)))

    serialize_splits(split_seq, tv_type=tv_type, seq_type=seq_type, output_path=output_path)
