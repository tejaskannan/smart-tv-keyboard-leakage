import numpy as np
import os.path
import sys
from argparse import ArgumentParser
from enum import Enum, auto
from typing import List, Optional, Any, Tuple, Dict

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio import make_move_extractor, Move
from smarttvleakage.audio.tv_classifier import SmartTVTypeClassifier
from smarttvleakage.utils.constants import SmartTVType, Direction, SuggestionsType, SUGGESTIONS_CUTOFF
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence, CreditCardSequence
from smarttvleakage.utils.file_utils import read_pickle_gz, save_json
from smarttvleakage.suggestions_model.determine_autocomplete import classify_moves


MIN_LENGTH = 3


class SequenceType(Enum):
    STANDARD = auto()
    CREDIT_CARD = auto()


def split_into_instances_appletv(move_sequence: List[Move], min_num_selections: int) -> Tuple[SequenceType, List[Any], List[SuggestionsType]]:
    split_sequence: List[List[Move]] = []
    current_split: List[Move] = []

    for move in move_sequence:
        if move.end_sound == sounds.APPLETV_TOOLBAR_MOVE:
            # There can only be 2 or 3 moves to the end (including the toolbar sound which makes the final move).
            # Otherwise, the user made a mistake and there is no information gained by the ending move so we just clip it off
            if move.num_moves in (1, 2):
                toolbar_move = Move(num_moves=move.num_moves + 1,  # Account for the move to the done key (which makes the toolbar sound)
                                    end_sound=move.end_sound,
                                    directions=move.directions + [Direction.ANY],
                                    start_time=move.start_time,
                                    end_time=move.end_time,
                                    move_times=move.move_times + [move.move_times[-1]],
                                    num_scrolls=move.num_scrolls)

                # There can only be 2 or 3 moves to the end. Otherwise, the user made a mistake
                # and there is no information gained by the ending move so we just clip it off
                current_split.append(toolbar_move)

            if len(current_split) >= min_num_selections:
                split_sequence.append(current_split)

            current_split = []
        else:
            current_split.append(move)

    if len(current_split) >= min_num_selections:
        split_sequence.append(current_split)

    suggestions_types = [SuggestionsType.STANDARD for _ in split_sequence]
    return SequenceType.STANDARD, split_sequence, suggestions_types


def split_into_instances_samsung(move_sequence: List[Move], min_num_selections: int, keyboard_model: Any) -> Tuple[SequenceType, List[Any], List[SuggestionsType]]:
    if len(move_sequence) == 0:
        return SequenceType.STANDARD, [], []

    # Handle Credit Card Entries based on counting instead of timing. Note that when we detect credit cards,
    # we only extract the information relevant to credit cards even if there are other typing instance. This
    # feature is purely out of convenience during experimentation. We can use the same timing principles as below
    # to extract other fields if desired.
    credit_card_splits: Optional[List[CreditCardSequence]] = extract_credit_card_sequence(move_sequence, min_seq_length=min_num_selections)

    if credit_card_splits is not None:
        suggestions_types = [SuggestionsType.STANDARD for _ in credit_card_splits]
        return SequenceType.CREDIT_CARD, credit_card_splits, suggestions_types

    # Get the time differences between consecutive moves
    time_diffs: List[int] = []

    for idx in range(1, len(move_sequence)):
        prev_move = move_sequence[idx - 1]
        curr_move = move_sequence[idx]
        diff = curr_move.start_time - prev_move.end_time
        time_diffs.append(diff)

    # Get the cutoff time between moves
    cutoff_time = np.average(time_diffs) + 1.5 * np.std(time_diffs)

    split_sequence: List[List[Move]] = []
    current_split: List[Move] = [move_sequence[0]]
    suggestions_types: List[SuggestionsType] = []

    for idx in range(1, len(move_sequence)):
        time_diff = time_diffs[idx - 1]

        if time_diff >= cutoff_time:
            if len(current_split) >= MIN_LENGTH:
                processed, suggestions_type = process_split(current_split)

                if len(processed) >= MIN_LENGTH:
                    split_sequence.append(processed)
                    suggestions_types.append(suggestions_type)

            current_split = []

        current_split.append(move_sequence[idx])

    if (len(current_split) >= MIN_LENGTH):
        processed, suggestions_type = process_split(current_split)

        if len(processed) >= MIN_LENGTH:
            split_sequence.append(processed)
            suggestions_types.append(suggestions_type)

    for move_seq, suggestions_type in zip(split_sequence, suggestions_types):
        print('==========')
        print(suggestions_type)

        for move in move_seq:
            print(move)

    return SequenceType.STANDARD, split_sequence, suggestions_types


def process_split(move_seq: List[Move]) -> Tuple[List[Move], SuggestionsType]:
    # Clip of leading 'select' sounds. This is a heuristic, as technically one can start on the keyboard
    # with a selection. But more often than not, the first 'select' is the action that actually opens the keyboard up

    select_idx = 0
    while (select_idx < len(move_seq)) and (move_seq[select_idx].end_sound != sounds.SAMSUNG_SELECT):
        select_idx += 1

    start_idx = 0
    if select_idx < len(move_seq) and ((select_idx == 0) or all((move_seq[idx].num_moves == 0) for idx in range(0, select_idx))):
        start_idx = select_idx + 1

    move_seq = move_seq[start_idx:]
    suggestions_type = SuggestionsType.SUGGESTIONS if classify_moves(keyboard_model, move_seq, cutoff=SUGGESTIONS_CUTOFF) else SuggestionsType.STANDARD

    # Remove the last movement if it doesn't end in a 'select' and the recording shows only 1 move.
    # In this case, the 'done' key was suggested, so the movement tells us nothing about the position of the prior key
    # TODO: Amend this rule specifically for `suggestions` types
    if (move_seq[-1].end_sound != sounds.SAMSUNG_SELECT) and ((move_seq[-1].num_moves <= 1) or (suggestions_type == SuggestionsType.SUGGESTIONS)):
        move_seq = move_seq[:-1]

    return move_seq, suggestions_type


def serialize_splits(split_seq: List[Any], suggestions_types: List[SuggestionsType], tv_type: SmartTVType, seq_type: SequenceType, output_path: str):
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
        'move_sequences': dictionary_splits,
        'suggestions_types': [suggestions_type.name.lower() for suggestions_type in suggestions_types]
    }
    save_json(result, output_path)


if __name__ == '__main__':
    parser = ArgumentParser('Splits the Smart TV audio file into keyboard instances and extracts the move sequence for each instance.')
    parser.add_argument('--spectrogram-path', type=str, required=True, help='The path to the spectrogram file (pkl.gz).')
    parser.add_argument('--keyboard-model-path', type=str, required=True, help='The path to the samsung suggestions model. Only needed for Samsung instances.')
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

    # Infer the Smart TV Type
    tv_clf = SmartTVTypeClassifier()
    tv_type = tv_clf.get_tv_type(target_spectrogram)

    # Extract the move sequence
    move_extractor = make_move_extractor(tv_type=tv_type)

    # Get the move sequence independent of the keyboard instances
    move_seq = move_extractor.extract_moves(target_spectrogram)

    # Split the move sequence into keyboard instances
    if tv_type == SmartTVType.SAMSUNG:
        keyboard_model = read_pickle_gz(args.keyboard_model_path)
        seq_type, split_seq, suggestions_types = split_into_instances_samsung(move_sequence=move_seq, min_num_selections=1, keyboard_model=keyboard_model)
    elif tv_type == SmartTVType.APPLE_TV:
        seq_type, split_seq, suggestions_types = split_into_instances_appletv(move_sequence=move_seq, min_num_selections=3)
    else:
        raise ValueError('Unknown TV Type: {}'.format(tv_type))

    print('Number of splits: {}'.format(len(split_seq)))

    serialize_splits(split_seq, suggestions_types=suggestions_types, tv_type=tv_type, seq_type=seq_type, output_path=output_path)
