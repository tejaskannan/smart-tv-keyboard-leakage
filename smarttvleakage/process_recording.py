"""
This script extracts move sequences from a video file. The output is a json of move sequences in order of their occurence in the video.
These entries should be manually annotated with the actual typed string for ground-truth comparison purposes.
"""
import os
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Any, Tuple

from smarttvleakage.audio import make_move_extractor, SmartTVTypeClassifier, MoveExtractor, Move
from smarttvleakage.audio import SAMSUNG_KEY_SELECT, SAMSUNG_SELECT, SAMSUNG_DELETE, APPLETV_TOOLBAR_MOVE, APPLETV_KEYBOARD_SELECT
from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms
from smarttvleakage.utils.constants import SmartTVType, BIG_NUMBER, KeyboardType
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence
from smarttvleakage.utils.file_utils import save_json_gz, iterate_dir, read_pickle_gz


SAMSUNG_TIME_DELAY = 750
APPLETV_PASSWORD_THRESHOLD = 800
APPLETV_TIME_DELAY = 2000
APPLETV_TOOLBAR_DUPLICATE_TIME = 30


def split_samsung(move_seq: List[Move]) -> List[Tuple[List[Move], KeyboardType]]:
    """
    Splits the move sequence into keyboard instances based on timing
    """
    move_seq_splits: List[List[Move]] = []
    current_split: List[Move] = []

    for move_idx in range(len(move_seq) - 1):
        curr_move = move_seq[move_idx]
        next_move = move_seq[move_idx + 1]
        current_split.append(curr_move)

        diff = abs(next_move.start_time - curr_move.end_time)

        if diff > SAMSUNG_TIME_DELAY:
            #print('Split Diff: {}'.format(diff))
            #print('Curr End Sound: {}, Next End Sound: {}'.format(curr_move.end_sound, next_move.end_sound))
            move_seq_splits.append(current_split)
            current_split: List[Move] = []

    # Add in the final move
    current_split.append(move_seq[-1])
    move_seq_splits.append(current_split)

    # Filter out all splits without a key select
    move_splits = list(filter(lambda split: any(s.end_sound == SAMSUNG_KEY_SELECT for s in split), move_seq_splits))
    return [(split, KeyboardType.SAMSUNG) for split in move_splits]


def get_appletv_keyboard_type(move_seq: List[Move], toolbar_move_times: List[int]) -> KeyboardType:
    start_time = move_seq[0].start_time
    toolbar_moves_before = list(filter(lambda t: t < start_time, toolbar_move_times))

    if (len(toolbar_moves_before) == 0) or (start_time - toolbar_moves_before[-1]) < APPLETV_PASSWORD_THRESHOLD:
        return KeyboardType.APPLE_TV_SEARCH
    else:
        return KeyboardType.APPLE_TV_PASSWORD


def split_appletv(move_seq: List[Move], toolbar_move_times: List[int]) -> List[Tuple[List[Move], KeyboardType]]:
    """
    Splits the Apple TV move sequence into distinct keyboard instances based on timing.
    """
    # Get the toolbar moves
    move_seq_splits: List[Tuple[List[Move], KeyboardType]] = []
    current_split: List[Move] = []

    for move_idx in range(len(move_seq) - 1):
        curr_move = move_seq[move_idx]
        next_move = move_seq[move_idx + 1]
        current_split.append(curr_move)

        print('Bounds: ({}, {}), Toolbar Times: {}'.format(curr_move.end_time, next_move.start_time, toolbar_move_times))

        are_toolbar_moves_between = any(map(lambda t: (t >= curr_move.end_time) and (t <= next_move.start_time), toolbar_move_times))
        time_diff = abs(next_move.end_time - curr_move.end_time)

        print('Time Diff: {}, Toolbar Moves Between: {}'.format(time_diff, are_toolbar_moves_between))

        if (time_diff >= APPLETV_TIME_DELAY) and (are_toolbar_moves_between):
            # Find the keyboard type based on the time between the toolbar move and the start of this move sequence
            keyboard_type = get_appletv_keyboard_type(move_seq=current_split, toolbar_move_times=toolbar_move_times)
            move_seq_splits.append((current_split, keyboard_type))
            current_split: List[Move] = []

    # Handle the last sequence
    current_split.append(move_seq[-1])
    keyboard_type = get_appletv_keyboard_type(move_seq=current_split, toolbar_move_times=toolbar_move_times)
    move_seq_splits.append((current_split, keyboard_type))

    return move_seq_splits


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    args = parser.parse_args()

    # Fetch the input file paths and specify the output directory
    if os.path.isdir(args.video_path):
        output_folder = args.video_path
        video_files = list(iterate_dir(args.video_path))
    else:
        output_folder = os.path.dirname(args.video_path)
        video_files = [args.video_path]

    # Load the suggestions model (at a known location in the repository)
    suggestions_model = read_pickle_gz('suggestions_model/model_sim.pkl.gz')

    # Make the TV Type classifier
    tv_type_clf = SmartTVTypeClassifier()

    # Process each video
    for video_file in video_files:
        if not (video_file.endswith('.mp4') or video_file.endswith('.MOV') or video_file.endswith('.mov')):
            continue

        # Fetch the audio
        audio = SmartTVAudio(video_file)
        audio_signal = audio.get_audio()

        # Infer the smart TV type
        tv_type = tv_type_clf.get_tv_type(audio=audio_signal)

        # Extract the move sequence (currently assumes all move sequences include the final move to `done`)
        move_extractor = make_move_extractor(tv_type=tv_type)
        move_sequence, did_use_autocomplete, _ = move_extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=True)

        # Attempt to split the sequence based on credit card detection
        credit_card_seq = extract_credit_card_sequence(move_sequence)

        if credit_card_seq is not None:
            move_sequence_splits = [move_sequence]  # We will re-extract the sequence on the other side during recovery (it is easier to keep track of the unified result this way)
        else:
            # Split the move sequence into distinct keyboard instances
            if tv_type == SmartTVType.SAMSUNG:
                move_sequence_splits = split_samsung(move_seq=move_sequence)
            elif tv_type == SmartTVType.APPLE_TV:
                toolbar_move_times, _ = move_extractor.find_instances_of_sound(audio_signal, sound=APPLETV_TOOLBAR_MOVE)
                key_select_times = [m.end_time for m in move_sequence if (m.end_sound == APPLETV_KEYBOARD_SELECT)]

                # Filter out duplicate sounds
                toolbar_move_times = [t for t in toolbar_move_times if all(abs(t - t0) > APPLETV_TOOLBAR_DUPLICATE_TIME for t0 in key_select_times)]
                move_sequence_splits = split_appletv(move_seq=move_sequence, toolbar_move_times=toolbar_move_times)
            else:
                raise ValueError('No split routine for {}'.format(tv_type.name))

        print('Found {} Splits for {}'.format(len(move_sequence_splits), video_file))

        # Make the output file name
        output_path = os.path.join(output_folder, '{}.json'.format(audio.file_name))

        # Determine whether or not the system used suggestions based
        # on the move sequence and serialize the output
        results: List[Dict[str, Any]] = []

        for (move_sequence, keyboard_type) in move_sequence_splits:
            # Determine whether to include the final move (based on whether we ended with a proper 'done' or not)
            if (tv_type == SmartTVType.SAMSUNG) and (move_sequence[-1].end_sound != SAMSUNG_SELECT) and (move_sequence[-1].num_moves == 1):
                move_sequence = move_sequence[0:-1]  # The last move was a suggested done key. This gives us no information, so we remove it.

            # Filter out any other 'select' sounds which may occur in the interim
            start_idx = 0
            while (start_idx < len(move_sequence)) and (move_sequence[start_idx].end_sound == SAMSUNG_SELECT) and (move_sequence[start_idx].num_moves != 1):
                start_idx += 1

            if start_idx >= len(move_sequence):
                continue

            move_sequence = move_sequence[start_idx:]
            move_counts = list(map(lambda m: m.num_moves, move_sequence))
            did_use_suggestions = (tv_type == SmartTVType.SAMSUNG) and (classify_ms(suggestions_model, move_counts)[0] == 1)

            print('Move Counts: {}'.format(move_counts))
            print('End Sounds: {}'.format(list(map(lambda m: m.end_sound, move_sequence))))

            result = {
                'move_seq': list(map(lambda m: m.to_dict(), move_sequence)),
                'smart_tv_type': tv_type.name.lower(),
                'keyboard_type': keyboard_type.name.lower(),
                'did_use_suggestions': did_use_suggestions
            }
            results.append(result)

        # Save the result
        save_json_gz(results, output_path)
