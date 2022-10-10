"""
This script extracts move sequences from a video file. The output is a json of move sequences in order of their occurence in the video.
These entries should be manually annotated with the actual typed string for ground-truth comparison purposes.
"""
import os
import moviepy.editor as mp
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Any

from smarttvleakage.audio import make_move_extractor, SmartTVTypeClassifier, MoveExtractor, Move
from smarttvleakage.audio import SAMSUNG_KEY_SELECT, SAMSUNG_SELECT, SAMSUNG_DELETE
from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms
from smarttvleakage.utils.constants import SmartTVType, BIG_NUMBER
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence
from smarttvleakage.utils.file_utils import save_json, iterate_dir, read_pickle_gz


SAMSUNG_TIME_DELAY = 750


def split_samsung(move_seq: List[Move]) -> List[List[Move]]:
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
    return list(filter(lambda split: any(s.end_sound == SAMSUNG_KEY_SELECT for s in split), move_seq_splits))


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
        video_clip = mp.VideoFileClip(video_file)
        audio_signal = video_clip.audio.to_soundarray()

        # Infer the smart TV type
        tv_type = tv_type_clf.get_tv_type(audio=audio_signal)

        # Extract the move sequence (currently assumes all move sequences include the final move to `done`)
        move_extractor = make_move_extractor(tv_type=tv_type)
        move_sequence, did_use_autocomplete, keyboard_type = move_extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=True)

        # Attempt to split the sequence based on credit card detection
        credit_card_seq = extract_credit_card_sequence(move_sequence)

        if credit_card_seq is not None:
            move_sequence_splits = [move_sequence]  # We will re-extract the sequence on the other side during recovery (it is easier to keep track of the unified result this way)
        else:
            # Split the move sequence into distinct keyboard instances
            if tv_type == SmartTVType.SAMSUNG:
                move_sequence_splits = split_samsung(move_seq=move_sequence)
            else:
                raise ValueError('No split routine for {}'.format(tv_type.name))

        print('Found {} Splits for {}'.format(len(move_sequence_splits), video_file))

        # Make the output file name
        file_name = os.path.basename(video_file).replace('.MOV', '').replace('.mp4', '')
        output_path = os.path.join(output_folder, '{}.json'.format(file_name))

        # Determine whether or not the system used suggestions based
        # on the move sequence and serialize the output
        results: List[Dict[str, Any]] = []

        for move_sequence in move_sequence_splits:
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
        save_json(results, output_path)
