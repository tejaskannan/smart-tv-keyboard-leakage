"""
This script extracts move sequences from a video file. The output is a json of move sequences in order of their occurence in the video.
These entries should be manually annotated with the actual typed string for ground-truth comparison purposes.
"""
import os
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Dict, Any

from smarttvleakage.audio import make_move_extractor, SmartTVTypeClassifier, MoveExtractor, Move
from smarttvleakage.audio import SAMSUNG_KEY_SELECT, SAMSUNG_SELECT, SAMSUNG_DELETE
from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms
from smarttvleakage.utils.constants import SmartTVType, BIG_NUMBER
from smarttvleakage.utils.file_utils import save_json, iterate_dir, read_pickle_gz


SAMSUNG_TIME_DELAY = 1500


def split_samsung(audio: np.ndarray, move_seq: List[Move], extractor: MoveExtractor) -> List[List[Move]]:
    # Get the times of key selects (which denote keyboard instances)
    key_select_times, _ = extractor.find_instances_of_sound(audio=audio, sound=SAMSUNG_KEY_SELECT)
    delete_times, _ = extractor.find_instances_of_sound(audio=audio, sound=SAMSUNG_DELETE)
    candidate_times = np.sort(np.concatenate([key_select_times, delete_times], axis=0))

    split_points: List[int] = []
    for idx in range(len(candidate_times) - 1):
        curr_time = candidate_times[idx]
        next_time = candidate_times[idx + 1]

        if abs(next_time - curr_time) >= SAMSUNG_TIME_DELAY:
            split_points.append(int((next_time + curr_time) / 2.0))

    move_seq_splits: List[List[Move]] = []
    current_split: List[Move] = []
    split_idx = 0

    for move in move_seq:
        split_point = split_points[split_idx] if split_idx < len(split_points) else BIG_NUMBER

        if move.end_time > split_point:
            move_seq_splits.append(current_split)
            current_split = []
            split_idx += 1

        current_split.append(move)

    if len(current_split) > 0:
        move_seq_splits.append(current_split)

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
        # Fetch the audio
        video_clip = mp.VideoFileClip(video_file)
        audio_signal = video_clip.audio.to_soundarray()

        # Infer the smart TV type
        tv_type = tv_type_clf.get_tv_type(audio=audio_signal)

        # Extract the move sequence (currently assumes all move sequences include the final move to `done`)
        move_extractor = make_move_extractor(tv_type=tv_type)
        move_sequence, did_use_autocomplete, keyboard_type = move_extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=True)

        # Split the move sequence into distinct keyboard instances
        if tv_type == SmartTVType.SAMSUNG:
            move_sequence_splits = split_samsung(audio=audio_signal, move_seq=move_sequence, extractor=move_extractor)
        else:
            raise ValueError('No split routine for {}'.format(tv_type.name))

        # Make the output file name
        file_name = os.path.basename(video_file).replace('.MOV', '').replace('.mp4', '')
        output_path = os.path.join(output_folder, '{}.json'.format(file_name))

        # Determine whether or not the system used suggestions based
        # on the move sequence and serialize the output
        results: List[Dict[str, Any]] = []

        for move_sequence in move_sequence_splits:
            move_counts = list(map(lambda m: m.num_moves, move_sequence))
            did_use_suggestions = (tv_type == SmartTVType.SAMSUNG) and (classify_ms(suggestions_model, move_counts)[0] == 1)

            result = {
                'move_seq': list(map(lambda m: m.to_dict(), move_sequence)),
                'smart_tv_type': tv_type.name.lower(),
                'keyboard_type': keyboard_type.name.lower(),
                'did_use_suggestions': did_use_suggestions
            }
            results.append(result)

        # Save the result
        save_json(results, output_path)
