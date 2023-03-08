import os.path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import OrderedDict
from typing import List, Tuple

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.data_types import Move
from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, LABEL_SIZE, LEGEND_SIZE, TITLE_SIZE, TV_COLORS, FIGSIZE
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.file_utils import read_json, iterate_dir
from smarttvleakage.utils.constants import KeyboardType, SmartTVType, BIG_NUMBER


WIDTH = 0.25


def get_typed_length(move_seq: List[Move], tv_type: SmartTVType) -> int:
    if tv_type == SmartTVType.SAMSUNG:
        return sum((int(m.end_sound == sounds.SAMSUNG_KEY_SELECT) for m in move_seq))
    elif tv_type == SmartTVType.APPLE_TV:
        return sum((int(m.end_sound == sounds.APPLETV_KEYBOARD_SELECT) for m in move_seq))
    else:
        raise ValueError('Unknown tv type: {}'.format(tv_type))


def compare_against_optimal(move_sequences: List[List[OrderedDict]], labels: List[str], keyboard: MultiKeyboardGraph, tv_type: SmartTVType) -> Tuple[float, float, float, int]:
    assert len(move_sequences) == len(labels), 'Must provide the same number of move sequences as labels.'

    use_shortcuts_list: List[bool] = [False, True, False, True]
    use_wraparound_list: List[bool] = [False, False, True, True]

    result_correct = 0
    result_total = 0
    result_distances: List[float] = []

    for move_seq_dicts, label in zip(move_sequences, labels):
        move_seq = [Move.from_dict(m) for m in move_seq_dicts]

        if tv_type == SmartTVType.SAMSUNG:
            use_done = (move_seq[-1].end_sound == sounds.SAMSUNG_SELECT)
            has_delete = any((m.end_sound == sounds.SAMSUNG_DELETE) for m in move_seq)
            start_key = 'q'
        elif tv_type == SmartTVType.APPLE_TV:
            use_done = (move_seq[-1].end_sound == sounds.APPLETV_TOOLBAR_MOVE)
            has_delete = any((m.end_sound == sounds.APPLETV_KEYBOARD_DELETE) for m in move_seq)
            start_key = 'a'
        else:
            raise ValueError('Unknown tv type: {}'.format(tv_type))

        # Skip detected deletes and instances where the typed lengths exceed that of the target string
        if has_delete:
            continue

        typed_length = get_typed_length(move_seq, tv_type)
        if typed_length != len(label):
            continue

        optimal_move_sequences: List[List[Move]] = []

        for use_shortcuts, use_wraparound in zip(use_shortcuts_list, use_wraparound_list):
            try:
                optimal_seq = findPath(word=label,
                                       use_shortcuts=use_shortcuts,
                                       use_wraparound=use_wraparound,
                                       use_done=use_done,
                                       mistake_rate=0.0,
                                       decay_rate=1.0,
                                       max_errors=0,
                                       keyboard=keyboard,
                                       start_key=start_key)
            except AssertionError as ex:
                continue

            if len(optimal_seq) != len(move_seq):
                continue

            optimal_move_sequences.append(optimal_seq)

        # Record the results based on the best configuration at each step
        for move_idx in range(len(move_seq)):
            # Get the optimal configuration with the lowest distance
            best_dist = None
            best_idx = None

            for optimal_idx, optimal_seq in enumerate(optimal_move_sequences):
                dist = abs(optimal_seq[move_idx].num_moves - move_seq[move_idx].num_moves)
                
                if (best_dist is None) or (dist < best_dist):
                    best_idx = optimal_idx
                    best_dist = dist

            # Record the best results for each step
            if (best_idx is not None) and (best_dist is not None):
                result_distances.append(best_dist)
                result_correct += int(move_seq[move_idx].num_moves == optimal_move_sequences[best_idx][move_idx].num_moves)
                result_total += 1

    return result_correct, result_total, result_distances



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    samsung_keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)
    appletv_keyboard = MultiKeyboardGraph(KeyboardType.APPLE_TV_PASSWORD)

    samsung_correct_count = 0
    samsung_total_count = 0
    samsung_distances: List[float] = []

    appletv_correct_count = 0
    appletv_total_count = 0
    appletv_distances: List[float] = []

    for user_folder in iterate_dir(args.folder):
        # Get the move sequences on both systems
        samsung_path = os.path.join(user_folder, 'samsung_passwords.json')
        samsung_move_seq = read_json(samsung_path)['move_sequences']

        appletv_path = os.path.join(user_folder, 'appletv_passwords.json')
        appletv_move_seq = read_json(appletv_path)['move_sequences']

        # Get the labels. Users type the same passwords on both systems, but the splitting
        # may make mistakes so we get both labels independently for proper alignment.
        samsung_labels = read_json(os.path.join(user_folder, 'samsung_passwords_labels.json'))['labels']
        appletv_labels = read_json(os.path.join(user_folder, 'appletv_passwords_labels.json'))['labels']

        # Skip users where the audio splitting messes up on either version.
        if len(samsung_labels) != len(appletv_labels):
            continue

        correct, total, dist = compare_against_optimal(samsung_move_seq, samsung_labels, keyboard=samsung_keyboard, tv_type=SmartTVType.SAMSUNG)
        samsung_correct_count += correct
        samsung_total_count += total
        samsung_distances.extend(dist)

        correct, total, dist = compare_against_optimal(appletv_move_seq, appletv_labels, keyboard=appletv_keyboard, tv_type=SmartTVType.APPLE_TV)
        appletv_correct_count += correct
        appletv_total_count += total
        appletv_distances.extend(dist)

    samsung_accuracy = 100.0 * (samsung_correct_count / samsung_total_count)
    appletv_accuracy = 100.0 * (appletv_correct_count / appletv_total_count)

    samsung_avg_dist, samsung_std_dist = np.mean(samsung_distances), np.std(samsung_distances)
    appletv_avg_dist, appletv_std_dist = np.mean(appletv_distances), np.std(appletv_distances)

    with plt.style.context(PLOT_STYLE):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

        ax0.bar(-(WIDTH / 2), height=samsung_accuracy, width=WIDTH, label='Samsung', color=TV_COLORS['samsung'])
        ax0.bar((WIDTH / 2), height=appletv_accuracy, width=WIDTH, label='Apple TV', color=TV_COLORS['appletv'])

        ax0.annotate('{:.3f}%'.format(samsung_accuracy), (-(WIDTH / 2), samsung_accuracy), xytext=(-(WIDTH / 2) - 0.1, samsung_accuracy + 1), fontsize=LABEL_SIZE)
        ax0.annotate('{:.3f}%'.format(appletv_accuracy), (WIDTH / 2, appletv_accuracy), xytext=((WIDTH / 2) - 0.1, appletv_accuracy + 1), fontsize=LABEL_SIZE)

        ax0.set_xticks([0])
        ax0.set_xticklabels([''])
        ax0.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax0.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        ax0.legend(fontsize=LEGEND_SIZE)
        ax0.set_ylabel('Accuracy (%)', fontsize=AXIS_SIZE)
        ax0.set_title('Optimal Move Accuracy', fontsize=TITLE_SIZE)

        ax1.hist(x=samsung_distances, bins=10, label='Samsung', density=True, color=TV_COLORS['samsung'])
        ax1.hist(x=appletv_distances, bins=10, label='Apple TV', density=True, color=TV_COLORS['appletv'], alpha=0.7)

        ax1.text(3, 0.5, s='Samsung: ${:.3f} (\\pm {:.3f})$ \nApple TV: ${:.3f} (\\pm {:.3f})$'.format(samsung_avg_dist, samsung_std_dist, appletv_avg_dist, appletv_std_dist), fontsize=LABEL_SIZE)

        ax1.legend(fontsize=LABEL_SIZE)
        ax1.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax1.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        ax1.set_xlabel('Distance', fontsize=AXIS_SIZE)
        ax1.set_ylabel('Density', fontsize=AXIS_SIZE)
        ax1.set_title('Distance from Optimal', fontsize=TITLE_SIZE)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
