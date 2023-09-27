import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from typing import List

from smarttvleakage.analysis.utils import PLOT_STYLE, TITLE_SIZE, AXIS_SIZE, LINE_WIDTH, MARKER_SIZE, MARKER
from smarttvleakage.analysis.utils import compute_rank, TV_COLORS, CREDIT_CARD_FIGSIZE, LABEL_SIZE, LEGEND_SIZE, BASELINE_COLOR
from smarttvleakage.utils.file_utils import read_json, iterate_dir


matplotlib.rc('pdf', fonttype=42)  # Embed fonts in pdf


TOP_CCN = [1, 5, 10, 50, 100, 250]
TOP_FULL = [1, 10, 100, 1000, 2500, 5000]

STANDARD_NAME = 'recovered_credit_card_details.json'
EXHAUSTIVE_NAME = 'exhaustive.json'
LABELS_NAME = 'credit_card_details_labels.json'
MAX_RANK = 10000


if __name__ == '__main__':
    parser = ArgumentParser('Script to compare recovery rates for exhaustive and timing-based suboptimal detection.')
    parser.add_argument('--user-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    standard_ccn_found: List[int] = [0 for _ in range(len(TOP_CCN))]
    standard_full_found: List[int] = [0 for _ in range(len(TOP_FULL))]

    exhaustive_ccn_found: List[int] = [0 for _ in range(len(TOP_CCN))]
    exhaustive_full_found: List[int] = [0 for _ in range(len(TOP_FULL))]

    search_factors: List[float] = []
    rank_improvement: List[int] = []
    num_subjects = 0

    for subject_folder in iterate_dir(args.user_folder):
        standard_path = os.path.join(subject_folder, STANDARD_NAME)
        exhaustive_path = os.path.join(subject_folder, EXHAUSTIVE_NAME)
        num_subjects += 1

        if (not os.path.exists(standard_path)) or (not os.path.exists(exhaustive_path)):
            continue

        labels = read_json(os.path.join(subject_folder, LABELS_NAME))['labels']
        standard_log = read_json(standard_path)
        exhaustive_log = read_json(exhaustive_path)

        assert len(standard_log) == len(exhaustive_log), 'Found {} standard and {} exhaustive entries'.format(len(standard_log), len(exhaustive_log))

        for standard_result, exhaustive_result, label_record in zip(standard_log, exhaustive_log, labels):
            standard_rank, exhaustive_rank = standard_result['rank'], exhaustive_result['rank']

            standard_ccn_rank = compute_rank(standard_result['ccn'], label=label_record['credit_card'])
            exhaustive_ccn_rank = compute_rank(exhaustive_result['ccn'], label=label_record['credit_card'])

            if (standard_rank > 0) and (exhaustive_rank > 0):
                rank_improvement.append(exhaustive_rank - standard_rank)

            for top_idx, top_count in enumerate(TOP_CCN):
                standard_ccn_found[top_idx] += int((standard_ccn_rank <= top_count) and (standard_ccn_rank > 0))
                exhaustive_ccn_found[top_idx] += int((exhaustive_ccn_rank <= top_count) and (exhaustive_ccn_rank > 0))

            for top_idx, top_count in enumerate(TOP_FULL):
                standard_full_found[top_idx] += int((standard_rank <= top_count) and (standard_rank > 0))
                exhaustive_full_found[top_idx] += int((exhaustive_rank <= top_count) and (exhaustive_rank > 0))

    total_count = 3 * num_subjects
    standard_ccn_accuracy = [100.0 * (count / total_count) for count in standard_ccn_found]
    exhaustive_ccn_accuracy = [100.0 * (count / total_count) for count in exhaustive_ccn_found]
    standard_full_accuracy = [100.0 * (count / total_count) for count in standard_full_found]
    exhaustive_full_accuracy = [100.0 * (count / total_count) for count in exhaustive_full_found]

    print('Top 1000 Full. Std: {:.3f}%, Exhaustive: {:.3f}%'.format(standard_full_accuracy[3], exhaustive_full_accuracy[3]))

    with plt.style.context(PLOT_STYLE):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=CREDIT_CARD_FIGSIZE)

        ax0.plot(TOP_CCN, standard_ccn_accuracy, linewidth=LINE_WIDTH, marker=MARKER, markersize=MARKER_SIZE, label='With Timing', color=TV_COLORS['samsung'][0])
        top_count, accuracy = TOP_CCN[-1], standard_ccn_accuracy[-1]
        ax0.annotate('{:.2f}%'.format(accuracy), (top_count, accuracy), (top_count - 50, accuracy - 10), fontsize=AXIS_SIZE)

        ax0.plot(TOP_CCN, exhaustive_ccn_accuracy, linewidth=LINE_WIDTH, marker=MARKER, markersize=MARKER_SIZE, label='Without Timing', color=BASELINE_COLOR)
        top_count, accuracy = TOP_CCN[-1], exhaustive_ccn_accuracy[-1]
        ax0.annotate('{:.2f}%'.format(accuracy), (top_count, accuracy), (top_count - 50, accuracy - 7), fontsize=AXIS_SIZE)

        ax1.plot(TOP_FULL, standard_full_accuracy, linewidth=LINE_WIDTH, marker=MARKER, markersize=MARKER_SIZE, label='With Timing', color=TV_COLORS['samsung'][0])
        top_count, accuracy = TOP_FULL[-1], standard_full_accuracy[-1]
        ax1.annotate('{:.2f}%'.format(accuracy), (top_count, accuracy), (top_count - 1000, accuracy - 7), fontsize=AXIS_SIZE)

        ax1.plot(TOP_FULL, exhaustive_full_accuracy, linewidth=LINE_WIDTH, marker=MARKER, markersize=MARKER_SIZE, label='Without Timing', color=BASELINE_COLOR)
        top_count, accuracy = TOP_FULL[-1], exhaustive_full_accuracy[-1]
        ax1.annotate('{:.2f}%'.format(accuracy), (top_count, accuracy), (top_count - 1000, accuracy - 7), fontsize=AXIS_SIZE)

        ax0.legend(fontsize=AXIS_SIZE)
        ax1.legend(fontsize=AXIS_SIZE)

        ax0.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax0.yaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax1.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax1.yaxis.set_tick_params(labelsize=TITLE_SIZE)

        ax0.set_title('CCN Accuracy on Users', fontsize=TITLE_SIZE)
        ax0.set_xlabel('Guess Cutoff', fontsize=TITLE_SIZE)
        ax0.set_ylabel('Accuracy (%)', fontsize=TITLE_SIZE)

        ax1.set_title('Full Accuracy on Users', fontsize=TITLE_SIZE)
        ax1.set_xlabel('Guess Cutoff', fontsize=TITLE_SIZE)
        ax1.set_ylabel('Accuracy (%)', fontsize=TITLE_SIZE)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
