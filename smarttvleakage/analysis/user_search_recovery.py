import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple, Dict

from smarttvleakage.utils.file_utils import iterate_dir, read_json
from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, MARKER, LINE_WIDTH, COLORS_0, FIGSIZE
from smarttvleakage.analysis.utils import compute_rank, top_k_accuracy, compute_accuracy, compute_baseline_accuracy


TOP = [1, 5, 10, 50, 100]
GUESSES_NAME = 'recovered_web_searches.json'
LABELS_NAME = 'web_searches_labels.json'
DICTIONARY_SIZE = 95892


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    correct_counts: List[int] = [0 for _ in range(len(TOP))]
    total_counts: List[int] = [0 for _ in range(len(TOP))]

    for subject_folder in iterate_dir(args.user_folder):
        # Read the serialized password guesses
        guesses_path = os.path.join(subject_folder, GUESSES_NAME)
        if not os.path.exists(guesses_path):
            continue

        subject_name = os.path.split(subject_folder)[-1]
        saved_guesses = read_json(guesses_path)
        guesses = [entry['guesses'] for entry in saved_guesses]

        # Read the labels
        labels_path = os.path.join(subject_folder, LABELS_NAME)
        labels = read_json(labels_path)['labels']

        # Get the ranks of the results
        ranks = [compute_rank(guess, label=labels[idx]) for idx, guess in enumerate(guesses)]

        # Get the correct counts for each `top` entry
        for top_idx, top_count in enumerate(TOP):
            num_correct, total = top_k_accuracy(ranks=ranks, top=top_count)
            correct_counts[top_idx] += num_correct
            total_counts[top_idx] += max(total, len(labels))

        print('Subject {}, Correct: {}, Total: {}'.format(subject_name, correct_counts, total_counts))

    accuracy_results = compute_accuracy(num_correct=correct_counts, total_counts=total_counts)
    baseline_accuracy = compute_baseline_accuracy(baseline_size=DICTIONARY_SIZE, top_counts=TOP)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        ax.plot(TOP, accuracy_results, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Audio', color=COLORS_0[0])
        ax.plot(TOP, baseline_accuracy, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Random Guess', color=COLORS_0[-1])

        for top_count, accuracy in zip(TOP, accuracy_results):
            if top_count == 100:
                xoffset = -8.0
                yoffset = -1.5
            elif top_count == 50:
                xoffset = -3.0
                yoffset = -1.5
            else:
                xoffset = 3.5
                yoffset = -1.0

            ax.annotate('{:.2f}%'.format(accuracy), (top_count, accuracy), (top_count + xoffset, accuracy + yoffset), size=12)

        ax.annotate('{:.2f}%'.format(baseline_accuracy[-1]), (TOP[-1], baseline_accuracy[-1]), (TOP[-1] - 8.0, baseline_accuracy[-1] + 0.5), size=LABEL_SIZE)

        ax.legend(fontsize=LEGEND_SIZE)
        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        ax.set_xlabel('Guess Cutoff (K)', fontsize=AXIS_SIZE)
        ax.set_ylabel('Accuracy (%)', fontsize=AXIS_SIZE)
        ax.set_title('Web Search Top-K Recovery Accuracy on Human Users', fontsize=TITLE_SIZE)

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
