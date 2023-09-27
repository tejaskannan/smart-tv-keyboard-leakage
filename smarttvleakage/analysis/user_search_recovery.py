import os.path
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple, Dict

from smarttvleakage.utils.file_utils import iterate_dir, read_json
from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, MARKER, LINE_WIDTH, COLORS_0, FIGSIZE
from smarttvleakage.analysis.utils import compute_rank, top_k_accuracy, compute_accuracy, compute_baseline_accuracy


matplotlib.rc('pdf', fonttype=42)  # Embed fonts in pdf
plt.rcParams['pdf.fonttype'] = 42


TOP = [1, 5, 10, 50, 100, 250]
GUESSES_NAME = 'recovered_web_searches.json'
LABELS_NAME = 'web_searches_labels.json'
DICTIONARY_SIZE = 95892


if __name__ == '__main__':
    parser = ArgumentParser('Script to display the results of the attack on web searches.')
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--use-forced', action='store_true', help='Whether to use version with forced consideration of dynamic suggestions.')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    correct_counts: List[int] = [0 for _ in range(len(TOP))]
    total_counts: List[int] = [0 for _ in range(len(TOP))]

    file_name = GUESSES_NAME if not args.use_forced else 'forced_{}'.format(GUESSES_NAME)

    for subject_folder in iterate_dir(args.user_folder):
        # Read the serialized password guesses
        guesses_path = os.path.join(subject_folder, file_name)
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

    accuracy_results = compute_accuracy(num_correct=correct_counts, total_counts=total_counts)
    baseline_accuracy = compute_baseline_accuracy(baseline_size=DICTIONARY_SIZE, top_counts=TOP)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        ax.plot(TOP, accuracy_results, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Audio', color=COLORS_0[0])
        ax.plot(TOP, baseline_accuracy, marker=MARKER, markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label='Random Guess', color=COLORS_0[-1])

        for idx, (top_count, accuracy) in enumerate(zip(TOP, accuracy_results)):
            if idx == (len(TOP) - 1):
                xoffset = -18.0
                yoffset = -2.5
            elif idx == 0:
                xoffset = 3.5
                yoffset = 0.75
            elif idx == 1:
                xoffset = 3.5
                yoffset = -2.0
            else:
                xoffset = 3.5
                yoffset = -1.5

            ax.annotate('{:.2f}%'.format(accuracy), (top_count, accuracy), (top_count + xoffset, accuracy + yoffset), size=AXIS_SIZE)

        ax.annotate('{:.2f}%'.format(baseline_accuracy[-1]), (TOP[-1], baseline_accuracy[-1]), (TOP[-1] - 18.0, baseline_accuracy[-1] + 0.5), size=AXIS_SIZE)

        ax.legend(fontsize=AXIS_SIZE)
        ax.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax.yaxis.set_tick_params(labelsize=TITLE_SIZE)

        ax.set_xlabel('Guess Cutoff', fontsize=TITLE_SIZE)
        ax.set_ylabel('Accuracy (%)', fontsize=TITLE_SIZE)
        ax.set_title('Web Search Accuracy on Human Users', fontsize=TITLE_SIZE)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
