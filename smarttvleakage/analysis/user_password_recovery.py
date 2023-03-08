import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple, Dict

from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, MARKER, LINE_WIDTH, COLORS_0
from smarttvleakage.analysis.utils import compute_rank, top_k_accuracy, compute_accuracy, compute_baseline_accuracy
from smarttvleakage.utils.file_utils import iterate_dir, read_json


TOP = [1, 5, 10, 50, 100]
GUESSES_NAME = 'recovered_{}_passwords_{}.json'
LABELS_NAME = '{}_passwords_labels.json'
PRIORS = ['phpbb', 'rockyou-5gram']
PHPBB_COUNT = 184388
PRIOR_LABELS = {
    'phpbb': 'Audio + PHPBB Prior',
    'rockyou-5gram': 'Audio + Rockyou Prior'
}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--tv-type', type=str, required=True, choices=['samsung', 'appletv'], help='The name of the TV type.')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    correct_counts: Dict[str, Dict[str, List[int]]] = dict()  # Prior -> { Subject -> [ counts per rank cutoff ] }
    total_counts: Dict[str, Dict[str, List[int]]] = dict()  # Prior -> { Subject -> [ total count per rank cutoff ] }

    for prior_name in PRIORS:
        correct_counts[prior_name] = dict()  # { Subject -> [ counts per rank cutoff ] }
        total_counts[prior_name] = dict()  # { Subject -> [ total count per rank cutoff ] }

        for subject_folder in iterate_dir(args.user_folder):
            # Read the serialized password guesses
            guesses_path = os.path.join(subject_folder, GUESSES_NAME.format(args.tv_type, prior_name))
            if not os.path.exists(guesses_path):
                continue

            subject_name = os.path.split(subject_folder)[-1]
            correct_counts[prior_name][subject_name] = [0 for _ in range(len(TOP))]
            total_counts[prior_name][subject_name] = [0 for _ in range(len(TOP))]

            saved_password_guesses = read_json(guesses_path)
            password_guesses = [entry['guesses'] for entry in saved_password_guesses]

            # Read the labels
            labels_path = os.path.join(subject_folder, LABELS_NAME.format(args.tv_type))
            password_labels = read_json(labels_path)['labels']

            ranks = [compute_rank(guesses=guesses, label=password_labels[idx]) for idx, guesses in enumerate(password_guesses)]

            # Get the correct counts for each `top` entry
            for top_idx, top_count in enumerate(TOP):
                correct, total = top_k_accuracy(ranks, top=top_count)
                correct_counts[prior_name][subject_name][top_idx] += correct
                total_counts[prior_name][subject_name][top_idx] += max(total, len(password_labels))

            print('Prior: {}, Subject {}, Correct: {}, Total: {}'.format(prior_name, subject_name, correct_counts[prior_name][subject_name], total_counts[prior_name][subject_name]))

    # Compute the accuracy across all cutoffs for each prior
    accuracy_dict: Dict[str, List[float]] = dict()

    for prior_name in PRIORS:
        merged_count_list: List[int] = [0 for _ in range(len(TOP))]
        merged_total_list: List[int] = [0 for _ in range(len(TOP))]

        for subject_name in correct_counts[prior_name].keys():
            for top_idx in range(len(TOP)):
                merged_count_list[top_idx] += correct_counts[prior_name][subject_name][top_idx]
                merged_total_list[top_idx] += total_counts[prior_name][subject_name][top_idx]

        if all((t > 0) for t in merged_total_list):
            accuracy_list: List[float] = compute_accuracy(num_correct=merged_count_list,
                                                          total_counts=merged_total_list)
            accuracy_dict[prior_name] = accuracy_list

    baseline_accuracy: List[float] = compute_baseline_accuracy(baseline_size=PHPBB_COUNT, top_counts=TOP)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))

        for prior_idx, prior_name in enumerate(PRIORS):
            if prior_name not in accuracy_dict:
                continue

            ax.plot(TOP, accuracy_dict[prior_name], marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=PRIOR_LABELS[prior_name], color=COLORS_0[prior_idx])

            # Write the data labels
            for idx, (topk, accuracy) in enumerate(zip(TOP, accuracy_dict[prior_name])):
                if prior_name == 'phpbb':
                    if args.tv_type == 'samsung':
                        xoffset = -10 if (topk > 10) else 0
                        yoffset = -4.0
                    else:
                        xoffset = -7 if (topk > 10) else 2
                        yoffset = -2.7 if (topk > 10) else -1.2

                elif prior_name == 'rockyou-5gram':
                    if args.tv_type == 'samsung':
                        xoffset = -5 if (topk <= 5) else -7
                        yoffset = 3.0
                    else:
                        if (topk == 1):
                            xoffset = -5
                        elif (topk == 5):
                            xoffset = 2
                        elif (topk == 10):
                            xoffset = 7
                        else:
                            xoffset = -7
                        
                        yoffset = 1.0
                else:
                    raise ValueError('Unknown prior name: {}'.format(prior_name))

                ax.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=LABEL_SIZE)

        ax.plot(TOP, baseline_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Random Guess', color=COLORS_0[-1])

        if args.tv_type == 'samsung':
            last_accuracy = baseline_accuracy[-1]
            last_count = TOP[-1]
            ax.annotate('{:.2f}%'.format(last_accuracy), xy=(last_count, last_accuracy), xytext=(last_count + xoffset, last_accuracy + yoffset), size=LABEL_SIZE)

        ax.legend(fontsize=LEGEND_SIZE)
        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)

        ax.set_title('{} Password Top-K Accuracy on Human Users'.format(args.tv_type.capitalize()), size=TITLE_SIZE)
        ax.set_xlabel('Guess Cutoff (K)', size=AXIS_SIZE)
        ax.set_ylabel('Accuracy (%)', size=AXIS_SIZE)

        # Show or save the result
        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
