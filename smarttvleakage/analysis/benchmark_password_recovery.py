import os.path
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from smarttvleakage.analysis.password_types import PasswordTypeCounts
from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, MARKER, LINE_WIDTH, TV_COLORS, TV_LABELS, FIGSIZE, BASELINE_COLOR
from smarttvleakage.analysis.utils import PHPBB_COUNT, PASSWORD_PRIORS, PASSWORD_PRIOR_LABELS, PasswordAccuracy
from smarttvleakage.analysis.utils import compute_rank, top_k_accuracy, compute_accuracy, compute_baseline_accuracy
from smarttvleakage.analysis.utils import has_special, has_number, has_uppercase, print_as_table
from smarttvleakage.utils.file_utils import iterate_dir, read_json


matplotlib.rc('pdf', fonttype=42)  # Embed fonts in pdf
plt.rcParams['pdf.fonttype'] = 42


TOP = [1, 5, 10, 50, 100, 250]
TOP_TO_ANNOTATE = (1, 10, 100, 250)
GUESSES_NAME = 'recovered_{}_passwords_{}.json'
LABELS_NAME = '{}_passwords_labels.json'


def get_accuracy_results(folder: str, tv_type: str) -> PasswordAccuracy:
    # Store the results
    correct_counts: Dict[str, List[int]] = dict()  # Prior -> [counts per cutoff]
    total_counts: Dict[str, int] = dict()  # Prior -> count

    type_counts = PasswordTypeCounts(prior_names=PASSWORD_PRIORS, top=TOP)

    for prior_name in PASSWORD_PRIORS:
        correct_counts[prior_name] = [0 for _ in range(len(TOP))]
        total_counts[prior_name] = 0

        for subject_folder in iterate_dir(folder):
            # Read the serialized password guesses
            guesses_path = os.path.join(subject_folder, GUESSES_NAME.format(tv_type, prior_name))
            if not os.path.exists(guesses_path):
                continue

            saved_password_guesses = read_json(guesses_path)
            password_guesses = [entry['guesses'] for entry in saved_password_guesses]

            # Read the labels
            labels_path = os.path.join(subject_folder, LABELS_NAME.format(tv_type))
            password_labels = read_json(labels_path)['labels']

            ranks = [compute_rank(guesses=guesses, label=password_labels[idx]) for idx, guesses in enumerate(password_guesses)]
            total_counts[prior_name] += len(password_labels)

            # Get the correct counts for each `top` entry
            for top_idx, top_count in enumerate(TOP):
                correct, _ = top_k_accuracy(ranks, top=top_count)
                correct_counts[prior_name][top_idx] += correct

            for rank, label in zip(ranks, password_labels):
                type_counts.count(rank=rank, label=label, prior_name=prior_name)

    return PasswordAccuracy(correct=correct_counts,
                            total=total_counts,
                            stratified_correct=type_counts.get_correct(),
                            stratified_total=type_counts.get_total())


def get_label_offset(tv_type: str, prior_name: str, top_count: int) -> Tuple[float, float]:
    if tv_type == 'samsung':
        if prior_name == 'phpbb':
            if top_count == 1:
                return (-5.0, -10.0)
            elif top_count == 10:
                return (-5.0, -9.0)
        elif prior_name == 'rockyou-5gram':
            if top_count == 1:
                return (4.0, -4.0)
            elif top_count == 10:
                return (5.0, -4.5)

        return (-30.0, -10.0)
    else:
        if prior_name == 'phpbb':
            if top_count == 1:
                return (4.0, -5.0)
            elif top_count == 10:
                return (-15.0, 3.0)
            elif top_count == 250:
                return (-35.0, 3.0)
        elif prior_name == 'rockyou-5gram':
            if top_count == 1:
                return (-6.0, 9.0)
            elif top_count == 10:
                return (-10.0, 4.0)

        return (-30.0, 3.0)


def main(folders: List[str], tv_types: List[str], output_file: Optional[str]):
    # Holds the accuracy for every TV type, prior, and top-K cutoff
    tv_accuracy: Dict[str, Dict[str, List[float]]] = {name: dict() for name in tv_types}

    # Compute the baseline accuracy as would be done with random guessing
    baseline_accuracy: List[float] = compute_baseline_accuracy(baseline_size=PHPBB_COUNT, top_counts=TOP)

    for folder, tv_type in zip(folders, tv_types):
        results = get_accuracy_results(folder, tv_type=tv_type)

        # Collect the Top-K accuracy for each prior
        for prior_name in PASSWORD_PRIORS:
            accuracy = [100.0 * (correct / results.total[prior_name]) for correct in results.correct[prior_name]]
            tv_accuracy[tv_type][prior_name] = accuracy

        # Print the stratified results in a table
        for prior_name in PASSWORD_PRIORS:
            if results.total[prior_name] > 0:
                print('==========')
                print('TV: {}, PRIOR: {}'.format(tv_type, prior_name))
                print_as_table(results.stratified_correct[prior_name], results.stratified_total[prior_name], top_counts=TOP)
                print('==========')

        # Compare the results to the baseline
        for prior_name in PASSWORD_PRIORS:
            prior_accuracy_list = tv_accuracy[tv_type][prior_name]

            if len(prior_accuracy_list) > 0:
                comp = min([acc / base for acc, base in zip(prior_accuracy_list, baseline_accuracy)])

        print('Comparison between {} and baseline for {}: {:.4f}x'.format(prior_name, tv_type, comp))

    # Plot the results
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for tv_type in tv_types:
            for prior_idx, prior_name in enumerate(PASSWORD_PRIORS):
                label = '{}, {}'.format(TV_LABELS[tv_type], PASSWORD_PRIOR_LABELS[prior_name])
                ax.plot(TOP, tv_accuracy[tv_type][prior_name], marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=label, color=TV_COLORS[tv_type][prior_idx])

                for top_idx, top_count in enumerate(TOP):
                    xoffset, yoffset = get_label_offset(tv_type=tv_type, prior_name=prior_name, top_count=top_count)

                    if top_count in TOP_TO_ANNOTATE:
                        accuracy = tv_accuracy[tv_type][prior_name][top_idx]

                        if (tv_type == 'appletv') and (prior_name == 'rockyou-5gram') and (top_idx == 0):
                            bbox = dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0)
                            ax.annotate('{:.2f}%'.format(accuracy), xy=(top_count, accuracy), xytext=(top_count + xoffset, accuracy + yoffset), size=AXIS_SIZE, bbox=bbox)
                        else:
                            ax.annotate('{:.2f}%'.format(accuracy), xy=(top_count, accuracy), xytext=(top_count + xoffset, accuracy + yoffset), size=AXIS_SIZE)

        # Plot the baseline and the final data label
        ax.plot(TOP, baseline_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Random Guess', color=BASELINE_COLOR)

        last_accuracy = baseline_accuracy[-1]
        last_count = TOP[-1]
        ax.annotate('{:.2f}%'.format(last_accuracy), xy=(last_count, last_accuracy), xytext=(last_count - 25.0, last_accuracy + 3.0), size=AXIS_SIZE)

        ax.set_ylim(top=110)

        legend_pos = (1.0, 0.75)
        ax.legend(fontsize=AXIS_SIZE, bbox_to_anchor=legend_pos)
        ax.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax.yaxis.set_tick_params(labelsize=TITLE_SIZE)

        ax.set_title('Password Accuracy in Emulation', size=TITLE_SIZE)
        ax.set_xlabel('Guess Cutoff', size=TITLE_SIZE)
        ax.set_ylabel('Accuracy (%)', size=TITLE_SIZE)

        plt.tight_layout()

        # Show or save the result
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser('Script to displays results for password recovery on benchmarks.')
    parser.add_argument('--benchmark-folders', type=str, required=True, help='Name of the folders containing the move sequences.', nargs='+')
    parser.add_argument('--tv-types', type=str, required=True, choices=['samsung', 'appletv'], help='The name of the TV types (aligned with the folders).', nargs='+')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    assert len(args.benchmark_folders) == len(args.tv_types), 'Must provide the same number of folders and TV types'

    main(folders=args.benchmark_folders, tv_types=args.tv_types, output_file=args.output_file)
