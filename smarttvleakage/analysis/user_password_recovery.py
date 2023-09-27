import numpy as np
import os.path
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional, Any

from smarttvleakage.analysis.password_types import PasswordTypeCounts
from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, MARKER, LINE_WIDTH, TV_COLORS, BASELINE_COLOR, TV_LABELS, FIGSIZE
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


def get_mean_std(correct_counts: Dict[str, Dict[str, List[int]]],
                 total_counts: Dict[str, Dict[str, List[int]]],
                 prior_name: str,
                 top_idx: int) -> Tuple[float, float]:
    """
    Computes the mean and standard deviation Top-K accuracy for the given prior
    across all users.
    """
    user_accuracy: List[float] = []

    for subject, subject_results in correct_counts[prior_name].items():
        correct_count = correct_counts[prior_name][subject][top_idx]
        total_count = total_counts[prior_name][subject][top_idx]
        user_accuracy.append(correct_count / total_count)

    mean = np.mean(user_accuracy)
    std = np.std(user_accuracy)

    return float(mean), float(std)


def get_standard_error(stddev: float, num_users: int) -> float:
    """
    Computes the standard error under a 95% confidence interval
    """
    return 1.97 * stddev / np.sqrt(float(num_users))


def get_accuracy_results(folder: str, tv_type: str) -> PasswordAccuracy:
    # Store the results
    correct_counts: Dict[str, Dict[str, List[int]]] = dict()  # Prior -> {subject -> [counts per cutoff]}
    total_counts: Dict[str, Dict[str, int]] = dict()  # Prior -> {subject -> count}

    type_counts = PasswordTypeCounts(prior_names=PASSWORD_PRIORS, top=TOP)

    for prior_name in PASSWORD_PRIORS:
        correct_counts[prior_name] = dict()
        total_counts[prior_name] = dict()

        for subject_folder in iterate_dir(folder):
            # Read the serialized password guesses
            guesses_path = os.path.join(subject_folder, GUESSES_NAME.format(tv_type, prior_name))
            if not os.path.exists(guesses_path):
                continue

            # Initialize the tracking information for this subject
            subject_name = os.path.split(subject_folder)[-1]
            correct_counts[prior_name][subject_name] = [0 for _ in TOP]
            total_counts[prior_name][subject_name] = 0

            # Read the labels
            saved_password_guesses = read_json(guesses_path)
            password_guesses = [entry['guesses'] for entry in saved_password_guesses]

            # Read the labels
            labels_path = os.path.join(subject_folder, LABELS_NAME.format(tv_type))
            password_labels = read_json(labels_path)['labels']

            ranks = [compute_rank(guesses=guesses, label=password_labels[idx]) for idx, guesses in enumerate(password_guesses)]
            total_counts[prior_name][subject_name] += len(password_labels)

            # Get the correct counts for each `top` entry
            for top_idx, top_count in enumerate(TOP):
                correct, _ = top_k_accuracy(ranks, top=top_count)
                correct_counts[prior_name][subject_name][top_idx] += correct

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
                return (5.0, 0.0)
            elif top_count == 250:
                return (-30.0, 3.0)
        elif prior_name == 'rockyou-5gram':
            if top_count == 1:
                return (-35.0, 3.5)
            elif top_count == 10:
                return (5.0, -2.0)
            elif top_count == 250:
                return (-30.0, 3.0)

        return (-20.0, 3.0)
    else:
        if prior_name == 'phpbb':
            if top_count == 1:
                return (-40.0, 2.5)
            elif top_count == 10:
                return (-20.0, 4.0)
            elif top_count == 250:
                return (-30.0, 3.0)
        elif prior_name == 'rockyou-5gram':
            if top_count == 1:
                return (10.0, 2.5)

    return (-20.0, 3.0)


def get_bbox(tv_type: str, prior_name: str, top_count: int) -> Dict[str, Any]:
    if ((tv_type == 'appletv') and (top_count == 1)) or ((tv_type == 'samsung') and (prior_name == 'rockyou-5gram') and (top_count in (1, 10))):
        return dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0)
    return dict()


def main(folder: str, tv_types: List[str], output_file: Optional[str]):
    # Holds the accuracy for every TV type, prior, and top-K cutoff
    tv_accuracy: Dict[str, Dict[str, List[float]]] = {name: dict() for name in tv_types}

    # Compute the baseline accuracy as would be done with random guessing
    baseline_accuracy: List[float] = compute_baseline_accuracy(baseline_size=PHPBB_COUNT, top_counts=TOP)

    for tv_type in tv_types:
        tv_accuracy[tv_type] = dict()
        results = get_accuracy_results(folder=folder, tv_type=tv_type)

        # Compute the accuracy for each prior and TV type
        target_top = TOP.index(100)

        print('==========')

        for prior_name in PASSWORD_PRIORS:
            correct_counts: List[float] = [0 for _ in TOP]
            total_count = 0

            user_accuracy: List[int] = []

            for subject in results.correct[prior_name].keys():
                for top_idx in range(len(TOP)):
                    correct_counts[top_idx] += results.correct[prior_name][subject][top_idx]

                total_count += results.total[prior_name][subject]
                user_accuracy.append(100.0 * (results.correct[prior_name][subject][target_top] / results.total[prior_name][subject]))

            # Compute the 95% confidence intervals for both priors
            mean_accuracy, stddev_accuracy = np.mean(user_accuracy), np.std(user_accuracy)
            tv_accuracy[tv_type][prior_name] = [100.0 * (correct / total_count) for correct in correct_counts]
            error_margin = get_standard_error(stddev=stddev_accuracy, num_users=len(user_accuracy))
            lower, upper = mean_accuracy - error_margin, mean_accuracy + error_margin

            print('{}, {} Top 100. Mean -> {:.4f}, Std -> {:.4f}, MoE -> {:.4f}, 95% Interval -> ({:.4f}, {:.4f})'.format(tv_type, prior_name, mean_accuracy, stddev_accuracy, error_margin, lower, upper))

        # Print the results based on the password type
        for prior_name in PASSWORD_PRIORS:
            print('TV: {}, PRIOR: {}'.format(tv_type, prior_name))
            print_as_table(results.stratified_correct[prior_name], results.stratified_total[prior_name], top_counts=TOP)
            print('==========')

        # Compare the results to the baseline
        for prior_name in PASSWORD_PRIORS:
            accuracy_list = tv_accuracy[tv_type][prior_name]

            if len(accuracy_list) > 0:
                comp = min([acc / base for acc, base in zip(accuracy_list, baseline_accuracy)])
                print('Comparison between {} and baseline: {:.4f}x'.format(prior_name, comp))

    # Plot the results
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for tv_type in tv_types:
            for prior_idx, prior_name in enumerate(PASSWORD_PRIORS):
                label = '{}, {}'.format(TV_LABELS[tv_type], PASSWORD_PRIOR_LABELS[prior_name])
                ax.plot(TOP, tv_accuracy[tv_type][prior_name], marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=label, color=TV_COLORS[tv_type][prior_idx])

                for top_idx, top_count in enumerate(TOP):
                    xoffset, yoffset = get_label_offset(tv_type=tv_type, prior_name=prior_name, top_count=top_count)
                    bbox = get_bbox(tv_type=tv_type, prior_name=prior_name, top_count=top_count)

                    if top_count in TOP_TO_ANNOTATE:
                        accuracy = tv_accuracy[tv_type][prior_name][top_idx]

                        if (tv_type == 'appletv') and (prior_name == 'rockyou-5gram') and (top_count == 10):
                            continue

                        if len(bbox) > 0:
                            ax.annotate('{:.2f}%'.format(accuracy), xy=(top_count, accuracy), xytext=(top_count + xoffset, accuracy + yoffset), size=AXIS_SIZE, bbox=bbox)
                        else:
                            ax.annotate('{:.2f}%'.format(accuracy), xy=(top_count, accuracy), xytext=(top_count + xoffset, accuracy + yoffset), size=AXIS_SIZE)

        # Plot the baseline and the final data label
        ax.plot(TOP, baseline_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Random Guess', color=BASELINE_COLOR)

        last_accuracy = baseline_accuracy[-1]
        last_count = TOP[-1]
        ax.annotate('{:.2f}%'.format(last_accuracy), xy=(last_count, last_accuracy), xytext=(last_count - 37.0, last_accuracy + 2.0), size=AXIS_SIZE, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0))

        ax.set_ylim(top=70)

        legend_pos = (1.0, 0.75)
        ax.legend(fontsize=AXIS_SIZE, bbox_to_anchor=legend_pos)
        ax.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax.yaxis.set_tick_params(labelsize=TITLE_SIZE)

        ax.set_title('Password Accuracy on Human Users', size=TITLE_SIZE)
        ax.set_xlabel('Guess Cutoff', size=TITLE_SIZE)
        ax.set_ylabel('Accuracy (%)', size=TITLE_SIZE)

        plt.tight_layout()

        # Show or save the result
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, transparent=True, bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser('Script to display results for password recovery on users.')
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--tv-types', type=str, required=True, choices=['samsung', 'appletv'], help='The name of the TV types.', nargs='+')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    main(folder=args.user_folder, tv_types=args.tv_types, output_file=args.output_file)
