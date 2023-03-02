import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple, Dict

from smarttvleakage.utils.file_utils import iterate_dir, read_json


TOP = [1, 5, 10, 50, 100, 500]
GUESSES_NAME = 'recovered_{}_passwords_{}.json'
LABELS_NAME = '{}_passwords_labels.json'
PRIORS = ['phpbb', 'rockyou-5gram']
PHPBB_COUNT = 184388
PRIOR_LABELS = {
    'phpbb': 'PHPBB Prior',
    'rockyou-5gram': 'Rockyou Prior'
}


def top_k_accuracy(password_guesses: List[List[str]], targets: List[str], top: int) -> Tuple[int, int]:
    assert top >= 1, 'Must provide a positive `top` count'
    assert len(password_guesses) == len(targets), 'Must provide the same number of guesses as targets'

    recovered_count = 0

    for guesses, target in zip(password_guesses, targets):
        for rank, guess in enumerate(guesses):
            if rank >= top:
                break
            elif guess == target:
                recovered_count += 1
                break
    
    return recovered_count, len(targets)


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

            # Get the correct counts for each `top` entry
            for top_idx, top_count in enumerate(TOP):
                correct, total = top_k_accuracy(password_guesses, targets=password_labels, top=top_count)
                correct_counts[prior_name][subject_name][top_idx] += correct
                total_counts[prior_name][subject_name][top_idx] += total

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
            accuracy_list: List[float] = [100.0 * (count / total) for count, total in zip(merged_count_list, merged_total_list)]
            accuracy_dict[prior_name] = accuracy_list

    baseline_accuracy: List[float] = [100.0 * (top / PHPBB_COUNT) for top in TOP]

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots()

        for prior_name in PRIORS:
            if prior_name not in accuracy_dict:
                continue

            ax.plot(TOP, accuracy_dict[prior_name], marker='o', linewidth=3, markersize=8, label=PRIOR_LABELS[prior_name])

            # Write the data labels
            for idx, (topk, accuracy) in enumerate(zip(TOP, accuracy_dict[prior_name])):
                if prior_name == 'phpbb':
                    if topk <= 10:
                        xoffset = 0
                    elif topk <= 100:
                        xoffset = -10
                    else:
                        xoffset = -200

                    yoffset = -4.0
                elif prior_name == 'rockyou-5gram':
                    if (topk > 1) and (topk <= 10):
                        xoffset = -1
                    elif topk <= 100:
                        xoffset = 0
                    else:
                        xoffset = -200

                    yoffset = 3.0
                else:
                    raise ValueError('Unknown prior name: {}'.format(prior_name))

                ax.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=12)

        ax.plot(TOP, baseline_accuracy, marker='o', linewidth=3, markersize=8, label='Random Guess')

        for idx, (topk, accuracy) in enumerate(zip(TOP, baseline_accuracy)):
            if topk <= 10:
                xoffset = 0
            elif topk <= 100:
                xoffset = -10
            else:
                xoffset = -200

            yoffset = 3.0

            if accuracy >= 0.01:
                ax.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=12)

        #ax.set_xticks(TOP)
        ax.set_xscale('log')
        ax.legend()

        ax.set_title('Password Top-K Accuracy for Human Users', size=16)
        ax.set_xlabel('Guess Cutoff (K)', size=14)
        ax.set_ylabel('Accuracy (%)', size=14)

        # Show or save the result
        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
