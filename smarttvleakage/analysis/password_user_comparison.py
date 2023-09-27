import os.path
from argparse import ArgumentParser
from typing import List, Tuple, Dict

from smarttvleakage.analysis.utils import compute_rank
from smarttvleakage.utils.file_utils import iterate_dir, read_json


TOP = 100
GUESSES_NAME = 'recovered_{}_passwords_{}.json'
LABELS_NAME = '{}_passwords_labels.json'
PRIORS = ['phpbb', 'rockyou-5gram']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--tv-type', type=str, required=True, choices=['samsung', 'appletv'], help='The name of the TV type.')
    args = parser.parse_args()

    for prior_name in PRIORS:
        password_ranks = dict()  # { password -> { subject -> rank }

        for subject_folder in iterate_dir(args.user_folder):
            # Read the serialized password guesses
            guesses_path = os.path.join(subject_folder, GUESSES_NAME.format(args.tv_type, prior_name))
            if not os.path.exists(guesses_path):
                continue

            subject_name = os.path.split(subject_folder)[-1]
            saved_password_guesses = read_json(guesses_path)
            password_guesses = [entry['guesses'] for entry in saved_password_guesses]

            # Read the labels
            labels_path = os.path.join(subject_folder, LABELS_NAME.format(args.tv_type))
            password_labels = read_json(labels_path)['labels']

            for idx, guesses in enumerate(password_guesses):
                label = password_labels[idx]
                rank = compute_rank(guesses, label=label)
                
                if label not in password_ranks:
                    password_ranks[label] = dict()

                prev_rank = password_ranks[label].get(subject_name, -1)
                if prev_rank == -1:
                    password_ranks[label][subject_name] = rank

        recovered_count = 0
        total_count = 0

        for password, recovery in password_ranks.items():
            total_count += 1
            if len(recovery) <= 1:
                continue

            recovered_count += int(all(((rank <= TOP) and (rank >= 1)) for rank in recovery.values()))

        recovery_rate = (recovered_count / total_count) * 100.0
        print('{} Prior. Duplicate Recovery Rate: {:.5f}\% ({} / {})'.format(prior_name, recovery_rate, recovered_count, total_count))
