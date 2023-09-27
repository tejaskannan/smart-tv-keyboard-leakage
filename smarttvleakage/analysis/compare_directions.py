import os.path
from argparse import ArgumentParser
from typing import List

from smarttvleakage.utils.file_utils import read_json, iterate_dir


WITH_DIRECTIONS = 'recovered_samsung_passwords_{}.json'
WITHOUT_DIRECTIONS = 'no_directions_recovered_samsung_passwords_{}.json'
LABELS_FILE = 'samsung_passwords_labels.json'


def compute_rank_with_score(guesses: List[str], scores: List[float], label: str) -> int:
    # First, find the score of the target. If it is not found, then return a rank of -1
    correct_score = -1.0

    for guess, score in zip(guesses, scores):
        if guess == label:
            correct_score = score
            break

    if correct_score < 0.0:
        return -1

    # Count the number of elements with score strictly less than the correct score
    rank = 0
    while (scores[rank] < correct_score):
        rank += 1

    return rank + 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--user-folder', type=str, required=True)
    parser.add_argument('--prior', type=str, required=True, choices=['phpbb', 'rockyou-5gram'])
    args = parser.parse_args()

    num_improved = 0
    num_better = 0
    total_found = 0
    total_with = 0
    total_without = 0
    total_count = 0

    for subject_folder in iterate_dir(args.user_folder):
        with_directions_log = read_json(os.path.join(subject_folder, WITH_DIRECTIONS.format(args.prior)))
        without_directions_log = read_json(os.path.join(subject_folder, WITHOUT_DIRECTIONS.format(args.prior)))
        labels = read_json(os.path.join(subject_folder, LABELS_FILE))['labels']

        assert len(with_directions_log) == len(without_directions_log), 'Found {} without and {} with directions'.format(len(without_directions_log), len(with_directions_log))

        for with_result, without_result, label in zip(with_directions_log, without_directions_log, labels):
            with_rank = compute_rank_with_score(with_result['guesses'], scores=with_result['scores'], label=label)
            without_rank = compute_rank_with_score(without_result['guesses'], scores=without_result['scores'], label=label)

            if (with_rank > 0) and (without_rank <= 0):
                num_improved += 1
                num_better += 1
            elif (with_rank > 0):
                num_improved += int(with_rank <= without_rank)
                num_better += int(with_rank < without_rank)

                if (with_rank < without_rank):
                    print('With: {}, Without: {}, Password: {}'.format(with_rank, without_rank, label))

            if (with_rank > 0) or (without_rank > 0):
                total_found += 1

            total_with += int(with_rank > 0)
            total_without += int(without_rank > 0)
            total_count += 1

    print('Num Improved: {} / {} ({:.5f})'.format(num_improved, total_found, num_improved / total_found))
    print('Num Better: {} / {} ({:.5f})'.format(num_better, total_found, num_better / total_found))
    print('Total Counts. With: {}, Without: {}, Overall: {}'.format(total_with, total_without, total_count))
