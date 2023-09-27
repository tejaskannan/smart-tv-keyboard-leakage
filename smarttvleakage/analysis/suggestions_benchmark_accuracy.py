"""
Utility tool to check the accuracy of suggestions classifiers on a benchmark.
"""
import os.path
from argparse import ArgumentParser
from collections import Counter

from smarttvleakage.audio.data_types import Move
from smarttvleakage.utils.file_utils import iterate_dir, read_pickle_gz, read_json
from smarttvleakage.suggestions_model.determine_autocomplete import classify_moves


def get_observed_accuracy(folder: str) -> float:
    num_correct = 0
    total_count = 0

    for batch_folder in iterate_dir(folder):
        batch_file = os.path.join(batch_folder, 'samsung_passwords.json')
        suggestions_types = read_json(batch_file)['suggestions_types']

        num_correct += sum([int(t == 'standard') for t in suggestions_types])
        total_count += len(suggestions_types)

    return 100.0 * (num_correct / total_count)


if __name__ == '__main__':
    parser = ArgumentParser('Script to measure the accuracy of classifying passwords on a set of emulated move count sequences.')
    parser.add_argument('--benchmark-folder', type=str, required=True, help='Path to the directory containing the password benchmark.')
    args = parser.parse_args()

    observed_accuracy = get_observed_accuracy(args.benchmark_folder)
    print('Observed Accuracy: {:.4f}%'.format(observed_accuracy))

    suggestions_model = read_pickle_gz(os.path.join('..', 'suggestions_model', 'suggestions_model.pkl.gz'))
    cutoffs = [0.5, 0.55, 0.6, 0.65]
    correct_counts: Counter = Counter()
    total_counts: Counter = Counter()

    for batch_idx, batch_folder in enumerate(iterate_dir(args.benchmark_folder)):
        if batch_idx >= 3:
            break

        batch_file = os.path.join(batch_folder, 'samsung_passwords.json')
        move_sequences = read_json(batch_file)['move_sequences']

        for cutoff in cutoffs:
            for move_seq in move_sequences:
                parsed_move_seq = [Move.from_dict(m) for m in move_seq]
                clf_result = classify_moves(suggestions_model, parsed_move_seq, cutoff=cutoff)
                correct_counts[cutoff] += int(clf_result == 0)
                total_counts[cutoff] += 1

    for cutoff in cutoffs:
        accuracy = 100.0 * (correct_counts[cutoff] / total_counts[cutoff])
        print('Cutoff: {:.2f}, Accuracy: {:.3f}%'.format(cutoff, accuracy))
