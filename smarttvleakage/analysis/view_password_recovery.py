"""
Script to print out the recovery results.
"""
from argparse import ArgumentParser

from smarttvleakage.analysis.utils import compute_rank
from smarttvleakage.utils.file_utils import read_json


def main(recovery_path: str, labels_path: str):
    recovery_results = read_json(recovery_path)
    labels = read_json(labels_path)['labels']

    found: List[Tuple[str, int]] = []
    not_found: List[str] = []

    for target, recovery_record in zip(labels, recovery_results):
        guesses = recovery_record['guesses']
        rank = compute_rank(guesses=guesses, label=target)

        if rank == -1:
            not_found.append(target)
        else:
            found.append((target, rank))

    print('Found: {}'.format(', '.join(map(lambda t: '{} (rank {})'.format(t[0], t[1]), found))))
    print('Not Found: {}'.format(' , '.join(not_found)))

    accuracy = len(found) / len(labels)
    print('Recovery Accuracy: {:.4f}% ({} / {})'.format(accuracy * 100.0, len(found), len(labels)))


if __name__ == '__main__':
    parser = ArgumentParser('Script to display recovery results.')
    parser.add_argument('--recovery-file', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    args = parser.parse_args()

    main(recovery_path=args.recovery_file,
         labels_path=args.labels_file)
