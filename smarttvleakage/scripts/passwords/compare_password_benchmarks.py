import os.path
from argparse import ArgumentParser
from typing import Set

from smarttvleakage.utils.file_utils import read_json, iterate_dir


def get_passwords(folder: str) -> Set[str]:
    passwords: Set[str] = set()

    for part in iterate_dir(folder):
        # Get the path to the labels
        labels_path = os.path.join(part, 'samsung_passwords_labels.json')
        if not os.path.exists(labels_path):
            labels_path = os.path.join(part, 'appletv_passwords_labels.json')

        labels = read_json(labels_path)['labels']
        passwords.update(labels)

    return passwords

def main(folder0: str, folder1: str):

    # Get the first list of passwords
    passwords0 = get_passwords(folder0)
    passwords1 = get_passwords(folder1)

    intersection = passwords0.intersection(passwords1)

    print('# Passwords 0: {}, # Passwords 1: {}, # Intersection: {}'.format(len(passwords0), len(passwords1), len(intersection)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder0', type=str, required=True)
    parser.add_argument('--folder1', type=str, required=True)
    args = parser.parse_args()

    main(args.folder0, args.folder1)
