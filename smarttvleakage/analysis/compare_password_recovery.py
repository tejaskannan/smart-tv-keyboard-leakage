import os.path
from argparse import ArgumentParser
from typing import List

from smarttvleakage.utils.file_utils import read_json


def get_rank(guesses: List[str], target: str) -> int:
    for idx, guess in enumerate(guesses):
        if guess == target:
            return idx + 1

    return -1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--samsung-path', type=str, required=True)
    parser.add_argument('--appletv-path', type=str, required=True)
    args = parser.parse_args()

    samsung_recovered = read_json(args.samsung_path)
    appletv_recovered = read_json(args.appletv_path)

    samsung_folder, _ = os.path.split(args.samsung_path)
    labels_path = os.path.join(samsung_folder, 'samsung_passwords_labels.json')
    labels = read_json(labels_path)['labels']

    length = min(len(samsung_recovered), len(appletv_recovered))

    for idx in range(length):
        samsung_entry = samsung_recovered[idx]
        appletv_entry = appletv_recovered[idx]
        label = labels[idx]

        samsung_rank = get_rank(samsung_entry['guesses'], target=label)
        appletv_rank = get_rank(appletv_entry['guesses'], target=label)

        if (samsung_rank > 0) and (appletv_rank < 0):
            print('Label: {}, Samsung Rank: {}, Apple TV Rank: {}'.format(label, samsung_rank, appletv_rank))

