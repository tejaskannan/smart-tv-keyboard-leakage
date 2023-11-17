import os.path
import numpy as np
from argparse import ArgumentParser
from typing import List

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.analysis.compare_user_keyboard_accuracy import compare_against_optimal
from smarttvleakage.utils.constants import KeyboardType, SmartTVType
from smarttvleakage.utils.file_utils import iterate_dir, read_json


if __name__ == '__main__':
    parser = ArgumentParser('Finds the rate at which the user traverses an optimal path between keys when entering credit card numbers.')
    parser.add_argument('--user-folder', type=str, required=True, help='Path to the folder containing the user results.')
    args = parser.parse_args()

    samsung_keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)

    correct_count = 0
    total_count = 0
    distances: List[float] = []

    subject_accuracy: List[float] = []

    for subject_folder in iterate_dir(args.user_folder):
        # Get the move sequences for credit card numbers
        path = os.path.join(subject_folder, 'credit_card_details.json')
        credit_card_details = read_json(path)['move_sequences']
        ccn_seqs = [entry['credit_card'] for entry in credit_card_details]

        # Get the labels
        labels_path = os.path.join(subject_folder, 'credit_card_details_labels.json')
        labels = read_json(labels_path)['labels']
        ccn_labels = [entry['credit_card'] for entry in labels]

        correct, total, dist = compare_against_optimal(ccn_seqs, ccn_labels, keyboard=samsung_keyboard, tv_type=SmartTVType.SAMSUNG)

        correct_count += correct
        total_count += total
        distances.extend(dist)

    accuracy = correct_count / total_count
    print('Accuracy: {:.2f}% ({} / {})'.format(accuracy * 100.0, correct_count, total_count))
    print('Dist: {:.4f} ({:.4f})'.format(np.mean(distances), np.std(distances)))
