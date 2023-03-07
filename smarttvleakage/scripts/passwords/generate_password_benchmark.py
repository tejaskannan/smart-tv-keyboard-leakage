import io
import numpy as np
import string
import time
import os.path
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, List

from smarttvleakage.audio.data_types import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType
from smarttvleakage.utils.file_utils import save_json, iterate_dir, make_dir


MIN_PASSWORD_LENGTH = 8
BATCH_SIZE = 500


def create_records(input_path: str, max_num_records: int, keyboard_type: KeyboardType) -> Iterable[Dict[str, Any]]:
    words: List[str] = []

    with open(input_path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all((c in string.printable) for c in word) and (len(word) >= MIN_PASSWORD_LENGTH):
                words.append(word)

    num_words = len(words)

    word_indices = np.arange(num_words)
    rand = np.random.RandomState(seed=5481)
    word_indices = rand.choice(num_words, size=max_num_records, replace=False)

    print('Read {} passwords. Generating dataset...'.format(num_words))
    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)
    start_key = START_KEYS[keyboard.get_start_keyboard_mode()]

    for count, word_idx in enumerate(word_indices):
        if count >= max_num_records:
            break

        word = words[word_idx]

        try:
            moves = findPath(word, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
            move_seq = [m.to_dict() for m in moves]
            yield { 'target': word, 'move_seq': move_seq }

            if ((count + 1) % BATCH_SIZE) == 0:
                print('Completed {} records.'.format(count + 1), end='\r')
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()


def save_batch(move_sequences: List[List[Move]], labels: List[str], batch_idx: int, output_folder: str):
    output_dir = os.path.join(output_folder, 'part_{}'.format(batch_idx))
    make_dir(output_dir)

    passwords_path = os.path.join(output_dir, '{}_passwords.json'.format(tv_type.name.lower()))
    passwords = {
        'tv_type': tv_type.name.lower(),
        'seq_type': 'standard',
        'move_sequences': move_batch
    }
    save_json(passwords, passwords_path)

    labels_path = os.path.join(output_dir, '{}_passwords_labels.json'.format(tv_type.name.lower()))
    labels = {
        'labels': labels_batch
    }
    save_json(labels, labels_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--max-num-records', type=int, required=True)
    parser.add_argument('--tv-type', type=str, choices=['samsung', 'apple_tv'], required=True)
    args = parser.parse_args()

    move_batch: List[List[Move]] = []
    labels_batch: List[str] = []
    batch_idx = 0

    tv_type = SmartTVType[args.tv_type.upper()]

    if tv_type == SmartTVType.SAMSUNG:
        keyboard_type = KeyboardType.SAMSUNG
    elif tv_type == SmartTVType.APPLE_TV:
        keyboard_type = KeyboardType.APPLE_TV_PASSWORD
    else:
        raise ValueError('Unknown tv type: {}'.format(tv_type))

    for record in create_records(args.input_path, max_num_records=args.max_num_records, keyboard_type=keyboard_type):
        move_batch.append(record['move_seq'])
        labels_batch.append(record['target'])

        if len(move_batch) >= BATCH_SIZE:
            save_batch(move_batch, labels_batch, batch_idx=batch_idx, output_folder=args.output_folder)
            print('\nSaving batch {}...'.format(batch_idx + 1))

            batch_idx += 1
            move_batch = []
            labels_batch = []

    if len(move_batch) > 0:
        save_batch(move_batch, labels_batch, batch_idx=batch_idx, output_folder=args.output_folder)

