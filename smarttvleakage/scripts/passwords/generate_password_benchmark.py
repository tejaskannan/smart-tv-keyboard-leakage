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
from smarttvleakage.utils.constants import KeyboardType, SmartTVType, SuggestionsType
from smarttvleakage.utils.file_utils import save_json, iterate_dir, make_dir, read_pickle_gz
from smarttvleakage.suggestions_model.determine_autocomplete import classify_moves


MIN_PASSWORD_LENGTH = 8
BATCH_SIZE = 500


def create_records(input_path: str, max_num_records: int, suggestions_model: Any, tv_type: SmartTVType, keyboard_type: KeyboardType) -> Iterable[Dict[str, Any]]:
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

            if tv_type == SmartTVType.SAMSUNG:
                suggestions_type =  SuggestionsType.SUGGESTIONS if classify_moves(suggestions_model, moves) else SuggestionsType.STANDARD
            else:
                suggestions_type = SuggestionsType.STANDARD

            move_seq = [m.to_dict() for m in moves]
            yield { 'target': word, 'move_seq': move_seq, 'suggestions_type': suggestions_type.name.lower() }

            if ((count + 1) % BATCH_SIZE) == 0:
                print('Completed {} records.'.format(count + 1), end='\r')
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()


def save_batch(move_sequences: List[List[Move]], suggestions_types: List[str], labels: List[str], batch_idx: int, tv_type: SmartTVType, output_folder: str):
    output_dir = os.path.join(output_folder, 'part_{}'.format(batch_idx))
    make_dir(output_dir)

    output_name = tv_type.name.lower().replace('_', '')

    passwords_path = os.path.join(output_dir, '{}_passwords.json'.format(output_name))
    passwords = {
        'tv_type': tv_type.name.lower(),
        'seq_type': 'standard',
        'move_sequences': move_batch,
        'suggestions_types': suggestions_types
    }
    save_json(passwords, passwords_path)

    labels_path = os.path.join(output_dir, '{}_passwords_labels.json'.format(output_name))
    labels = {
        'labels': labels_batch
    }
    save_json(labels, labels_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True, help='Path to a list of passwords to consider.')
    parser.add_argument('--output-folder', type=str, required=True, help='The folder in which to write output batches.')
    parser.add_argument('--max-num-records', type=int, required=True, help='The maximum number of records to consider.')
    parser.add_argument('--suggestions-model', type=str, required=True, help='Model to classify the keyboard suggestions type (needed for samsung only).')
    parser.add_argument('--tv-type', type=str, choices=['samsung', 'apple_tv'], required=True, help='The type of Smart TV to generate move sequences for.')
    args = parser.parse_args()

    move_batch: List[List[Move]] = []
    suggestions_types_batch: List[str] = []
    labels_batch: List[str] = []
    batch_idx = 0

    tv_type = SmartTVType[args.tv_type.upper()]

    if tv_type == SmartTVType.SAMSUNG:
        keyboard_type = KeyboardType.SAMSUNG
    elif tv_type == SmartTVType.APPLE_TV:
        keyboard_type = KeyboardType.APPLE_TV_PASSWORD
    else:
        raise ValueError('Unknown tv type: {}'.format(tv_type))

    # Read the keyboard model
    suggestions_model = read_pickle_gz(args.suggestions_model)

    for record in create_records(args.input_path, max_num_records=args.max_num_records, suggestions_model=suggestions_model, tv_type=tv_type, keyboard_type=keyboard_type):
        move_batch.append(record['move_seq'])
        labels_batch.append(record['target'])
        suggestions_types_batch.append(record['suggestions_type'])

        if len(move_batch) >= BATCH_SIZE:
            save_batch(move_batch, labels=labels_batch, suggestions_types=suggestions_types_batch, tv_type=tv_type, batch_idx=batch_idx, output_folder=args.output_folder)
            print('\nSaving batch {}...'.format(batch_idx + 1))

            batch_idx += 1
            move_batch = []
            labels_batch = []
            suggestions_types_batch = []

    if len(move_batch) > 0:
        save_batch(move_batch, labels=labels_batch, suggestions_types=suggestions_types_batch, tv_type=tv_type, batch_idx=batch_idx, output_folder=args.output_folder)
