import io
import numpy as np
import string
import time
import os.path
import re
from argparse import ArgumentParser
from collections import Counter
from typing import Any, Dict, Iterable, List

from smarttvleakage.analysis.utils import has_special, has_number, has_uppercase
from smarttvleakage.audio.data_types import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType, SuggestionsType, SUGGESTIONS_CUTOFF
from smarttvleakage.utils.file_utils import save_json, iterate_dir, make_dir, read_pickle_gz
from smarttvleakage.suggestions_model.determine_autocomplete import classify_moves


MIN_PASSWORD_LENGTH = 8
BATCH_SIZE = 500


def sample_list(word_list: List[str], num_records: int, rand: np.random.RandomState) -> List[str]:
    selected_indices = rand.choice(len(word_list), size=num_records, replace=False)
    return [word_list[idx] for idx in selected_indices]


def create_records(input_path: str, max_num_records: int, suggestions_model: Any, tv_type: SmartTVType, keyboard_type: KeyboardType) -> Iterable[Dict[str, Any]]:
    special_words: List[str] = []
    upper_words: List[str] = []
    number_words: List[str] = []
    lower_words: List[str] = []

    with open(input_path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all((c in string.printable) for c in word) and (len(word) >= MIN_PASSWORD_LENGTH):
                # Split words into mutually exclusive groups to avoid sampling the same password more than once
                if has_special(word):
                    special_words.append(word)
                elif has_number(word):
                    number_words.append(word)
                elif has_uppercase(word):
                    upper_words.append(word)
                else:
                    lower_words.append(word)

    # Create the list of words
    split_size = int(max_num_records / 4)

    rand = np.random.RandomState(seed=2890)
    words: List[str] = []

    words.extend(sample_list(special_words, split_size, rand=rand))
    words.extend(sample_list(number_words, split_size, rand=rand))
    words.extend(sample_list(upper_words, split_size, rand=rand))
    words.extend(sample_list(lower_words, split_size, rand=rand))

    assert len(words) == max_num_records, 'Found {} words (expected {})'.format(len(words), max_num_records)

    num_words = len(words)
    word_indices = np.arange(num_words)
    rand.shuffle(word_indices)

    password_type_counter: Counter = Counter()

    print('Read {} passwords. Generating dataset...'.format(num_words))
    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)
    start_key = START_KEYS[keyboard.get_start_keyboard_mode()]

    for count, word_idx in enumerate(word_indices):
        if count >= max_num_records:
            break

        word = words[word_idx]

        try:
            use_shortcuts = (rand.uniform() < 0.5) or (tv_type == SmartTVType.APPLE_TV)
            use_wraparound = (rand.uniform() < 0.5) 
            use_done = (rand.uniform() < 0.5)
            moves = findPath(word, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound, use_done=use_done, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)

            if (tv_type == SmartTVType.SAMSUNG) and (keyboard_type != KeyboardType.ABC):
                suggestions_type =  SuggestionsType.SUGGESTIONS if classify_moves(suggestions_model, moves, cutoff=SUGGESTIONS_CUTOFF) else SuggestionsType.STANDARD
            else:
                suggestions_type = SuggestionsType.STANDARD

            move_seq = [m.to_dict() for m in moves]
            yield { 'target': word, 'move_seq': move_seq, 'suggestions_type': suggestions_type.name.lower() }

            if has_special(word):
                password_type_counter['special'] += 1

            if has_number(word):
                password_type_counter['number'] += 1

            if has_uppercase(word):
                password_type_counter['upper'] += 1

            if ((count + 1) % BATCH_SIZE) == 0:
                print('Completed {} records.'.format(count + 1), end='\r')
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()
    print(password_type_counter)


def save_batch(move_sequences: List[List[Move]], suggestions_types: List[str], labels: List[str], batch_idx: int, tv_type: SmartTVType, keyboard_type: KeyboardType, output_folder: str):
    output_dir = os.path.join(output_folder, 'part_{}'.format(batch_idx))
    make_dir(output_dir)

    output_name = tv_type.name.lower().replace('_', '')

    passwords_path = os.path.join(output_dir, '{}_passwords.json'.format(output_name))
    passwords = {
        'tv_type': tv_type.name.lower(),
        'keyboard_type': keyboard_type.name.lower(),
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
    parser.add_argument('--keyboard-type', type=str, help='The optional name of the keyboard to use. Overrides the default specified by the TV type.')
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

    if args.keyboard_type is not None:
        keyboard_type = KeyboardType[args.keyboard_type.upper()]

    # Read the keyboard model
    suggestions_model = read_pickle_gz(args.suggestions_model)

    for record in create_records(args.input_path, max_num_records=args.max_num_records, suggestions_model=suggestions_model, tv_type=tv_type, keyboard_type=keyboard_type):
        move_batch.append(record['move_seq'])
        labels_batch.append(record['target'])
        suggestions_types_batch.append(record['suggestions_type'])

        if len(move_batch) >= BATCH_SIZE:
            save_batch(move_sequences=move_batch,
                       labels=labels_batch,
                       suggestions_types=suggestions_types_batch,
                       tv_type=tv_type,
                       keyboard_type=keyboard_type,
                       batch_idx=batch_idx,
                       output_folder=args.output_folder)
            print('\nSaving batch {}...'.format(batch_idx + 1))

            batch_idx += 1
            move_batch = []
            labels_batch = []
            suggestions_types_batch = []

    if len(move_batch) > 0:
        save_batch(move_sequences=move_batch,
                   labels=labels_batch,
                   suggestions_types=suggestions_types_batch,
                   tv_type=tv_type,
                   keyboard_type=keyboard_type,
                   batch_idx=batch_idx,
                   output_folder=args.output_folder)
