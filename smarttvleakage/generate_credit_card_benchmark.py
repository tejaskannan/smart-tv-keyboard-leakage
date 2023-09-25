"""
Script to generate move count sequences from credit card details.
"""
import numpy as np
import os.path
from argparse import ArgumentParser
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Iterable

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType, SuggestionsType
from smarttvleakage.utils.credit_card_detection import validate_credit_card_number
from smarttvleakage.utils.file_utils import save_json, iterate_dir, make_dir


CreditCardEntry = namedtuple('CreditCardEntry', ['ccn', 'cvv', 'zip_code', 'exp_month', 'exp_year'])
BATCH_SIZE = 500


def sample_random_zip_code(rand: np.random.RandomState, zip_codes: List[Tuple[str, int]]) -> str:
    zip_code_indices = np.arange(len(zip_codes))
    counts = list(map(lambda z: z[1], zip_codes))
    total_count = sum(counts)
    weights = [c / total_count for c in counts]

    selected_idx = int(rand.choice(zip_code_indices, size=1, p=weights))
    return zip_codes[selected_idx][0]


def read_zip_codes(path: str) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []

    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.split(' ')
            zip_code, count = tokens[0], int(tokens[1])
            results.append((zip_code, count))

    return results


def read_cc_entries(paths: List[str]) -> List[str]:
    records: List[str] = []

    for path in paths:
        with open(path, 'r') as fin:
            for line in fin:
                records.append(line.strip())

    return records


def generate_full_details(cc_record: str, zip_codes: List[Tuple[str, int]], rand: np.random.RandomState) -> CreditCardEntry:
    tokens = cc_record.split(',')

    expiration_tokens = tokens[1].split('/')
    exp_month = '{:02d}'.format(int(expiration_tokens[0]))
    exp_year = expiration_tokens[1]

    ccn = tokens[0]
    assert validate_credit_card_number(ccn), '{} is not a valid CCN'.format(ccn)

    return CreditCardEntry(ccn=tokens[0],
                           cvv=tokens[2],
                           zip_code=sample_random_zip_code(rand=rand, zip_codes=zip_codes),
                           exp_month=exp_month,
                           exp_year=exp_year)


def generate_records(input_paths: List[str], zip_code_path: str, rand: np.random.RandomState) -> Iterable[CreditCardEntry]:
    zip_codes = read_zip_codes(zip_code_path)

    for cc_record in read_cc_entries(input_paths):
        yield generate_full_details(cc_record=cc_record, zip_codes=zip_codes, rand=rand)


def create_move_sequence(target: str, keyboard: MultiKeyboardGraph, rand: np.random.RandomState) -> List[Dict[str, Any]]:
    use_shortcuts = (rand.uniform() < 0.5)
    use_wraparound = (rand.uniform() < 0.5)
    move_seq = findPath(target, use_shortcuts=use_shortcuts, use_wraparound=use_wraparound, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key='q')
    return [m.to_dict() for m in move_seq]


def save_batch(input_batch: List[Dict[str, Any]], label_batch: List[Dict[str, Any]], batch_idx: int, output_folder: str):

    output_folder = os.path.join(output_folder, 'part_{}'.format(batch_idx))
    make_dir(output_folder)

    inputs_file = os.path.join(output_folder, 'credit_card_details.json')
    labels_file = os.path.join(output_folder, 'credit_card_details_labels.json')

    inputs_dict = {
        'seq_type': 'credit_card',
        'tv_type': 'samsung',
        'move_sequences': input_batch,
        'suggestions_types': [SuggestionsType.STANDARD.name.lower() for _ in range(len(input_batch))]
    }
    save_json(inputs_dict, inputs_file)

    labels_dict = {
        'seq_type': 'credit_card',
        'labels': label_batch
    }
    save_json(labels_dict, labels_file)


def main(input_paths: str, zip_code_path: str, output_folder: str):
    # Make the output folder
    make_dir(output_folder)

    # Make the keyboard
    keyboard = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)

    # Hold all of the move count sequences
    input_batch: List[Dict[str, Any]] = []
    label_batch: List[Dict[str, str]] = []

    rand = np.random.RandomState(seed=32489)
    batch_idx = 0

    for record in generate_records(input_paths=input_paths, zip_code_path=zip_code_path, rand=rand):
        label_dict = {
            'credit_card': record.ccn,
            'security_code': record.cvv,
            'exp_month': record.exp_month,
            'exp_year': record.exp_year,
            'zip_code': record.zip_code
        }

        move_sequences = {
            'credit_card': create_move_sequence(record.ccn, keyboard, rand),
            'security_code': create_move_sequence(record.cvv, keyboard, rand),
            'exp_month': create_move_sequence(record.exp_month, keyboard, rand),
            'exp_year': create_move_sequence(record.exp_year, keyboard, rand),
            'zip_code': create_move_sequence(record.zip_code, keyboard, rand)
        }

        input_batch.append(move_sequences)
        label_batch.append(label_dict)

        if len(input_batch) >= BATCH_SIZE:
            print('Saving batch {}...'.format(batch_idx + 1), end='\r')
            save_batch(input_batch, label_batch, batch_idx=batch_idx, output_folder=output_folder)
            batch_idx += 1

            input_batch = []
            label_batch = []

    # Write out any outstanding inputs
    if len(input_batch) > 0:
        print('Saving batch {}...'.format(batch_idx + 1), end='\r')
        save_batch(input_batch, label_batch, batch_idx=batch_idx, output_folder=output_folder)

    print('\nDone.')


if __name__ == '__main__':
    parser = ArgumentParser('Script to generate move count sequences from credit card details.')
    parser.add_argument('--input-files', type=str, required=True, nargs='+', help='Paths of the .csv files containing credit card details.')
    parser.add_argument('--zip-code-file', type=str, required=True, help='Path to a text file containing ZIP codes (with populations).')
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    main(input_paths=args.input_files, zip_code_path=args.zip_code_file, output_folder=args.output_folder)
