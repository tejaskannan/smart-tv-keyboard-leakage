import io
import numpy as np
import string
import time
from argparse import ArgumentParser
from typing import Any, Dict, Iterable

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.file_utils import save_jsonl_gz


def create_records(input_path: str, max_num_records: int, keyboard_type: KeyboardType) -> Iterable[Dict[str, Any]]:
    words: List[str] = []

    with open(input_path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all((c in string.printable) and (c != '_') for c in word):
                words.append(word)

    num_words = len(words)

    #word_indices = np.arange(num_words)
    #rand = np.random.RandomState(seed=5481)
    #word_indices = rand.choice(num_words, size=max_num_records, replace=False)

    print('Read {} passwords. Generating dataset...'.format(num_words))
    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)

    for count, word in enumerate(words):
        if count >= max_num_records:
            break

        try:
            moves = findPath(word, True, True, 0.0, 1.0, 0, keyboard)
            move_seq = [{'moves': m.num_moves, 'end_sound': m.end_sound} for m in moves]
            yield { 'target': word, 'move_seq': move_seq }

            if ((count + 1) % 10000) == 0:
                print('Completed {} records.'.format(count + 1), end='\r')
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-records', type=int, required=True)
    parser.add_argument('--keyboard-type', type=str, choices=['samung', 'apple_tv_password'], required=True)
    args = parser.parse_args()

    records = list(create_records(args.input_path, max_num_records=args.max_num_records, keyboard_type=KeyboardType[args.keyboard_type.upper()]))
    save_jsonl_gz(records, args.output_path)
