import io
import numpy as np
import string
from argparse import ArgumentParser
from typing import Any, Dict, Iterable

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.file_utils import save_jsonl_gz


def create_records(input_path: str, max_num_records: int) -> Iterable[Dict[str, Any]]:
    words: List[str] = []

    with open(input_path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all((c in string.printable) and (c != '_') for c in word):
                words.append(word)

    num_words = len(words)
    rand = np.random.RandomState(seed=5481)
    word_indices = rand.choice(num_words, size=max_num_records, replace=False)

    print('Read {} passwords. Generating dataset...'.format(num_words))

    for count, word_idx in enumerate(word_indices):
        word = words[word_idx]
        try:
            moves = findPath(word, True, True, 0.0, 1.0, 0)
            move_seq = [{'moves': m.num_moves, 'end_sound': m.end_sound} for m in moves]

            yield { 'target': word, 'move_seq': move_seq }

            if ((count + 1) % 100) == 0:
                print('Completed {} records.'.format(count + 1), end='\r')
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-records', type=int, required=True)
    args = parser.parse_args()

    records = list(create_records(args.input_path, max_num_records=args.max_num_records))
    save_jsonl_gz(records, args.output_path)
