import io
import numpy as np
import string
import time
from argparse import ArgumentParser
from typing import Any, Dict, Iterable

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary.dictionaries import NgramDictionary, restore_dictionary
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import move_seq_to_vector


SCORE_THRESHOLD = 1e5


def create_records(input_path: str, max_num_records: int, dictionary: NgramDictionary) -> Iterable[Dict[str, Any]]:
    words: List[str] = []

    with open(input_path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all((c in string.printable) and (c != '_') for c in word):
                words.append(word)

    num_words = len(words)

    if num_words < max_num_records:
        word_indices = np.arange(num_words)
    else:
        rand = np.random.RandomState(seed=5481)
        word_indices = rand.choice(num_words, size=max_num_records, replace=False)

    print('Read {} passwords. Generating dataset...'.format(num_words))
    keyboard = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)

    for count, word_idx in enumerate(word_indices):
        word = words[word_idx]
        word_score = dictionary.get_score_for_string(word)

        if word_score > SCORE_THRESHOLD:
            continue

        try:
            moves = findPath(word, False, False, 0.0, 1.0, 0, keyboard)
            move_vector = move_seq_to_vector(moves, tv_type=SmartTVType.SAMSUNG)
            yield { 'target': word, 'move_seq': move_vector, 'score': word_score}

            wraparound_moves = findPath(word, False, True, 0.0, 1.0, 0, keyboard)
            wraparound_vector = move_seq_to_vector(wraparound_moves, tv_type=SmartTVType.SAMSUNG)

            if wraparound_vector != move_vector:
                yield { 'target': word, 'move_seq': wraparound_vector, 'score': word_score }

            shortcut_moves = findPath(word, True, True, 0.0, 1.0, 0, keyboard)
            shortcut_vector = move_seq_to_vector(shortcut_moves, tv_type=SmartTVType.SAMSUNG)

            if (shortcut_vector != wraparound_vector) and (shortcut_vector != move_vector):
                yield { 'target': word, 'move_seq': shortcut_vector, 'score': word_score }

            if ((count + 1) % 10000) == 0:
                print('Completed {} records.'.format(count + 1), end='\r')
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-records', type=int, required=True)
    args = parser.parse_args()

    print('Loading the dictionary...')
    dictionary = restore_dictionary(args.dictionary_path)

    records = list(create_records(args.input_path, max_num_records=args.max_num_records, dictionary=dictionary))

    print('Saving {} Records...'.format(len(records)))
    save_jsonl_gz(records, args.output_path)
