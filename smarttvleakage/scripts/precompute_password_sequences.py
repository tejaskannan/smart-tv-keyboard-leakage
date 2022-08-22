import io
import numpy as np
import string
import time
import sqlite3
from argparse import ArgumentParser
from typing import Any, Dict, Iterable, List, Tuple, Optional

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary.dictionaries import NgramDictionary, restore_dictionary
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import move_seq_to_vector


SCORE_THRESHOLD = 1e5


def create_records(input_path: str, max_num_records: Optional[int], dictionary: NgramDictionary, keyboard_type: KeyboardType) -> Iterable[Dict[str, Any]]:
    words: List[str] = []

    if keyboard_type == KeyboardType.SAMSUNG:
        tv_type = SmartTVType.SAMSUNG
    else:
        tv_type = SmartTVType.APPLE_TV

    with open(input_path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if len(word) == 0:
                continue

            if all((c in string.printable) and (c != '_') for c in word):
                words.append(word)

    print('Read {} passwords. Generating dataset...'.format(len(words)))
    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)

    for idx, word in enumerate(words):
        if (max_num_records is not None) and (idx >= max_num_records):
            break

        if ((idx + 1) % 10000) == 0:
            print('Completed {} records.'.format(idx + 1), end='\r')

        word_score = dictionary.get_score_for_string(word)
        if word_score > SCORE_THRESHOLD:
            continue

        try:
            moves = findPath(word, False, False, 0.0, 1.0, 0, keyboard)
            move_vector = move_seq_to_vector(moves, tv_type=tv_type)

            if all(m.num_moves is not None for m in moves):
                yield { 'target': word, 'move_seq': move_vector, 'score': word_score}

            wraparound_moves = findPath(word, False, True, 0.0, 1.0, 0, keyboard)
            wraparound_vector = move_seq_to_vector(wraparound_moves, tv_type=tv_type)

            if wraparound_vector != move_vector and all(m.num_moves is not None for m in wraparound_moves):
                yield { 'target': word, 'move_seq': wraparound_vector, 'score': word_score }

            shortcut_moves = findPath(word, True, True, 0.0, 1.0, 0, keyboard)
            shortcut_vector = move_seq_to_vector(shortcut_moves, tv_type=tv_type)

            if (shortcut_vector != wraparound_vector) and (shortcut_vector != move_vector) and all(m.num_moves is not None for m in shortcut_moves):
                yield { 'target': word, 'move_seq': shortcut_vector, 'score': word_score }
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    print()


def make_sql_batch(records_batch: List[Dict[str, Any]]) -> List[Tuple[str, str, float]]:
    result: List[Tuple[str, str, float]] = []

    for record in records_batch:
        sql_element = (record['idx'], record['move_seq'], record['target'], record['score'])
        result.append(sql_element)

    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-records', type=int)
    parser.add_argument('--keyboard-type', type=str, choices=[t.name.lower() for t in KeyboardType], required=True)
    args = parser.parse_args()

    print('Loading the dictionary...')
    dictionary = restore_dictionary(args.dictionary_path)

    keyboard_type = KeyboardType[args.keyboard_type.upper()]

    # Open the database connection
    conn = sqlite3.connect(args.output_path)
    cursor = conn.cursor()

    batch: List[Dict[str, Any]] = []
    for idx, record in enumerate(create_records(args.input_path, max_num_records=args.max_num_records, dictionary=dictionary, keyboard_type=keyboard_type)):
        record['idx'] = idx
        batch.append(record)

        if len(batch) % 1000 == 0:
            sql_batch = make_sql_batch(batch)
            cursor.executemany('INSERT INTO passwords VALUES(?, ?, ?, ?)', sql_batch)
            conn.commit()
            batch = []

    if len(batch) > 0:
        sql_batch = make_sql_batch(batch)
        cursor.executemany('INSERT INTO passwords VALUES(?, ?, ?, ?)', sql_batch)
        conn.commit()
