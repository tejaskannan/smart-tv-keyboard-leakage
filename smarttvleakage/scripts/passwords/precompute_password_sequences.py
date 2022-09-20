import os
import io
import numpy as np
import string
import time
import sqlite3
import sys
from argparse import ArgumentParser
from functools import partial
from more_itertools import chunked
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Iterable, List, Tuple, Optional

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.dictionary.dictionaries import NgramDictionary, restore_dictionary
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import move_seq_to_vector


SCORE_THRESHOLD = 1e5
BATCH_SIZE = 1000


def load_passwords(path: str, max_num_records: Optional[int]) -> Iterable[str]:
    words: List[str] = []

    with open(path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if len(word) == 0:
                continue

            if all(c in string.printable for c in word):
                words.append(word)

            if (max_num_records is not None) and (len(words) >= max_num_records):
                break

    return words


#def create_records(word: str, dictionary_path: str, keyboard_type: KeyboardType, keyboard: MultiKeyboardGraph) -> List[Dict[str, Any]]:
def create_records(word: str, dictionary: NgramDictionary, keyboard_type: KeyboardType, keyboard: MultiKeyboardGraph) -> List[Dict[str, Any]]:
    # Get the start key
    start_key = START_KEYS[keyboard.get_start_keyboard_mode()]

    # Set the dictionary characters
    dictionary.set_characters(keyboard.get_characters())

    # Get the score for this string
    word_score = dictionary.get_score_for_string(word, length=len(word))
    if word_score > SCORE_THRESHOLD:
        return []

    if word.lower() == 'hallo':
        print('{}: {}'.format(word, word_score))

    result: List[Dict[str, Any]] = []

    if keyboard_type == KeyboardType.APPLE_TV_PASSWORD:
        try:
            moves = findPath(word, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
            move_vector = move_seq_to_vector(moves, tv_type=tv_type)

            if all(m.num_moves is not None for m in moves):
                result.append({'target': word, 'move_seq': move_vector, 'score': word_score})
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))
    else:
        try:
            moves = findPath(word, use_shortcuts=False, use_wraparound=False, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
            move_vector = move_seq_to_vector(moves, tv_type=tv_type)

            if all(m.num_moves is not None for m in moves):
                result.append({'target': word, 'move_seq': move_vector, 'score': word_score})

            wraparound_moves = findPath(word, use_shortcuts=False, use_wraparound=True, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
            wraparound_vector = move_seq_to_vector(wraparound_moves, tv_type=tv_type)

            if wraparound_vector != move_vector and all(m.num_moves is not None for m in wraparound_moves):
                result.append({'target': word, 'move_seq': wraparound_vector, 'score': word_score})

            shortcut_moves = findPath(word, use_shortcuts=True, use_wraparound=True, use_done=False, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
            shortcut_vector = move_seq_to_vector(shortcut_moves, tv_type=tv_type)

            if (shortcut_vector != wraparound_vector) and (shortcut_vector != move_vector) and all(m.num_moves is not None for m in shortcut_moves):
                result.append({'target': word, 'move_seq': shortcut_vector, 'score': word_score})
        except AssertionError as ex:
            print('\nWARNING: Caught {} for {}'.format(ex, word))

    return result


def make_sql_batch(records_batch: List[Dict[str, Any]], record_idx: int) -> Tuple[List[Tuple[str, str, float]], int]:
    result: List[Tuple[str, str, float]] = []

    for record in records_batch:
        sql_element = (record_idx, record['move_seq'], record['target'], record['score'])
        result.append(sql_element)
        record_idx += 1

    return result, record_idx


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-records', type=int)
    parser.add_argument('--keyboard-type', type=str, choices=[t.name.lower() for t in KeyboardType], required=True)
    args = parser.parse_args()

    assert not args.dictionary_path.endswith('.db'), 'Must provide an in-memory (pickle) dictionary file for efficiency purposes.'

    # Make the keyboard
    keyboard_type = KeyboardType[args.keyboard_type.upper()]
    if keyboard_type == KeyboardType.SAMSUNG:
        tv_type = SmartTVType.SAMSUNG
    else:
        tv_type = SmartTVType.APPLE_TV

    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)

    # Delete any existing database file (if present)
    if os.path.exists(args.output_path):
        print('You are going to overwrite the file {}. Is this okay? [y/n]'.format(args.output_path), end=' ')
        decision = input().lower()

        if decision not in ('yes', 'y'):
            print('Quitting.')
            sys.exit(0)

        os.remove(args.output_path)

    # Create the dictionary
    print('Restoring dictionary...')
    dictionary = restore_dictionary(args.dictionary_path)

    # Open the database connection
    conn = sqlite3.connect(args.output_path)
    cursor = conn.cursor()

    # Make the table
    cursor.execute('CREATE TABLE passwords(id int IDENTITY PRIMARY KEY, seq varchar(500), password varchar(100), score double);')
    cursor.execute('CREATE INDEX seq_index ON passwords (seq);')

    # Load the passwords into memory
    passwords = load_passwords(args.input_path, max_num_records=args.max_num_records)
    record_idx = 0
    password_idx = 0

    for password_batch in chunked(passwords, BATCH_SIZE):
        # Create the list of move sequences for all the passwords in this batch
        create_fn = partial(create_records, dictionary=dictionary, keyboard=keyboard, keyboard_type=keyboard_type)
        record_lists = list(map(create_fn, password_batch))

        # Group and write the results in one transaction
        records_batch: List[Dict[str, Any]] = []
        for record_list in record_lists:
            records_batch.extend(record_list)

        sql_batch, record_idx = make_sql_batch(records_batch, record_idx=record_idx)
        cursor.executemany('INSERT INTO passwords VALUES(?, ?, ?, ?)', sql_batch)
        conn.commit()

        password_idx += len(password_batch)
        print('Completed {} records.'.format(password_idx), end='\r')
    print()
