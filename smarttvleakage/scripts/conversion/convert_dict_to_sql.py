import sqlite3
from argparse import ArgumentParser
from collections import Counter
from typing import Any, Dict, List

from smarttvleakage.dictionary.dictionaries import restore_dictionary, NgramDictionary


BATCH_SIZE = 1000


def insert_records(ngram_counter: Dict[str, Counter], start_idx: int, length_bucket: int, conn: sqlite3.Connection) -> int:
    idx = start_idx
    batch: List[Any] = []

    cursor = conn.cursor()

    for prefix, counter in ngram_counter.items():
        for suffix, count in counter.items():
            record = (idx, prefix, suffix, length_bucket, count)
            batch.append(record)
            idx += 1

            if len(batch) % BATCH_SIZE == 0:
                cursor.executemany('INSERT INTO ngrams VALUES(?, ?, ?, ?, ?);', batch)
                conn.commit()
                batch = []

                print('Completed {} records'.format(idx), end='\r')

    if len(batch) > 0:
        cursor.executemany('INSERT INTO ngrams VALUES(?, ?, ?, ?, ?);', batch)
        conn.commit()

    print()
    return idx    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-file', type=str, required=True)
    parser.add_argument('--db-file', type=str, required=True)
    args = parser.parse_args()

    # Load the dictionary
    print('Loading dictionary into memory...')
    dictionary = restore_dictionary(args.dictionary_file)

    # Make the SQL table
    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE ngrams(id int IDENTITY PRIMARY KEY, prefix varchar(20), suffix varchar(5), length_bucket int, count int);')
    cursor.execute('CREATE INDEX prefix_index ON ngrams (prefix);')

    # Insert the records in each split
    start_idx = 0
    for length_bucket, ngram_counter in dictionary._counts_per_length.items():
        start_idx = insert_records(ngram_counter=ngram_counter,
                                   start_idx=start_idx,
                                   length_bucket=length_bucket,
                                   conn=conn)
