import sqlite3
import os
import sys
import re
from collections import Counter
from argparse import ArgumentParser
from io import TextIOWrapper
from typing import Any, Dict, List

from smarttvleakage.utils.ngrams import create_ngrams, split_ngram
from smarttvleakage.utils.file_utils import save_json


BATCH_SIZE = 10000
START_CHAR = '<S>'
END_CHAR = '<E>'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--db-file', type=str, required=True, help='Path to the output database file.')
    parser.add_argument('--dictionary-file', type=str, required=True, help='Path to the list of strings to upload.')
    parser.add_argument('--ngram-sizes', type=int, required=True, nargs='+', help='The ngram sizes to use.')
    args = parser.parse_args()

    assert args.db_file.endswith('.db'), 'Must provide a .db output file'

    if os.path.exists(args.db_file):
        print('This operation will overwrite the file {}. Is this okay? [Y/N]'.format(args.db_file), end=' ')

        user_input = input()
        user_input = user_input.lower()

        if user_input not in ('y', 'yes'):
            print('Quitting.')
            sys.exit(0)

        os.remove(args.db_file)

    # Sort and validate the ngram sizes
    assert all((size >= 2) for size in args.ngram_sizes), 'All ngram sizes must be >= 2'
    ngram_sizes = list(sorted(args.ngram_sizes))

    # Create the database connection and table
    conn = sqlite3.connect(args.db_file)

    cursor = conn.cursor()
    cursor.execute('CREATE TABLE ngrams(id int IDENTITY PRIMARY KEY, prefix varchar(20), next varchar(5), count int);')
    cursor.execute('CREATE INDEX ngram_index ON ngrams(prefix);')
    conn.commit()

    # Iterate through the dictionary and load the ngram values into memory
    ngram_dictionary: Dict[str, Counter] = dict()
    total_count = 0

    with open(args.dictionary_file, 'rb') as fin:
        io_wrapper = TextIOWrapper(fin, encoding='utf-8', errors='ignore')
        for line_idx, line in enumerate(io_wrapper):

            # Remove leading and trailing whitespace
            word = line.strip()
            if len(word) <= 0:
                continue

            # Determine if we can split the string into characters followed by
            # numbers. We can then match patterns based solely on numbers, which
            # can help with generalizing across datasets
            for ngram_idx, ngram_size in enumerate(ngram_sizes):
                count = pow(2, ngram_idx)

                for ngram in create_ngrams(word, ngram_size):
                    prefix, suffix = split_ngram(ngram)

                    # If we have a numeric suffix match, we add a generic number
                    # key to the prefix and remove start characters from the prefix
                    if prefix not in ngram_dictionary:
                        ngram_dictionary[prefix] = Counter()

                    ngram_dictionary[prefix][suffix] += count
                    total_count += count

            if (line_idx + 1) % BATCH_SIZE == 0:
                print('Completed parsing {} strings.'.format(line_idx + 1), end='\r')

    print('\nLoaded {} ngrams'.format(len(ngram_dictionary)))

    print(ngram_dictionary)

    # Upload the results to the database in batches
    data_batch: List[List[Any]] = []
    idx = 0

    for record_idx, (prefix, suffix_counter) in enumerate(ngram_dictionary.items()):
        # Create a record for each prefix / suffix pair
        for suffix, count in suffix_counter.items():
            record = [idx, prefix, suffix, count]
            data_batch.append(record)
            idx += 1

        # Write out the result
        if len(data_batch) >= BATCH_SIZE:
            cursor.executemany('INSERT INTO ngrams VALUES(?, ?, ?, ?);', data_batch)
            conn.commit()
            data_batch = []

        if (record_idx + 1) % BATCH_SIZE == 0:
            print('Completed {} / {} ngrams.'.format(record_idx + 1, len(ngram_dictionary)), end='\r')

    # Clean up any remaining elements 
    if len(data_batch) > 0:
        cursor.executemany('INSERT INTO ngrams VALUES(?, ?, ?, ?);', data_batch)
        conn.commit()

    print('Completed {} / {} ngrams.'.format(record_idx + 1, len(ngram_dictionary)))

    conn.close()

    # Save the metadata
    folder_name, file_name = os.path.split(args.db_file)
    metadata_file = os.path.join(folder_name, file_name.replace('.db', '_metadata.json'))
    metadata_dict = {
        'ngram_sizes': ngram_sizes,
        'total_count': total_count
    }

    save_json(metadata_dict, metadata_file)
