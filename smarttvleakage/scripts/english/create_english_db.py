import sqlite3
import os
import sys
from collections import Counter
from argparse import ArgumentParser
from typing import Any, Dict, List, Set

from smarttvleakage.utils.file_utils import save_json


BATCH_SIZE = 10000
START_CHAR = '<S>'
END_CHAR = '<E>'


if __name__ == '__main__':
    parser = ArgumentParser('Script to construct a local SQL database of English words.')
    parser.add_argument('--db-file', type=str, required=True, help='Path to the output database file.')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the list of strings to upload.')
    parser.add_argument('--min-count', type=int, required=True, help='The minimum word count to include.')
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

    # Validate the minimum word count
    assert args.min_count >= 1, 'Must provide a positive minimum count'

    # Create the database connection and table
    conn = sqlite3.connect(args.db_file)

    cursor = conn.cursor()
    cursor.execute('CREATE TABLE prefixes(id int IDENTITY PRIMARY KEY, prefix varchar(20), next varchar(5), count int);')
    cursor.execute('CREATE INDEX prefix_index ON prefixes(prefix);')
    cursor.execute('CREATE TABLE words(id int IDENTITY PRIMARY KEY, word varchar(250));')
    cursor.execute('CREATE INDEX words_index ON words(word);')
    conn.commit()

    # Iterate through the dictionary and load the ngram values into memory
    prefix_dictionary: Dict[str, Counter] = dict()
    word_set: Set[str] = set()
    total_count = 0
    word_count = 0
    weighted_word_count = 0

    with open(args.input_file, 'r') as fin:
        for line_idx, line in enumerate(fin):
            word, count = line.split()
            count = int(count)

            # Remove leading and trailing whitespace
            word = word.strip()
            if len(word) <= 0:
                continue

            if count < args.min_count:
                continue

            word_count += 1
            weighted_word_count += count
            word_set.add(word)

            for end_idx in range(len(word)):
                if end_idx == 0:
                    prefix = START_CHAR
                else:
                    prefix = word[0:end_idx]

                next_char = word[end_idx]

                if (prefix not in prefix_dictionary):
                    prefix_dictionary[prefix] = Counter()

                prefix_dictionary[prefix][next_char] += count
                total_count += count

            if (word not in prefix_dictionary):
                prefix_dictionary[word] = Counter()

            prefix_dictionary[word][END_CHAR] += count
            total_count += count

            if (line_idx + 1) % BATCH_SIZE == 0:
                print('Completed parsing {} strings.'.format(line_idx + 1), end='\r')

    print('\nLoaded {} prefixes and {} words.'.format(len(prefix_dictionary), word_count))

    # Upload the results to the database in batches
    data_batch: List[List[Any]] = []
    idx = 0

    for record_idx, (prefix, suffix_counter) in enumerate(prefix_dictionary.items()):
        # Create a record for each prefix / suffix pair
        for suffix, count in suffix_counter.items():
            record = [idx, prefix, suffix, count]
            data_batch.append(record)
            idx += 1

        # Write out the result
        if len(data_batch) >= BATCH_SIZE:
            cursor.executemany('INSERT INTO prefixes VALUES(?, ?, ?, ?);', data_batch)
            conn.commit()
            data_batch = []

        if (record_idx + 1) % BATCH_SIZE == 0:
            print('Completed {} / {} prefixes.'.format(record_idx + 1, len(prefix_dictionary)), end='\r')

    # Clean up any remaining elements
    if len(data_batch) > 0:
        cursor.executemany('INSERT INTO prefixes VALUES(?, ?, ?, ?);', data_batch)
        conn.commit()

    print('Completed {} / {} prefixes.'.format(record_idx + 1, len(prefix_dictionary)))

    data_batch = []
    for record_idx, word in enumerate(word_set):
        record = [record_idx, word]
        data_batch.append(record)

        if len(data_batch) >= BATCH_SIZE:
            cursor.executemany('INSERT INTO words VALUES (?, ?);', data_batch)
            conn.commit()
            data_batch = []

        if (record_idx + 1) % BATCH_SIZE == 0:
            print('Completed {} / {} words.'.format(record_idx + 1, len(word_set)), end='\r')

    if len(data_batch) > 0:
        cursor.executemany('INSERT INTO words VALUES (?, ?);', data_batch)
        conn.commit()
        data_batch = []

    print('Completed {} / {} words.'.format(record_idx + 1, len(word_set)))
    conn.close()

    # Save the metadata
    folder_name, file_name = os.path.split(args.db_file)
    metadata_file = os.path.join(folder_name, file_name.replace('.db', '_metadata.json'))
    metadata_dict = {
        'total_count': total_count,
        'word_count': word_count,
        'weighted_word_count': weighted_word_count
    }

    save_json(metadata_dict, metadata_file)
