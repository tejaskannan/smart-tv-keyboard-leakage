import sqlite3
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--db-file', type=str, required=True)
    parser.add_argument('--ngram', type=str, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()

    for row in cursor.execute('SELECT prefix, next, count FROM ngrams;'):
        print(row)

