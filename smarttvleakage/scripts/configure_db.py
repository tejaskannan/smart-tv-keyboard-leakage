import sqlite3
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--db-file', type=str, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_file)

    cursor = conn.cursor()
    cursor.execute('CREATE TABLE passwords(id int IDENTITY PRIMARY KEY, seq varchar(500), password varchar(100), score double);')
    cursor.execute('CREATE INDEX seq_index ON passwords (seq);')
