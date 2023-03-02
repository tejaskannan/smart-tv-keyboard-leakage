import sqlite3
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--db-file', type=str, required=True)
    args = parser.parse_args()

    # Create the database connection
    conn = sqlite3.connect(args.db_file)

    cursor = conn.cursor()
    result = cursor.execute("SELECT SUM(count) FROM ngrams;")

    print(result.fetchall())
