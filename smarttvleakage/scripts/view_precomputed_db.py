import sqlite3
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--password', type=str, required=True)
    parser.add_argument('--db-file', type=str, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()

    result = conn.execute('SELECT COUNT(password) FROM passwords')
    print(result.fetchall())

    result = conn.execute('SELECT * from passwords WHERE password=:pwd', {'pwd': args.password})
    print(result.fetchall())




