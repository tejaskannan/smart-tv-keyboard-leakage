import sqlite3
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from typing import Any


def fetch_record(password: str, db_file: str) -> Any:
    start = time.time()
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    end = time.time()

    print('Time to make connection: {}'.format(end - start))

    start = time.time()
    execution = cursor.execute('SELECT * from passwords WHERE password=:pwd', {'pwd': password})
    #execution = cursor.execute('SELECT password, score FROM passwords WHERE seq=:seq', { 'seq': move_seq })
    result = execution.fetchall()
    end = time.time()

    print('Time to fetch results: {}'.format(end - start))
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--passwords', type=str, required=True, nargs='+')
    parser.add_argument('--db-file', type=str, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()

    result = conn.execute('SELECT COUNT(password) FROM passwords')
    print(result.fetchall())

    process_fn = partial(fetch_record, db_file=args.db_file)

    start = time.time()

    with Pool(4) as pool:
        results = pool.map(process_fn, args.passwords)

    end = time.time()
    print('Elapsed: {}s'.format(end - start))
    print(results)
