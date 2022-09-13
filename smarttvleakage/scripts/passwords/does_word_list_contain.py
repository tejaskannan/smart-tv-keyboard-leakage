import gzip
import io
from argparse import ArgumentParser
from typing import List, Set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--word-file', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()

    did_find = False
    with open(args.word_file, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if word == args.target:
                did_find = True
                break

    if did_find:
        print('Yes')
    else:
        print('No')
