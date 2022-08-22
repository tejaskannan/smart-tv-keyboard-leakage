import io

from argparse import ArgumentParser
from typing import Set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    passwords: Set[str] = set()

    with open(args.input_file, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            password = line.strip()
            passwords.add(password)
            passwords.add(password.capitalize())
            passwords.add(password.upper())
            passwords.add(password.lower())

    print('Number of passwords: {}'.format(len(passwords)))

    with open(args.output_file, 'w') as fout:
        for password in passwords:
            fout.write(password)
            fout.write('\n')

