import io
import string
import numpy as np
from argparse import ArgumentParser
from typing import List



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--min-length', type=int, required=True)
    parser.add_argument('--must-contain-special', action='store_true')
    args = parser.parse_args()

    strings: List[str] = []

    with open(args.path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for idx, line in enumerate(io_wrapper):
            line = line.strip()
            if len(line) >= args.min_length and ((not args.must_contain_special) or any((c in string.punctuation) for c in line)):
                strings.append(line)

    # Get random strings
    num_strings = len(strings)
    rand_idx = np.random.randint(low=0, high=num_strings, size=args.count)

    print('Num Strings: {}'.format(num_strings))

    for idx in rand_idx:
        print(strings[idx])
