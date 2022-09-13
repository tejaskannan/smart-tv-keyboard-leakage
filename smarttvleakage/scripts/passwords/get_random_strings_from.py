import io
import numpy as np
from argparse import ArgumentParser
from typing import List



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()

    strings: List[str] = []

    with open(args.path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for idx, line in enumerate(io_wrapper):
            line = line.strip()
            if len(line) >= 8:
                strings.append(line)

    # Get random strings
    num_strings = len(strings)
    rand_idx = np.random.randint(low=0, high=num_strings, size=args.count)

    print('Num Strings: {}'.format(num_strings))

    for idx in rand_idx:
        print(strings[idx])
