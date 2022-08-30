import numpy as np
from argparse import ArgumentParser
from typing import List


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--min-count', type=int, required=True)
    parser.add_argument('--subset-size', type=int, required=True)
    args = parser.parse_args()

    words: List[str] = []
    with open(args.input_file, 'r') as fin:
        for line in fin:
            tokens = line.split()
            token, count = tokens[0], int(tokens[1])

            if count >= args.min_count:
                words.append(token)

    rand = np.random.RandomState(789235)
    sample_indices = np.arange(len(words))
    selected_indices = rand.choice(sample_indices, size=args.subset_size, replace=False)

    for idx in selected_indices:
        print(words[idx])

