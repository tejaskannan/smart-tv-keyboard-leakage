import gzip
import io
from argparse import ArgumentParser
from typing import List, Set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--target-file', type=str, required=True)
    parser.add_argument('--baseline-file', type=str, required=True)
    parser.add_argument('--num-words', type=int, required=True)
    args = parser.parse_args()

    baseline_words: Set[str] = set()
    with gzip.open(args.baseline_file, 'rt', encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            baseline_words.add(word)

    unique_words: List[str] = []
    common_words: List[str] = []

    with open(args.target_file, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if (word not in baseline_words) and (len(unique_words) < args.num_words):
                unique_words.append(word)

            if (word in baseline_words) and (len(common_words) < args.num_words):
                common_words.append(word)

            if (len(unique_words) >= args.num_words) and (len(common_words) >= args.num_words):
                break

    print('Unique: {}'.format(unique_words))
    print('Common: {}'.format(common_words))
