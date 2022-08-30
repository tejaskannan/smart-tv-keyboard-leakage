import csv
from argparse import ArgumentParser
from collections import Counter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--min-length', type=int, required=True)
    args = parser.parse_args()

    word_counts: Counter = Counter()

    with open(args.input_file, 'r') as fin:
        reader = csv.reader(fin, quotechar='|', delimiter=',')

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            tokens = line[1].split()

            for token in tokens:
                if len(token) >= args.min_length:
                    word_counts[token] += 1

    with open(args.output_file, 'w') as fout:
        for string, count in reversed(sorted(word_counts.items(), key=lambda t: t[1])):
            fout.write('{} {}'.format(string, count))
            fout.write('\n')
