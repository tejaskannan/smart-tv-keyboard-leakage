import csv
from argparse import ArgumentParser
from collections import Counter


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--num-words', type=int, required=True)
    args = parser.parse_args()

    word_counts: Counter = Counter()

    with open(args.input_file, 'r') as fin:
        reader = csv.reader(fin, quotechar='|', delimiter=',')

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            tokens = line[1].split()

            for token_idx in range(len(tokens) - args.num_words):
                string = ' '.join(tokens[token_idx:(token_idx + args.num_words)])
                word_counts[string] += 1

    with open(args.output_file, 'w') as fout:
        for string, count in word_counts.items():
            fout.write('{} {}'.format(string, count))
            fout.write('\n')
