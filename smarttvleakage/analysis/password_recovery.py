import gzip
import io
from argparse import ArgumentParser
from typing import Set

from smarttvleakage.utils.file_utils import read_jsonl_gz


def load_prior_words(prior_path: str) -> Set[str]:
    words: Set[str] = set()

    if prior_path.endswith('gz'):
        with gzip.open(prior_path, 'rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    words.add(line)
    else:
        with open(prior_path, 'r') as fin:
            io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    words.add(line)

    return words


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recovery-path', type=str, required=True)
    parser.add_argument('--prior-path', type=str, required=True)
    args = parser.parse_args()

    prior_words = load_prior_words(args.prior_path)

    seen_found = 0
    seen_total = 0

    unseen_found = 0
    unseen_total = 0

    for record in read_jsonl_gz(args.recovery_path):
        target = record['target']

        is_found = False
        for guess in record['guesses']:
            if target == guess:
                is_found = True
                break

        has_seen = (target in prior_words)

        if has_seen:
            seen_found += int(is_found)
            seen_total += 1
        else:
            unseen_found += int(is_found)
            unseen_total += 1

    print('Accuracy on words in prior: {:.4f} ({} / {})'.format(seen_found / seen_total, seen_found, seen_total))
    print('Accuracy on words not in prior: {:.4f} ({} / {})'.format(unseen_found / unseen_total, unseen_found, unseen_total))
