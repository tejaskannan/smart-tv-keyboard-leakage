import gzip
import io
from argparse import ArgumentParser
from typing import Set

from smarttvleakage.utils.constants import BIG_NUMBER
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.utils.edit_distance import compute_edit_distance


def load_prior_words(prior_path: str) -> Set[str]:
    words: Set[str] = set()

    if prior_path.endswith('gz'):
        with gzip.open(prior_path, 'rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    words.add(line)
    else:
        with open(prior_path, 'rb') as fin:
            io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

            for line in fin:
                line = line.strip()
                if len(line) > 0:
                    try:
                        words.add(line.decode())
                    except UnicodeDecodeError:
                        pass

    return words


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recovery-path', type=str, required=True)
    parser.add_argument('--prior-path', type=str, required=True)
    args = parser.parse_args()

    prior_words = load_prior_words(args.prior_path)

    seen_top1 = 0
    seen_top10 = 0
    seen_top50 = 0
    seen_found = 0
    seen_edit_dist = 0
    seen_total = 0

    unseen_top1 = 0
    unseen_top10 = 0
    unseen_top50 = 0
    unseen_found = 0
    unseen_edit_dist = 0
    unseen_total = 0

    for record in read_jsonl_gz(args.recovery_path):
        target = record['target']

        is_found = False
        rank = BIG_NUMBER
        for idx, guess in enumerate(record['guesses']):
            if target == guess:
                rank = idx + 1
                is_found = True
                break

        if len(record['guesses']) > 0:
            edit_dist = compute_edit_distance(str1=target, str2=record['guesses'][0])

        has_seen = (target in prior_words)

        if has_seen:
            seen_top1 += int(rank == 1)
            seen_top10 += int(rank <= 10)
            seen_top50 += int(rank <= 50)
            seen_found += int(is_found)
            seen_edit_dist += edit_dist
            seen_total += 1
        else:
            unseen_top1 += int(rank == 1)
            unseen_top10 += int(rank <= 10)
            unseen_top50 += int(rank <= 50)
            unseen_found += int(is_found)
            unseen_edit_dist += edit_dist
            unseen_total += 1

    seen_total = max(seen_total, 1e-7)
    seen_top1_acc = seen_top1 / seen_total
    seen_top10_acc = seen_top10 / seen_total
    seen_top50_acc = seen_top50 / seen_total
    seen_accuracy = seen_found / seen_total
    seen_avg_edit_dist = seen_edit_dist / seen_total

    unseen_total = max(unseen_total, 1e-7)
    unseen_top1_acc = unseen_top1 / unseen_total
    unseen_top10_acc = unseen_top10 / unseen_total
    unseen_top50_acc = unseen_top50 / unseen_total
    unseen_accuracy = unseen_found / unseen_total
    unseen_avg_edit_dist = unseen_edit_dist / unseen_total

    print('Accuracy on words in prior: {:.4f} ({} / {}), Top1: {:.4f}, Top10: {:.4f}, Top50: {:.4f}, Avg Edit Dist: {:.4f}'.format(seen_accuracy, seen_found, seen_total, seen_top1_acc, seen_top10_acc, seen_top50_acc, seen_avg_edit_dist))
    print('Accuracy on words not in prior: {:.4f} ({} / {}), Top1: {:.4f}, Top10: {:.4f}, Top50: {:.4f},  Avg Edit Dist: {:.4f}'.format(unseen_accuracy, unseen_found, unseen_total, unseen_top1_acc, unseen_top10_acc, unseen_top50_acc, unseen_avg_edit_dist))
    print('Total Avg Edit Dist: {:.4f}'.format((seen_edit_dist + unseen_edit_dist) / (seen_total + unseen_total)))
