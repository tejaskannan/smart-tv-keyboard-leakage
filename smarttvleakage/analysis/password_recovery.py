import gzip
import io
import numpy as np
import re
from argparse import ArgumentParser
from collections import namedtuple
from typing import Set, Dict, Tuple

from smarttvleakage.utils.constants import BIG_NUMBER
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.utils.edit_distance import compute_edit_distance


PasswordMetrics = namedtuple('PasswordMetrics', ['top1', 'top10', 'top50', 'avg_rank', 'std_rank', 'total_count'])
LowercaseRegex = re.compile(r'[a-z]+')
LettersRegex = re.compile(r'[a-zA-Z]*[A-Z][a-zA-Z]*')
AlphaNumericRegex = re.compile(r'[a-zA-Z0-9]*[0-9][a-zA-Z0-9]*')
SpecialRegex = re.compile(r'.*[!"#$%&\'()*+,-.\/:;<=>?@[\\\]^_`{|}~].*')


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


def compute_password_metrics(ranks_dict: Dict[str, int]) -> PasswordMetrics:
    top1_count = sum(int(rank == 1) for rank in ranks_dict.values())
    top10_count = sum(int((rank >= 1) and (rank <= 10)) for rank in ranks_dict.values())
    top50_count = sum(int((rank >= 1) and (rank <= 50)) for rank in ranks_dict.values())

    filtered_ranks = [rank for rank in ranks_dict.values() if rank >= 1]
    avg_rank, std_rank = np.average(filtered_ranks), np.std(filtered_ranks)
    total_count = len(ranks_dict)

    return PasswordMetrics(top1=top1_count / total_count,
                           top10=top10_count / total_count,
                           top50=top50_count / total_count,
                           avg_rank=avg_rank,
                           std_rank=std_rank,
                           total_count=total_count)


def compute_metrics_by_prior(ranks_dict: Dict[str, int], prior_words: Set[str]) -> Tuple[PasswordMetrics, PasswordMetrics]:
    seen_dict = {word: rank for word, rank in ranks_dict.items() if word in prior_words}
    unseen_dict = {word: rank for word, rank in ranks_dict.items() if word not in prior_words}
    return compute_password_metrics(seen_dict), compute_password_metrics(unseen_dict)


def compute_metrics_by_feature(ranks_dict: Dict[str, int], regex: re.Pattern) -> PasswordMetrics:
    matching_dict = {word: rank for word, rank in ranks_dict.items() if regex.fullmatch(word)}
    return compute_password_metrics(matching_dict)


def print_metrics(name: str, metrics: PasswordMetrics):
    print('{} ({}): Top1: {:.4f}, Top10: {:.4f}, Top50: {:.4f}, Avg Rank: {:.4f} ({:.4f})'.format(name, metrics.total_count, metrics.top1, metrics.top10, metrics.top50, metrics.avg_rank, metrics.std_rank))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--recovery-path', type=str, required=True)
    parser.add_argument('--prior-path', type=str, required=True)
    args = parser.parse_args()

    prior_words = load_prior_words(args.prior_path)
    ranks_dict: Dict[str, int] = dict()

    for record in read_jsonl_gz(args.recovery_path):
        target = record['target']

        is_found = False
        rank = -1
        for idx, guess in enumerate(record['guesses']):
            if target == guess:
                rank = idx + 1
                is_found = True
                break

        ranks_dict[target] = rank

    total_metrics = compute_password_metrics(ranks_dict)
    seen_metrics, unseen_metrics = compute_metrics_by_prior(ranks_dict, prior_words=prior_words)
    lowercase_metrics = compute_metrics_by_feature(ranks_dict, regex=LowercaseRegex)
    letters_metrics = compute_metrics_by_feature(ranks_dict, regex=LettersRegex)
    alpha_metrics = compute_metrics_by_feature(ranks_dict, regex=AlphaNumericRegex)
    special_metrics = compute_metrics_by_feature(ranks_dict, regex=SpecialRegex)

    print_metrics(name='Overall', metrics=total_metrics)
    print_metrics(name='In Prior', metrics=seen_metrics)
    print_metrics(name='Not In Prior', metrics=unseen_metrics)
    print_metrics(name='Lowercase', metrics=lowercase_metrics)
    print_metrics(name='1+ Uppercase', metrics=letters_metrics)
    print_metrics(name='1+ Digit', metrics=alpha_metrics)
    print_metrics(name='1+ Special', metrics=special_metrics)
