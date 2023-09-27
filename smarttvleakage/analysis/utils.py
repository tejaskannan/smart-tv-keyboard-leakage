import re
from collections import namedtuple
from typing import List, Tuple, Dict


PLOT_STYLE = 'seaborn-ticks'
AXIS_SIZE = 14
TITLE_SIZE = 16
LABEL_SIZE = 12
LEGEND_SIZE = 12
MARKER_SIZE = 8
MARKER = 'o'
LINE_WIDTH = 3
FIGSIZE = (9, 3.75)
CREDIT_CARD_FIGSIZE = (8, 3)

COLORS_0 = ['#253494', '#2c7fb8', '#a1dab4']
COLORS_1 = ['#bd0026', '#fd8d3c']

BASELINE_COLOR = '#a1dab4'

TV_COLORS = {
    'samsung': ['#253494', '#2c7fb8'],
    'appletv': ['#e6550d', '#fdae6b']
}

TV_LABELS = {
    'samsung': 'Samsung',
    'appletv': 'AppleTV'
}

SPECIAL_REGEX = re.compile(r'[^A-Za-z0-9]+')
NUMBER_REGEX = re.compile(r'[0-9]+')
UPPER_REGEX = re.compile(r'[A-Z]+')

PASSWORD_PRIORS = ['phpbb', 'rockyou-5gram']
PHPBB_COUNT = 184388
PASSWORD_PRIOR_LABELS = {
    'phpbb': 'Audio + PhpBB',
    'rockyou-5gram': 'Audio + RockYou'
}


PasswordAccuracy = namedtuple('PasswordAccuracy', ['correct', 'total', 'stratified_correct', 'stratified_total'])


def compute_rank(guesses: List[str], label: str) -> int:
    rank = 1
    for guess in guesses:
        if guess == label:
            return rank
        rank += 1

    return -1


def top_k_accuracy(ranks: List[int], top: int) -> Tuple[int, int]:
    assert top >= 1, 'Must provide a positive `top` count'

    recovered_count = 0
    for rank in ranks:
        recovered_count += int((rank >= 1) and (rank <= top))

    return recovered_count, len(ranks)


def compute_accuracy(num_correct: List[int], total_counts: List[int]) -> List[float]:
    assert len(num_correct) == len(total_counts), 'Correct ({}) and total ({}) lists must be of same size.'.format(num_correct, total_counts)
    return [100.0 * (correct / total) for correct, total in zip(num_correct, total_counts)]


def compute_baseline_accuracy(baseline_size: int, top_counts: List[int]) -> List[float]:
    return [100.0 * (top / baseline_size) for top in top_counts]


def has_special(string: str) -> bool:
    return SPECIAL_REGEX.search(string) is not None


def has_number(string: str) -> bool:
    return NUMBER_REGEX.search(string) is not None


def has_uppercase(string: str) -> bool:
    return UPPER_REGEX.search(string) is not None


def print_as_table(counts: Dict[str, List[int]], total_counts: Dict[str, int], top_counts: List[int]):
    title = 'Series'
    name_width = max(max(map(lambda k: len(str(k)), counts.keys())), len(title))
    data_width = 25

    print('{} & {}'.format(title.ljust(name_width), ' & '.join(map(lambda t: 'Top-{} Acc.'.format(t).ljust(data_width), top_counts))))
    print('----------')

    for name, count_results in sorted(counts.items()):
        total = total_counts[name]
        accuracy = [100.0 * (c / total) for c in count_results]
        formatted_acc = list(map(lambda t: '{:.2f}% ({} / {})'.format(t[0], t[1], total).ljust(data_width), zip(accuracy, count_results)))
        print('{} & {}'.format(str(name).ljust(name_width), ' & '.join(formatted_acc)))
