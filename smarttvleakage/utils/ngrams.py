from typing import Iterable, Tuple

from .constants import START_CHAR, END_CHAR


def create_ngrams(string: str, n: int) -> Iterable[str]:
    """
    Computes the moving-window n-grams in the given word
    """
    for idx in range(1, min(n, len(string) + 2)):
        start_characters = [START_CHAR for _ in range(n - idx)]
        ngram = ''.join(start_characters) + string[0:idx]

        if idx == len(string) + 1:
            yield '{}{}'.format(ngram, END_CHAR)
            return
        else:
            yield ngram

    for idx in range(len(string) - n + 1):
        yield string[idx:idx+n]

    if n > 1:
        yield '{}{}'.format(string[-(n - 1):], END_CHAR)


def split_ngram(ngram: str) -> Tuple[str, str]:
    if ngram.endswith(END_CHAR):
        ngram_prefix = ngram[0:(len(ngram) - len(END_CHAR))]
        ngram_suffix = ngram[(len(ngram) - len(END_CHAR)):]
        return ngram_prefix, ngram_suffix
    else:
        return ngram[0:len(ngram) - 1], ngram[-1]


def prepend_start_characters(prefix: str, ngram_size: int) -> str:
    if len(prefix) >= ngram_size:
        return prefix

    num_to_add = ngram_size - len(prefix)
    start_characters = [START_CHAR for _ in range(num_to_add)]
    return '{}{}'.format(''.join(start_characters), prefix)
