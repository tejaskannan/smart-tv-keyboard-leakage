from typing import Iterable


def create_ngrams(string: str, n: int) -> Iterable[str]:
    """
    Computes the moving-window n-grams in the given word
    """
    for idx in range(len(string) - n + 1):
        yield string[idx:idx+n]

