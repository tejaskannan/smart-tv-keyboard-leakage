import io
import string
import numpy as np
from typing import List


def read_passwords(path: str, count: int) -> List[str]:
    results: List[str] = []

    with open(path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all(should_keep(c) for c in word):
                results.append(word)

            if len(results) >= count:
                break

    return results


def read_english_words(path: str, count: int) -> List[str]:
    results: List[str] = []

    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            word = tokens[0]

            if all(should_keep(c) for c in word):
                results.append(word)

            if len(results) >= count:
                break

    return results


def read_english_words_random(path: str, count: int, rand: np.random.RandomState) -> List[str]:
    words: List[str] = []

    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.strip().split(' ')
            word = tokens[0]

            if all(should_keep(c) for c in word):
                words.append(word)

    indices = rand.choice(len(words), replace=False, size=count)
    return [words[idx] for idx in indices]


def should_keep(character: str) -> bool:
    return (character in string.printable) and (character != '_')
