import io
import string
from typing import List


def read_passwords(path: str, count: int) -> List[str]:
    results: List[str] = []

    with open(path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            word = line.strip()

            if all((c in string.printable) for c in word):
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
            results.append(word)

            if len(results) >= count:
                break

    return results
