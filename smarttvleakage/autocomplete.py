import string
from collections import Counter
from typing import List, Tuple

from smarttvleakage.utils.file_utils import read_json


VOWELS = ['a', 'e', 'i', 'o', 'u']


def load_words(path: str, min_count: int) -> Tuple[List[str], List[int]]:
    result_words: List[str] = []
    result_counts: List[int] = []


    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.split()
            word = tokens[0]
            count = int(tokens[1])

            if count > min_count:
                result_words.append(word)
                result_counts.append(count)

    return result_words, result_counts


def predict_next_letter(prefix: str, words: List[str], counts: List[int], cutoff: float) -> List[str]:
    counter: Counter = Counter()
    top_words: List[str] = []

    for word, count in zip(words, counts):
        if word.startswith(prefix) and (word != prefix):
            next_letter = word[len(prefix)]

            if next_letter in string.ascii_lowercase:
                counter[next_letter] += count
                top_words.append(word)

    result: Dict[str, float] = dict()
    total_count = sum(counter.values())

    for idx, (letter, count) in enumerate(counter.most_common(6)):
        freq = count / total_count
        result[letter] = freq

    for letter, count in counter.items():
        freq = count / total_count

        if (letter in VOWELS) and (freq >= 0.001):
            result[letter] = freq

    num_above_cutoff = len([letter for letter, freq in result.items() if freq >= cutoff])

    if num_above_cutoff < 4:
        result['<BACK>'] = 1.0
        result['<SPACE>'] = 1.0

    # Limit to top 6 in total
    cutoff_counter: Counter = Counter()
    for letter, freq in result.items():
        cutoff_counter[letter] = freq

    return { letter: freq for (letter, freq) in cutoff_counter.most_common(8) }


if __name__ == '__main__':
    path = '/local/dictionaries/enwiki-20210820-words-frequency.txt'
    word_list, counts_list = load_words(path=path, min_count=50)

    observed = read_json('graphs/autocomplete.json')

    is_correct = 0
    total_count = 0

    recs = predict_next_letter(prefix='pe', words=word_list, counts=counts_list, cutoff=0.05)

    for letter in string.ascii_lowercase:
        recs = predict_next_letter(prefix=letter, words=word_list, counts=counts_list, cutoff=0.05)

        for obs in observed[letter]:
            if obs == '<NONE>':
                continue

            if obs in recs:
                is_correct += 1

            total_count += 1

        #print('{} -> {}'.format(letter, recs))
        #print('\t{}'.format(observed[letter]))

    print('Accuracy: {:.4f}'.format(is_correct / total_count))
