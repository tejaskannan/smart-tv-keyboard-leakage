import csv
import gzip
import re
import string
from argparse import ArgumentParser
from typing import List, Set, Dict

STOPWORDS = frozenset(['i', 'a', 'about', 'an', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what', 'when', 'where', 'who', 'will', 'with'])


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    return ' '.join(list(filter(lambda word: word not in STOPWORDS, tokens)))


def remove_non_alphanumeric(text: str) -> str:
    characters: List[str] = []
    for char in text:
        if (char == ' ') or (char in string.ascii_letters) or (char in string.digits):
            characters.append(char)

    return ''.join(characters)


def read_ratings(path: str) -> Dict[str, int]:
    ratings: Dict[str, int] = dict()

    with gzip.open(path, 'rb') as fin:
        for idx, line in enumerate(fin):
            if idx > 0:
                line = line.decode()
                tokens = list(map(lambda s: s.strip(), line.split('\t')))
                title_id = tokens[0]
                num_votes = int(tokens[2])

                ratings[title_id] = num_votes

    return ratings


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--ratings-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    print('Reading ratings...')
    ratings = read_ratings(args.ratings_path)

    print('Reading title names...')

    # Read in the list of titles
    english_titles: Dict[str, List[str]] = dict()

    with gzip.open(args.dataset_path, 'rb') as fin:
        for idx, line in enumerate(fin):
            if idx > 0:
                line = line.decode()
                tokens = list(map(lambda s: s.strip(), line.split('\t')))
                title_id = tokens[0]
                title = tokens[2].lower()
                region = tokens[3]
                language = tokens[4]

                if (title_id not in english_titles) and ((region in ('US', 'CA')) or (language == 'en')):

                    english_titles[title_id] = [title]

                    no_punctuation = remove_non_alphanumeric(text=title)
                    if no_punctuation != title:
                        english_titles[title_id].append(no_punctuation)

                    no_stopwords = remove_stopwords(text=no_punctuation)
                    if (no_stopwords != title) and (no_stopwords != no_punctuation):
                        english_titles[title_id].append(no_stopwords)

    print('Writing {} results...'.format(len(english_titles)))

    # Write the result to an output file
    with open(args.output_path, 'wb') as fout:
        for title_id, title_names in english_titles.items():
            rating = ratings.get(title_id, 1)

            if title_id == 'tt10986410':
                print('Title Names: {}, Rating: {}'.format(title_names, rating))

            for title_name in title_names:
                fout.write('{} {}'.format(title_name, rating).encode('utf-8'))
                fout.write('\n'.encode('utf-8'))
