from argparse import ArgumentParser
from smarttvleakage.dictionary import EnglishDictionary, NgramDictionary


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--words-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--min-count', type=int, required=True)
    parser.add_argument('--has-counts', action='store_true')
    parser.add_argument('--dict-type', type=str, choices=['english', 'ngram'], required=True)
    args = parser.parse_args()

    if args.dict_type == 'english':
        dictionary = EnglishDictionary(max_depth=16)
    else:
        dictionary = NgramDictionary()

    print('Building dictionary...')
    dictionary.build(args.words_path, min_count=args.min_count, has_counts=args.has_counts)

    print('Built Dictionary. Saving...')
    dictionary.save(args.output_path)
