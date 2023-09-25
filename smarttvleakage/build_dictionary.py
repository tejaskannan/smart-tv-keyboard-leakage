"""
This program serializes a dictionary into a string prior for later use during string recovery.
This serialization is done for efficiency--building the dictionary from scratch each time is slow.
"""
from argparse import ArgumentParser
from smarttvleakage.dictionary import EnglishDictionary, NgramDictionary, ZipCodeDictionary


if __name__ == '__main__':
    parser = ArgumentParser('Script to build dictionary priors.')
    parser.add_argument('--words-path', type=str, required=True, help='Path to the file of words to index.')
    parser.add_argument('--output-path', type=str, required=True, help='Path of the output file.')
    parser.add_argument('--min-count', type=int, required=True, help='Minimum count of inputs to index. All inputs below this count will be ignored.')
    parser.add_argument('--has-counts', action='store_true', help='Whether the input file has counts. If not, all strings have equal weight.')
    parser.add_argument('--dict-type', type=str, choices=['english', 'ngram', 'zip_code'], required=True, help='The type of dictionary to create')
    parser.add_argument('--should-reverse', action='store_true', help='Whether to store strings in reverse.')
    args = parser.parse_args()

    if args.dict_type == 'english':
        dictionary = EnglishDictionary(max_depth=16)
    elif args.dict_type == 'ngram':
        dictionary = NgramDictionary()
    elif args.dict_type == 'zip_code':
        dictionary = ZipCodeDictionary()
    else:
        raise ValueError('Unknown dictionary type {}'.format(args.dict_type))

    print('Building dictionary...')
    dictionary.build(args.words_path, min_count=args.min_count, has_counts=args.has_counts, should_reverse=args.should_reverse)

    print('Built Dictionary. Saving...')
    dictionary.save(args.output_path)
