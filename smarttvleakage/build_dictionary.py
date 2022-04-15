from argparse import ArgumentParser
from smarttvleakage.dictionary import EnglishDictionary


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--words-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    dictionary = EnglishDictionary()

    print('Building dictionary...')
    dictionary.build(args.words_path)

    dictionary.save(args.output_path)
