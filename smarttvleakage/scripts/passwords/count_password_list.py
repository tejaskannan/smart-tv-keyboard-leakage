from argparse import ArgumentParser
from io import TextIOWrapper


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--password-list', type=str, required=True)
    args = parser.parse_args()

    total_count = 0

    with open(args.password_list, 'rb') as fin:
        io_wrapper = TextIOWrapper(fin, encoding='utf-8', errors='ignore')
        for line_idx, line in enumerate(io_wrapper):

            # Remove leading and trailing whitespace
            word = line.strip()
            if len(word) <= 0:
                continue

            total_count += 1

    print('Total Count: {}'.format(total_count))

