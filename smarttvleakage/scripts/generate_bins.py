from argparse import ArgumentParser
from typing import Iterable


def iterate_prefixes(string: str) -> Iterable[str]:
    for idx in range(1, len(string) + 1):
        yield string[0:idx]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    prefixes = set(['3', '34', '37', '4', '6', '60', '601', '6011', '65'])
    bin_ranges = [(51, 55), (2221, 2720), (622126, 622925), (624000, 626999), (628200, 628899), (644, 649)]

    for bin_range in bin_ranges:
        for bin_value in range(bin_range[0], bin_range[1] + 1):
            for prefix in iterate_prefixes(str(bin_value)):
                prefixes.add(prefix)

    with open(args.output_file, 'w') as fout:
        for prefix in prefixes:
            fout.write(prefix)
            fout.write('\n')
