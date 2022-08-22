import csv
from argparse import ArgumentParser
from typing import List, Tuple


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    zip_codes: List[Tuple[str, int]] = []
    with open(args.input_path, 'r') as fin:
        reader = csv.reader(fin, quotechar='|', delimiter=',')

        for idx, row in enumerate(reader):
            if idx > 0:
                zip_code = row[0]
                population = row[3]

                try:
                    zip_as_int = int(zip_code)
                    population = max(int(population), 1)
                    is_valid = (len(zip_code) == 5)
                except ValueError as ex:
                    is_valid = False

                if is_valid:
                    zip_codes.append((zip_code, population))

    with open(args.output_path, 'w') as fout:
        for (zip_code, population) in zip_codes:
            fout.write('{} {}'.format(zip_code, population))
            fout.write('\n')
