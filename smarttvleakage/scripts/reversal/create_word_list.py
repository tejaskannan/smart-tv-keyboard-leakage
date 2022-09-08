from argparse import ArgumentParser
from typing import List

from smarttvleakage.utils.file_utils import save_txt_lines, read_txt_lines


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--has-counts', action='store_true')
    args = parser.parse_args()

    # Read in the input strings
    strings: List[str] = []
    for idx, line in enumerate(read_txt_lines(args.input_path)):
        if (args.max_num_samples is not None) and (idx >= args.max_num_samples):
            break

        if args.has_counts:
            tokens = line.split()
            line = ' '.join(tokens[0:-1])

        strings.append(line)

    # Write out the result
    save_txt_lines(strings, args.output_path)
