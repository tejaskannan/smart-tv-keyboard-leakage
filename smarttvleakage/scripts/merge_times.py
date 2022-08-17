import os.path
from argparse import ArgumentParser
from typing import Any, Dict, List

from smarttvleakage.utils.file_utils import iterate_dir, save_jsonl_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    for path in iterate_dir(args.input_folder):
        file_name = os.path.basename(path)
        if (not file_name.startswith('times_')) or (not file_name.endswith('txt')):
            continue

        with open(path, 'r') as fin:
            contents = fin.read()

        lines = contents.split('\n')

        serialized = {
            'target': lines[0],
            'datetime': lines[1].split(' ')[-1].strip(),
            'perf_counter': lines[2].split(' ')[-1].strip(),
            'jtr_datetime': lines[3].split(' ')[-1].strip(),
            'jtr_perf_counter': lines[4].split(' ')[-1].strip()
        }
        results.append(serialized)

    save_jsonl_gz(results, args.output_file)
