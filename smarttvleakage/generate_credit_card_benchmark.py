import numpy as np
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.credit_card_detection import validate_credit_card_number
from smarttvleakage.utils.file_utils import read_json, save_json_gz


def sample_random_zip_code(rand: np.random.RandomState, zip_codes: List[Tuple[str, int]]) -> str:
    zip_code_indices = np.arange(len(zip_codes))
    counts = list(map(lambda z: z[1], zip_codes))
    total_count = sum(counts)
    weights = [c / total_count for c in counts]

    selected_idx = int(rand.choice(zip_code_indices, size=1, p=weights))
    return zip_codes[selected_idx][0]


def read_zip_codes(path: str) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []

    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.split(' ')
            zip_code, count = tokens[0], int(tokens[1])
            results.append((zip_code, count))

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cc-files', type=str, required=True, nargs='+')
    parser.add_argument('--zip-code-file', type=str, required=True)
    parser.add_argument('--keyboard-type', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    # Read in the credit card details
    credit_card_details_list: List[Dict[str, Any]] = []
    for cc_file in args.cc_files:
        credit_card_details_list.extend(read_json(cc_file))

    keyboard_type = KeyboardType[args.keyboard_type.upper()]
    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)
    start_key = START_KEYS[keyboard.get_start_keyboard_mode()]

    # Read in the zip codes
    zip_codes = read_zip_codes(args.zip_code_file)
    rand = np.random.RandomState(32489)

    results: List[Dict[str, Any]] = []
    for credit_card_details in credit_card_details_list:
        # Unpack the information
        ccn = str(credit_card_details['Card Number'])
        exp_month = str(credit_card_details['Expire'].split('/')[0])
        exp_year = str(credit_card_details['Expire'].split('/')[1])
        cvv = str(credit_card_details['CVC'])
        zip_code = sample_random_zip_code(rand=rand, zip_codes=zip_codes)

        assert validate_credit_card_number(ccn), 'Found invalid CCN: {}'.format(ccn)

        # Generate the move sequences for each field
        ccn_seq = findPath(ccn, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
        month_seq = findPath(exp_month, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
        year_seq = findPath(exp_year, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
        cvv_seq = findPath(cvv, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)
        zip_seq = findPath(zip_code, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key=start_key)

        # Stitch together into one move sequence. The extractor has to split this up at runtime.
        move_seq = ccn_seq + month_seq + year_seq + cvv_seq + zip_seq

        # Compile the result
        result = {
            'credit_card_number': ccn,
            'exp_month': exp_month,
            'exp_year': exp_year,
            'cvv': cvv,
            'zip_code': zip_code,
            'move_seq': list(map(lambda m: m.to_dict(), move_seq)),
            'keyboard_type': keyboard_type.name.lower()
        }
        results.append(result)

    # Save the result
    save_json_gz(results, args.output_file)
