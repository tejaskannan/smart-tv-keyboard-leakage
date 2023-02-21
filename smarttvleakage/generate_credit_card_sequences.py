import os.path
from argparse import ArgumentParser
from typing import Any, Dict, List

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.file_utils import read_json, save_json
from smarttvleakage.utils.constants import KeyboardType, SmartTVType



def generate_credit_card_sequences(labels: List[Dict[str, Any]], keyboard: MultiKeyboardGraph, output_path: str):
    results: List[Dict[str, List[Dict[str, Any]]]] = []

    fields = ['credit_card', 'security_code', 'zip_code', 'exp_month', 'exp_year']

    for label_dict in labels:
        result_dict: Dict[str, List[Dict[str, Any]]] = dict()

        for field in fields:
            move_seq = findPath(word=label_dict[field], use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=keyboard, start_key='q')
            result_dict[field] = [m.to_dict() for m in move_seq]

        results.append(result_dict)

    result = {
        'tv_type': SmartTVType.SAMSUNG.name.lower(),
        'seq_type': 'credit_card',
        'move_sequences': results
    }

    save_json(result, output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--labels-file', type=str, required=True)
    args = parser.parse_args()

    # Read the labels file
    labels_info = read_json(args.labels_file)
    labels = labels_info['labels']

    output_folder, file_name = os.path.split(args.labels_file)
    output_path = os.path.join(output_folder, file_name.replace('_labels.json', '.json'))
    print(output_path)

    keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)

    if labels_info['seq_type'] == 'credit_card':
        generate_credit_card_sequences(labels, keyboard, output_path)
    else:
        raise ValueError('Cannot generate sequences of type: {}'.format(labels_info['seq_type']))

