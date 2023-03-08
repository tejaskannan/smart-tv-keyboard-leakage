from argparse import ArgumentParser
from typing import Any, Dict, List

from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_json, save_json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--extracted-paths', type=str, required=True, nargs='+', help='Paths to the extracted JSON files in order.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the merged output file.')
    args = parser.parse_args()

    result_move_sequences: List[List[Dict[str, Any]]] = []
    suggestions_types: List[str] = []

    for extraction_path in args.extracted_paths:
        move_extraction = read_json(extraction_path)

        result_move_sequences.extend(move_extraction['move_sequences'])
        suggestions_types.extend(move_extraction['suggestions_types'])

    result = {
        'tv_type': SmartTVType.SAMSUNG.name.lower(),
        'seq_type': 'standard',
        'move_sequences': result_move_sequences,
        'suggestions_types': suggestions_types
    }
    save_json(result, args.output_path)
