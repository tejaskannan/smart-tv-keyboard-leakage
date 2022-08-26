import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Any

from smarttvleakage.audio import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import restore_dictionary
from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.utils.file_utils import read_jsonl_gz, save_jsonl_gz
from smarttvleakage.utils.edit_distance import compute_edit_distance
from smarttvleakage.utils.constants import SmartTVType, KeyboardType, BIG_NUMBER, Direction


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--precomputed-path', type=str)
    parser.add_argument('--max-num-results', type=int, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--keyboard-type', type=str, choices=['samsung', 'apple_tv_password'], required=True)
    args = parser.parse_args()

    keyboard_type = KeyboardType[args.keyboard_type.upper()]
    tv_type = SmartTVType.SAMSUNG if keyboard_type == KeyboardType.SAMSUNG else SmartTVType.APPLE_TV

    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    characters = graph.get_characters()

    print('Loading dictionary...')
    dictionary = restore_dictionary(args.dictionary_path)
    dictionary.set_characters(characters)

    print('Loading precomputed strings...')
    precomputed = PasswordRainbow(args.precomputed_path) if args.precomputed_path else None

    top1_found = 0
    top10_found = 0
    num_found = 0
    total_count = 0
    ranks: List[int] = []
    lengths: List[int] = []
    output_records: List[Dict[str, Any]] = []

    print('Starting recovery...')

    for record in read_jsonl_gz(args.benchmark_path):
        target = record['target']
        move_seq = [Move(num_moves=r['moves'], end_sound=r['end_sound'], directions=Direction.ANY) for r in record['move_seq']]

        did_find = False
        rank = BIG_NUMBER
        edit_dist = BIG_NUMBER
        guess_list: List[str] = []

        for idx, (guess, _, _) in enumerate(get_words_from_moves(move_seq, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=args.max_num_results, precomputed=precomputed, includes_done=True)):
            guess_list.append(guess)

            if guess == target:
                rank = idx + 1
                did_find = True
                break

        top1_found += int(rank == 1)
        top10_found += int(rank <= 10)
        num_found += int(did_find)
        total_count += 1
        lengths.append(len(target))

        output_record = {
            'target': target,
            'guesses': guess_list
        }
        output_records.append(output_record)

        if (total_count % 10) == 0:
            print('Completed {} records. Top 10 Accuracy So Far: {:.4f}'.format(total_count, top10_found / total_count), end='\r')

        if did_find:
            ranks.append(rank)

    print()
    save_jsonl_gz(output_records, args.output_path)

    print('Top 1 accuracy: {:.4f}'.format(top1_found / total_count))
    print('Top 10 accuracy: {:.4f}'.format(top10_found / total_count))
    print('Top {} accuracy: {:.4f}'.format(args.max_num_results, num_found / total_count))
    print('Avg Rank: {:.4f}, Med Rank: {:.4f}'.format(np.average(ranks), np.median(ranks)))
    print('Avg Length: {:.4f}, Med Length: {:.4f}'.format(np.average(lengths), np.median(lengths)))
