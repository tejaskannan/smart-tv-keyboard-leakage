import numpy as np
import os
import sys
from argparse import ArgumentParser
from functools import partial
from more_itertools import chunked
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Optional

from smarttvleakage.audio import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.dictionary import restore_dictionary, CharacterDictionary
from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.utils.file_utils import read_jsonl_gz, save_jsonl_gz, append_jsonl_gz
from smarttvleakage.utils.edit_distance import compute_edit_distance
from smarttvleakage.utils.constants import SmartTVType, KeyboardType, BIG_NUMBER, Direction
from smarttvleakage.utils.move_noise import add_move_noise


CHUNK_SIZE = 100
MOVE_NOISE_SCALE = 1


def process_record(record: Dict[str, Any], graph: MultiKeyboardGraph, dictionary_path: str, tv_type: SmartTVType, max_num_results: int, precomputed_path: Optional[str], move_noise_rate: float) -> Tuple[int, List[str]]:
    # Make the dictionary and precomputed db
    dictionary = restore_dictionary(dictionary_path)
    dictionary.set_characters(graph.get_characters())

    precomputed = PasswordRainbow(precomputed_path) if (precomputed_path is not None) else None

    # Unpack the record
    target = record['target']
    move_seq = [Move(num_moves=r['moves'], end_sound=r['end_sound'], directions=Direction.ANY) for r in record['move_seq']]

    # Apply the move noise (if specified)
    print('Before: {}'.format(list(map(lambda m: m.num_moves, move_seq))))
    move_seq = add_move_noise(move_seq, scale=MOVE_NOISE_SCALE, rate=move_noise_rate)
    print('After: {}'.format(list(map(lambda m: m.num_moves, move_seq))))

    # Run the recovery process
    rank = -1
    guess_list: List[str] = []
    start_key = START_KEYS[graph.get_start_keyboard_mode()]

    result_iter = get_words_from_moves(move_seq,
                                       graph=graph,
                                       dictionary=dictionary,
                                       tv_type=tv_type,
                                       max_num_results=max_num_results,
                                       precomputed=precomputed,
                                       start_key=start_key,
                                       includes_done=True,
                                       is_searching_reverse=False)

    for idx, (guess, _, _) in enumerate(result_iter):
        guess_list.append(guess)

        if guess == target:
            rank = idx + 1
            break

    return rank, guess_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--precomputed-path', type=str)
    parser.add_argument('--max-num-results', type=int, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--keyboard-type', type=str, choices=['samsung', 'apple_tv_password'], required=True)
    parser.add_argument('--move-noise-rate', type=float, default=0.0)
    args = parser.parse_args()

    assert args.dictionary_path.endswith('db'), 'Must provide a db dictionary file. This is needed for better parallelism.'

    if os.path.exists(args.output_path):
        print('You are going to overwrite the file {}. Is this okay [y/n]?'.format(args.output_path), end=' ')
        decision = input().lower()

        if decision not in ('yes', 'y'):
            print('Quitting.')
            sys.exit(0)

        os.remove(args.output_path)

    keyboard_type = KeyboardType[args.keyboard_type.upper()]
    tv_type = SmartTVType.SAMSUNG if keyboard_type == KeyboardType.SAMSUNG else SmartTVType.APPLE_TV

    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    characters = graph.get_characters()

    top1_found = 0
    top10_found = 0
    num_found = 0
    total_count = 0
    ranks: List[int] = []
    lengths: List[int] = []

    print('Starting recovery...')

    for record_chunk in chunked(read_jsonl_gz(args.benchmark_path), CHUNK_SIZE):
        output_records: List[Dict[str, Any]] = []

        process_fn = partial(process_record,
                             graph=graph,
                             dictionary_path=args.dictionary_path,
                             precomputed_path=args.precomputed_path,
                             tv_type=tv_type, max_num_results=args.max_num_results,
                             move_noise_rate=args.move_noise_rate)
        with Pool(4) as pool:
            results = pool.map(process_fn, record_chunk)

        for record, (rank, guess_list) in zip(record_chunk, results):
            target = record['target']

            top1_found += int(rank == 1)
            top10_found += int((rank <= 10) and (rank >= 1))
            num_found += int(rank >= 1)
            lengths.append(len(target))
            ranks.append(rank)

            output_record = {
                'target': target,
                'guesses': guess_list
            }
            output_records.append(output_record)

        # Append the results to the output file
        append_jsonl_gz(output_records, args.output_path)

        total_count += len(record_chunk)
        print('Completed {} records. Top 10 Accuracy So Far: {:.4f} ({} / {})'.format(total_count, top10_found / total_count, top10_found, total_count), end='\r')

    print()

    print('Top 1 accuracy: {:.4f}'.format(top1_found / total_count))
    print('Top 10 accuracy: {:.4f}'.format(top10_found / total_count))
    print('Top {} accuracy: {:.4f}'.format(args.max_num_results, num_found / total_count))
    print('Avg Rank: {:.4f}, Med Rank: {:.4f}'.format(np.average(ranks), np.median(ranks)))
    print('Avg Length: {:.4f}, Med Length: {:.4f}'.format(np.average(lengths), np.median(lengths)))
