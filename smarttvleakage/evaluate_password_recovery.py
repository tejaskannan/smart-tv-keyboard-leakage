import numpy as np
from argparse import ArgumentParser
from typing import List

from smarttvleakage.audio import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import UniformDictionary, EnglishDictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.utils.constants import SmartTVType, KeyboardType, BIG_NUMBER


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--max-num-results', type=int, required=True)
    args = parser.parse_args()

    tv_type = SmartTVType.SAMSUNG
    keyboard_type = KeyboardType.SAMSUNG

    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    characters = graph.get_characters()

    print('Loading dictionary...')
    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary()
    else:
        dictionary = EnglishDictionary.restore(path=args.dictionary_path)

    dictionary.set_characters(characters)

    top1_found = 0
    top10_found = 0
    num_found = 0
    total_count = 0
    ranks: List[int] = []

    print('Starting recovery...')

    for record in read_jsonl_gz(args.benchmark_path):
        target = record['target']
        move_seq = [Move(num_moves=r['moves'], end_sound=r['end_sound']) for r in record['move_seq']]

        did_find = False
        rank = BIG_NUMBER

        for idx, (guess, _, _) in enumerate(get_words_from_moves(move_seq, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=args.max_num_results)):
            if guess == target:
                rank = idx + 1
                did_find = True
                break

        top1_found += int(rank == 1)
        top10_found += int(rank <= 10)
        num_found += int(did_find)
        total_count += 1

        if (total_count % 10) == 0:
            print('Completed {} records.'.format(total_count), end='\r')

        if did_find:
            ranks.append(rank)

    print()
    print('Top 1 accuracy: {:.4f}'.format(top1_found / total_count))
    print('Top 10 accuracy: {:.4f}'.format(top10_found / total_count))
    print('Top {} accuracy: {:.4f}'.format(args.max_num_results, num_found / total_count))
    print('Avg Rank: {:.4f}, Med Rank: {:.4f}'.format(np.average(ranks), np.median(ranks)))
