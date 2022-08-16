import numpy as np
from argparse import ArgumentParser
from typing import List

from smarttvleakage.dictionary.dictionaries import restore_dictionary
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.file_utils import read_json
from smarttvleakage.utils.constants import KeyboardType, BIG_NUMBER, SmartTVType
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.search_without_autocomplete import get_words_from_moves


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--credit-card-json', type=str, required=True)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Read in the credit card information
    credit_card_info = read_json(args.credit_card_json)

    # Make the graph and build the dictionary
    graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)

    dictionary = restore_dictionary('credit_card')
    dictionary.set_characters(graph.get_characters())

    not_found: List[str] = []
    ranks: List[str] = []
    num_found = 0
    top1_found = 0
    top10_found = 0
    total_count = 0

    for record_idx, credit_card_record in enumerate(credit_card_info):
        if (args.max_num_samples is not None) and (record_idx >= args.max_num_samples):
            break

        card_number = credit_card_record['cardNumber']
        move_seq = findPath(card_number, True, True, 0.0, 1.0, 0, graph)

        rank = BIG_NUMBER
        did_find = False

        for idx, (guess, _, _) in enumerate(get_words_from_moves(move_seq, graph=graph, dictionary=dictionary, max_num_results=50, precomputed=None, tv_type=SmartTVType.SAMSUNG)):
            if guess == card_number:
                rank = idx + 1
                did_find = True
                break

        if did_find:
            ranks.append(rank)
        else:
            not_found.append(card_number)

        top1_found += int(rank == 1)
        top10_found += int(rank <= 10)
        num_found += int(did_find)
        total_count += 1

        if (record_idx + 1) % 10 == 0:
            print('Completed {} records.'.format(record_idx + 1), end='\r')

    print()
    print('Top 1 Accuracy: {:.4f}'.format(top1_found / total_count))
    print('Top 10 Accuracy: {:.4f}'.format(top10_found / total_count))
    print('Top 50 Accuracy: {:.4f}'.format(num_found / total_count))
    print('Avg Rank: {:.4f}, Med Rank: {:.4f}'.format(np.average(ranks), np.median(ranks)))
    print('Not Found: {}'.format(not_found))
