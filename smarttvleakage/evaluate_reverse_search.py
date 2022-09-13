import numpy as np
import os
from argparse import ArgumentParser

from smarttvleakage.dictionary.dictionaries import restore_dictionary
from smarttvleakage.utils.constants import KeyboardType, SmartTVType
from smarttvleakage.utils.file_utils import read_txt_lines
from smarttvleakage.utils.transformations import reverse_move_seq
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.search_numeric import get_digits_from_moves


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--max-num-results', type=int, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    args = parser.parse_args()

    targets = read_txt_lines(args.input_path)

    keyboard_type = KeyboardType.SAMSUNG
    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    characters = graph.get_characters()
    start_charset = graph.get_start_character_set()
    default_start_key = START_KEYS[graph.get_start_keyboard_mode()]

    # Create the forward and reverse dictionary
    dictionary = restore_dictionary(args.dictionary_path)
    dictionary.set_characters(characters)

    if args.dictionary_path in ('numeric', 'uniform'):
        reverse_dictionary = dictionary
    else:
        rev_dict_folder, rev_dict_file = os.path.split(args.dictionary_path)
        rev_dict_path = os.path.join(rev_dict_folder, '{}_{}'.format('rev', rev_dict_file))

        reverse_dictionary = restore_dictionary(rev_dict_path)
        reverse_dictionary.set_characters(characters)

    tv_type = SmartTVType.SAMSUNG
    rand = np.random.RandomState(seed=102)

    fixed_num_found = 0
    std_num_found = 0
    rev_num_found = 0

    fixed_rank_sum = 0
    std_rank_sum = 0
    rev_rank_sum = 0
    total_count = 0

    for target in targets:
        # Search in the forward direction with the normal start key
        moves = findPath(target, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key=default_start_key)
        rank = None

        for idx, (guess, score, candidates_count) in enumerate(get_digits_from_moves(moves, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=args.max_num_results, includes_done=True, start_key=default_start_key, is_searching_reverse=False)):
            if target == guess:
                rank = idx + 1
                break

        if rank is not None:
            fixed_rank_sum += rank
            fixed_num_found += 1

        start_idx = rand.randint(low=0, high=len(start_charset))
        start_key = start_charset[start_idx]

        # Search in the forward direction (what we would do normally)
        moves = findPath(target, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key=start_key)
        rank = None

        for idx, (guess, score, candidates_count) in enumerate(get_digits_from_moves(moves, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=args.max_num_results, includes_done=True, start_key=default_start_key, is_searching_reverse=False)):
            if target == guess:
                rank = idx + 1
                break

        if rank is not None:
            std_rank_sum += rank
            std_num_found += 1

        # Search in the reverse direction
        reversed_seq = reverse_move_seq(moves)
        rank = None

        for idx, (guess, score, candidates_count) in enumerate(get_digits_from_moves(reversed_seq, graph=graph, dictionary=reverse_dictionary, tv_type=tv_type, max_num_results=args.max_num_results, start_key='<DONE>', includes_done=False, is_searching_reverse=True)):
            guess = ''.join(list(reversed(guess)))

            if target == guess:
                rank = idx + 1
                break

        if rank is not None:
            rev_rank_sum += rank
            rev_num_found += 1

        total_count += 1

        if total_count % 100 == 0:
            print('Completed {} Samples'.format(total_count), end='\r')

    print('\nRev Top {} Accuracy: {:.5f}, Avg Rank (when found): {:.5f}'.format(args.max_num_results, rev_num_found / total_count, rev_rank_sum / total_count))
    print('Std Top {} Accuracy: {:.5f}, Avg Rank (when found): {:.5f}'.format(args.max_num_results, std_num_found / total_count, std_rank_sum / total_count))
    print('Fixed Top {} Accuracy: {:.5f}, Avg Rank (when found): {:.5f}'.format(args.max_num_results, fixed_num_found / total_count, fixed_rank_sum / total_count))

