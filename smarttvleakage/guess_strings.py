import matplotlib.pyplot as plt
import numpy as np
import os.path
from argparse import ArgumentParser
from typing import Tuple, List, Dict

from smarttvleakage.audio.data_types import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.dictionary import restore_dictionary, NumericDictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_suggestions, apply_autocomplete
from smarttvleakage.search_numeric import get_digits_from_moves
from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from smarttvleakage.utils.file_utils import read_json, iterate_dir


AUTOCOMPLETE_PREFIX_COUNT = 15


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--processed-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--max-num-results', type=int)
    parser.add_argument('--max-num-videos', type=int)
    parser.add_argument('--suggestions', type=str)
    args = parser.parse_args()

    if os.path.isdir(args.processed_path):
        processed_paths = list(iterate_dir(args.processed_path))
    else:
        processed_paths = [args.processed_path]

    # Load the dictionary
    print('Starting to load the dictionary...')
    dictionary = restore_dictionary(path=args.dictionary_path)
    print('Finished loading dictionary.')

    rank_list: List[int] = []
    num_candidates_list: List[int] = []
    rank_dict: Dict[str, int] = dict()
    candidates_dict: Dict[str, int] = dict()
    not_found_list: List[str] = []

    top10_correct = 0
    total_count = 0
    num_not_found = 0

    prefix_top10_correct = 0
    prefix_total_count = 0
    is_numeric = isinstance(dictionary, NumericDictionary)
    count = 0

    for file_path in processed_paths:
        if not file_path.endswith('.json'):
            continue

        serialized_list = read_json(file_path)
        for serialized in serialized_list:
            if (args.max_num_videos is not None) and (count >= args.max_num_videos):
                break

            count += 1

            # Extract the processed information
            true_word = serialized['target']
            move_sequence = [Move.from_dict(m) for m in serialized['move_seq']]
            tv_type = SmartTVType[serialized['smart_tv_type'].upper()]
            keyboard_type = KeyboardType[serialized['keyboard_type'].upper()]
            did_use_suggestions = serialized['did_use_suggestions']

            # Make the graph based on the keyboard type
            graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
            start_key = START_KEYS[graph.get_start_keyboard_mode()]

            # Set the dictionary characters
            dictionary.set_characters(graph.get_characters())

            if is_numeric:
                ranked_candidates = get_digits_from_moves(move_sequence=move_sequence,
                                                          graph=graph,
                                                          dictionary=dictionary,
                                                          tv_type=tv_type,
                                                          max_num_results=args.max_num_results,
                                                          is_searching_reverse=False,
                                                          start_key=start_key,
                                                          includes_done=True)
            elif did_use_suggestions:
                max_num_results = args.max_num_results if (not did_use_autocomplete) else AUTOCOMPLETE_PREFIX_COUNT
                ranked_candidates = get_words_from_moves_suggestions(move_sequence=move_sequence,
                                                                     graph=graph,
                                                                     dictionary=dictionary,
                                                                     did_use_autocomplete=did_use_autocomplete,
                                                                     max_num_results=max_num_results)

                if did_use_autocomplete:
                    prefixes: List[str] = []
                    true_prefix = true_word[0:len(move_sequence)]

                    rank = -1
                    for idx, (guess, score, num_candidates) in enumerate(ranked_candidates):
                        prefixes.append(guess)

                        if guess == true_prefix:
                            rank = idx + 1
                            break

                    prefix_top10_correct += int((rank >= 1) and (rank <= 10))
                    prefix_total_count += 1

                    ranked_candidates = apply_autocomplete(prefixes=prefixes,
                                                           dictionary=dictionary,
                                                           min_length=len(move_sequence) + 1,
                                                           max_num_results=args.max_num_results)
            else:
                ranked_candidates = get_words_from_moves(move_sequence=move_sequence,
                                                         graph=graph,
                                                         dictionary=dictionary,
                                                         tv_type=tv_type,
                                                         max_num_results=args.max_num_results,
                                                         precomputed=None,
                                                         is_searching_reverse=False,
                                                         start_key=start_key,
                                                         includes_done=True)

            did_find_word = False
            print('==========')

            for rank, (guess, score, num_candidates) in enumerate(ranked_candidates):
                print('Guess: {}, Score: {}'.format(guess, score))

                if guess == true_word:
                    rank_list.append(rank + 1)
                    rank_dict[true_word] = rank + 1
                    num_candidates_list.append(num_candidates)
                    candidates_dict[true_word] = num_candidates

                    did_find_word = True
                    break

            top10_correct += int(did_find_word and (rank <= 10))
            total_count += 1

            if (not did_find_word) and (args.max_num_results is not None):
                rank = args.max_num_results

            if (not did_find_word):
                not_found_list.append(true_word)

            move_sequence_vals = list(map(lambda m: m.num_moves, move_sequence))
            print('Word: {}'.format(true_word))
            print('Rank: {} (Did Find: {})'.format(rank + 1, did_find_word))
            print('Move Sequence: {}'.format(move_sequence_vals))
            print('Did Use Suggestions: {}'.format(did_use_suggestions))

            if not did_find_word:
                rank_list.append(rank + 1)
                rank_dict[true_word] = rank + 1
                candidates_dict[true_word] = num_candidates
                num_not_found += 1

    avg_rank = np.average(rank_list)
    med_rank = np.median(rank_list)

    avg_num_candidates = np.average(num_candidates_list)
    med_num_candidates = np.median(num_candidates_list)

    print('Ranking Dict: {}'.format(rank_dict))
    print('Candidates Dict: {}'.format(candidates_dict))
    print('Avg Rank: {:.4f}, Median Rank: {:.4f}'.format(avg_rank, med_rank))
    print('Avg # Candidates: {:.4f}, Median # Candidates: {:.4f}'.format(avg_num_candidates, med_num_candidates))
    print('Top 10 Accuracy: {:.4f}'.format(top10_correct / total_count))
    print('Num Not Found: {} ({:.4f})'.format(num_not_found, num_not_found / total_count))
    print('Words not found: {}'.format(not_found_list))

    if prefix_total_count > 0:
        print('Prefix Top 10 Accuracy: {:.4f} (Autocomplete Only)'.format(prefix_top10_correct / prefix_total_count))
