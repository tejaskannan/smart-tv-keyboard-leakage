import numpy as np
from argparse import ArgumentParser
from typing import List, Optional

from smarttvleakage.search_numeric import get_digits_from_moves
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import CharacterDictionary, restore_dictionary
from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from smarttvleakage.utils.transformations import reverse_move_seq
from smarttvleakage.keyboard_utils.word_to_move import findPath


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
    parser.add_argument('--target', type=str, required=True, help='The target string.')
    parser.add_argument('--max-num-results', type=int, required=True, help='The maximum number of search results.')
    parser.add_argument('--precomputed-path', type=str, help='Optional path to precomputed sequences.')
    parser.add_argument('--keyboard-type', type=str, choices=[t.name.lower() for t in KeyboardType], help='The type of keyboard TV.')
    args = parser.parse_args()

    keyboard_type = KeyboardType[args.keyboard_type.upper()]
    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    characters = graph.get_characters()
    keyboard_mode = graph.get_start_keyboard_mode()

    print('Restoring dictionary...')
    dictionary = restore_dictionary(args.dictionary_path)
    dictionary.set_characters(characters)

    if keyboard_type == KeyboardType.SAMSUNG:
        tv_type = SmartTVType.SAMSUNG
    else:
        tv_type = SmartTVType.APPLE_TV

    print('Target String: {}'.format(args.target))

    charset = graph.get_keyboard_characters(keyboard_mode)
    char_idx = np.random.randint(low=0, high=len(charset))
    start_key = charset[char_idx]

    print(start_key)

    moves = findPath(args.target, use_shortcuts=True, use_wraparound=True, use_done=True, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key=start_key)
    reversed_seq = reverse_move_seq(moves)

    print(list(map(lambda m: m.num_moves, moves)))
    print(list(map(lambda m: m.num_moves, reversed_seq)))

    for idx, (guess, score, candidates_count) in enumerate(get_digits_from_moves(reversed_seq, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=args.max_num_results, start_key='<DONE>', includes_done=False, is_searching_reverse=True)):
        guess = ''.join(list(reversed(guess)))
        print('{}. {}'.format(idx + 1, guess))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
