import time
import string
import os.path
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.utils.file_utils import read_json


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode', 'was_on_autocomplete', 'current_key', 'center_key'])
VisitedState = namedtuple('VisitedState', ['string', 'was_on_autocomplete'])


def filter_and_normalize_scores(key_counts: Dict[str, int], candidate_keys: List[str]) -> Dict[str, float]:
    filtered_scores = { key: float(key_counts[key]) for key in candidate_keys if key in key_counts }
    score_sum = sum(key_counts.values())
    return { key: (score / score_sum) for key, score in filtered_scores.items() }


def get_keyboard_mode(key: str, mode: KeyboardMode) -> KeyboardMode:
    if key != '<CHANGE>':
        return mode

    if mode == KeyboardMode.STANDARD:
        return KeyboardMode.SPECIAL_ONE
    elif mode == KeyboardMode.SPECIAL_ONE:
        return KeyboardMode.STANDARD
    else:
        raise ValueError('Unknown mode {}'.format(mode))


def get_characters_from_keys(keys: List[str]) -> str:
    characters: List[str] = []

    caps_lock = False
    prev_turn_off_caps_lock = False

    for idx, key in enumerate(keys):
        if key == '<CAPS>':
            if caps_lock:
                caps_lock = False
                prev_turn_off_caps_lock = True
            elif (idx > 0) and (keys[idx - 1] == '<CAPS>'):
                caps_lock = True
                prev_turn_off_caps_lock = False
        elif key == '<BACK>':
            if len(characters) > 0:
                characters.pop()
        elif key not in UNPRINTED_CHARACTERS:
            if caps_lock or ((idx > 0) and (keys[idx - 1] == '<CAPS>') and (not prev_turn_off_caps_lock)):
                character = key.upper()
            else:
                character = CHARACTER_TRANSLATION.get(key, key)

            characters.append(character)
            prev_turn_off_caps_lock = False

    return characters


def get_words_from_moves_autocomplete(num_moves: List[int], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    target_length = len(num_moves)

    directory = os.path.dirname(__file__)
    single_autocomplete = read_json(os.path.join(directory, 'graphs/autocomplete.json'))

    candidate_queue = PriorityQueue()

    init_state = SearchState(keys=[],
                             score=1.0,
                             keyboard_mode=KeyboardMode.STANDARD,
                             was_on_autocomplete=False,
                             current_key=None,
                             center_key=None)
    candidate_queue.put((-1 * init_state.score, init_state))

    scores: Dict[str, float] = dict()
    visited: Set[VisitedState] = set()
    seen_strings: Set[str] = set()

    result_count = 0
    candidate_count = 0

    while not candidate_queue.empty():
        _, current_state = candidate_queue.get()

        current_characters = get_characters_from_keys(keys=current_state.keys)
        current_string = ''.join(current_characters)
        candidate_count += 1

        #print('Current String: {}, Score: {}'.format(current_string, current_state.score))

        #print('Current String: {}, Center Key: {}'.format(current_string, current_state.center_key))

        if (len(current_state.keys) == target_length):
            if current_string not in seen_strings:
                yield current_string, current_state.score, candidate_count
                result_count += 1

                if (max_num_results is not None) and (result_count >= max_num_results):
                    return

            seen_strings.add(current_string)
            continue

        move_idx = len(current_state.keys)
        prev_key = current_state.current_key if current_state.current_key is not None else START_KEYS[current_state.keyboard_mode]

        if (prev_key in string.ascii_letters) and (len(current_state.keys) > 0):
            if current_state.center_key != current_state.current_key:
                prev_key = current_state.center_key
                move_count = num_moves[move_idx]
            else:
                move_count = max(num_moves[move_idx] - 1, 0)
        else:
            move_count = num_moves[move_idx]

        #adjusted_moves = max(num_moves[move_idx] - 1, 0) if (prev_key in string.ascii_letters) and (len(current_state.keys) > 0) and (current_state.center_key == current_state.current_key) else num_moves[move_idx]
        next_key_counts = dictionary.get_letter_counts(prefix=current_string,
                                                       length=target_length,
                                                       should_smooth=True)

        if (num_moves[move_idx] != 1) or (len(current_string) == 0):
            neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                      num_moves=move_count,
                                                      mode=current_state.keyboard_mode)

            filtered_probs = filter_and_normalize_scores(key_counts=next_key_counts,
                                                         candidate_keys=neighbors)

            for neighbor_key, score in filtered_probs.items():
                candidate_keys = current_state.keys + [neighbor_key]
                candidate_word = ''.join(candidate_keys)

                # TODO: Handle upper case / non-character keys in scoring

                visited_state = VisitedState(string=candidate_word, was_on_autocomplete=False)
                should_aggregate = len(candidate_keys) < target_length

                if visited_state not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                      mode=current_state.keyboard_mode)

                    next_state = SearchState(keys=candidate_keys,
                                             score=dictionary.get_score_for_string(candidate_word, should_aggregate=should_aggregate),
                                             keyboard_mode=next_keyboard,
                                             was_on_autocomplete=False,
                                             current_key=neighbor_key,
                                             center_key=neighbor_key)

                    candidate_queue.put((-1 * next_state.score, next_state))
                    visited.add(visited_state)

        # Consider autocomplete
        if (prev_key in string.ascii_letters) and ((num_moves[move_idx] == 1) or (num_moves[move_idx] <= 2 and current_state.was_on_autocomplete)) and (len(current_string) > 0):

            if len(current_string) == 1:
                autocomplete = single_autocomplete[current_string[0]]

                scores = [0.25, 0.125, 0.125, 0.125]
                scores = scores[0:len(autocomplete)]

                sorted_keys = [(letter, score) for letter, score in zip(autocomplete, scores)]
            else:
                autocomplete_counts = dictionary.get_letter_counts(prefix=current_string, should_smooth=False, length=None)
                total_count = sum(autocomplete_counts.values())
                next_key_freq = { key: count / total_count for key, count in autocomplete_counts.items() }

                sorted_keys = list(reversed(sorted(next_key_freq.items(), key=lambda t: t[1])))
                sorted_keys = list(filter(lambda t: t[0] in string.ascii_letters, sorted_keys))

            #autocomplete_base_keys = graph.get_keys_for_moves_from(start_key=current_state.center_key,
            #                                                       num_moves=1,
            #                                                       mode=current_state.keyboard_mode)

            # Use scores that take the max-length into account
            total_count = sum(next_key_counts.values())
            next_key_scores = { key: count / total_count for key, count in next_key_counts.items() }

            for (neighbor_key, _) in sorted_keys:
                score = next_key_scores[neighbor_key]

                if score < 0.001:
                    continue

                candidate_keys = current_state.keys + [neighbor_key]
                candidate_word = ''.join(candidate_keys)

                visited_state = VisitedState(string=candidate_word, was_on_autocomplete=True)
                should_aggregate = len(candidate_keys) < target_length

                if visited_state not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                  mode=current_state.keyboard_mode)

                    next_state = SearchState(keys=candidate_keys,
                                             score=dictionary.get_score_for_string(candidate_word, should_aggregate=should_aggregate),
                                             keyboard_mode=current_state.keyboard_mode,
                                             was_on_autocomplete=True,
                                             current_key=neighbor_key,
                                             center_key=current_state.center_key)

                    candidate_queue.put((-1 * next_state.score, next_state))
                    visited.add(candidate_word)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
    parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
    parser.add_argument('--target', type=str, required=True, help='The target string.')
    args = parser.parse_args()

    graph = MultiKeyboardGraph()

    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary()
    else:
        dictionary = EnglishDictionary.restore(path=args.dictionary_path)

    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves_autocomplete(num_moves=args.moves_list, graph=graph, dictionary=dictionary, max_num_results=None)):
        if idx >= 100:
            break

        #print('Guess: {}, Score: {}'.format(guess, score))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
