import time
import string
import os.path
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple, defaultdict
from typing import Set, List, Dict, Optional, Iterable, Tuple, DefaultDict

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
from smarttvleakage.utils.file_utils import read_json


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode', 'was_on_suggested', 'current_key', 'center_key'])
VisitedState = namedtuple('VisitedState', ['string', 'was_on_suggested'])

INCORRECT_FACTOR = 1e-3
SUGGESTION_THRESHOLD = 1e-3


def get_words_from_moves_autocomplete(move_sequence: List[int], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, did_use_autocomplete: bool, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    iterator = get_words_from_moves_autocomplete_helper(move_sequence, graph, dictionary, did_use_autocomplete, max_num_results)
    
    if did_use_autocomplete:
        prefix_list: List[str] = []
        num_results = 100
        for idx, (word, _, _)  in enumerate(iterator):
            if idx >= num_results:
                break

            prefix_list.append(word)

        prefix_counts: Dict[str, int] = defaultdict(int)

        #for word in dictionary.iterate_words('/local/dictionaries/enwiki-20210820-words-frequency.txt'):
        for word in dictionary.iterate_words('local/dictionaries/enwiki-20210820-words-frequency.txt'):

            for prefix in prefix_list:
                if word.startswith(prefix):
                    prefix_counts[prefix] += 1
                    if prefix_counts[prefix] > 25:
                        index = prefix_list.index(prefix)
                        prefix_list.pop(index)

                    yield word, 1.0, 0  # TODO: Fix the score and candidates here
    else:
        for result in iterator:
            yield result


def get_words_from_moves_autocomplete_helper(move_sequence: List[int], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, did_use_autocomplete: bool, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    directory = os.path.dirname(__file__)
    single_suggestions = read_json(os.path.join(directory, 'graphs/autocomplete.json'))

    # Remove false positive zeros from the front of the move sequence (use the rule: q always followed by u)
    while (len(move_sequence) > 1) and (move_sequence[0] == 0) and (move_sequence[1] != 1):
        move_sequence.pop(0)

    target_length = len(move_sequence)
    string_length = target_length if (not did_use_autocomplete) else None

    # Initialize a minimum priority queue
    candidate_queue = PriorityQueue()

    # Create the start state, which has reached zero keys
    init_state = SearchState(keys=[],
                             score=1.0,
                             keyboard_mode=KeyboardMode.STANDARD,
                             was_on_suggested=False,
                             current_key=None,
                             center_key=None)
    candidate_queue.put((-1 * init_state.score, init_state))

    scores: Dict[str, float] = dict()
    visited: Set[VisitedState] = set()
    seen_strings: Set[str] = set()

    result_count = 0
    candidate_count = 0

    while not candidate_queue.empty():
        # Pop the minimum element off the queue
        pq_score, current_state = candidate_queue.get()

        # Make the string from the given keys
        current_string = get_string_from_keys(keys=current_state.keys)
        candidate_count += 1

        # Check the stopping condition (whether we reach the target number of keys)
        if (len(current_state.keys) == target_length) or ((did_use_autocomplete) and (len(current_state.keys) == (target_length - 1))):

            # Make sure we do not produce duplicate strings
            if current_string not in seen_strings:
                yield current_string, current_state.score, candidate_count
                result_count += 1

                if (max_num_results is not None) and (result_count >= max_num_results):
                    return

            seen_strings.add(current_string)
            continue

        move_idx = len(current_state.keys)
        num_moves = move_sequence[move_idx]
        prev_key = current_state.center_key if len(current_state.keys) > 0 else START_KEYS[current_state.keyboard_mode]

        # Hold adjustment factors for move scores. If the last key was a letter (after the first move), then the keyboard
        # will suggest keys in the adjacent spaces. This reduces the amount of actual positions moved on the keyboard.
        suggestion_adjustment = 0

        if (len(current_state.keys) > 0) and (num_moves > 2) and (prev_key in string.ascii_letters):
            suggestion_adjustment = 1

            if current_state.center_key != current_state.current_key:
                suggestion_adjustment += 1

        move_score_factors: Dict[int, float] = dict()

        if (len(current_state.keys) > 0) and (num_moves > 2):
            move_score_factors[num_moves] = 1.0
            move_score_factors[max(num_moves - 1, 0)] = 1.0
            move_score_factors[max(num_moves - 2, 0)] = 1.0
        else:
            move_score_factors[num_moves] = 1.0

        #suggestion_adjustment = 1 if len(current_state.keys) > 0 and num_moves > 2 else 0
        #optimal_num_moves = max(num_moves - suggestion_adjustment, 0)

        #move_score_factors: Dict[int, float] = {
        #    optimal_num_moves: 1.0
        #}

        # Add in a mistake of 1 incorrect move (and another to correct)
        #if num_moves > 3:
        #    move_score_factors[num_moves - 2] = INCORRECT_FACTOR

        # Get the unnormalized scores for the next keys
        next_key_counts = dictionary.get_letter_counts(prefix=current_string,
                                                       length=string_length,
                                                       should_smooth=True)

        # If we didn't take 1 move (or are on the first move), then the user may be using the normal keyboard
        if (num_moves != 1) or (len(current_state.keys) == 0):
            for move_count, move_score in move_score_factors.items():
                # Get neighboring keys and normalize the resulting scores
                neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                          num_moves=move_count,
                                                          mode=current_state.keyboard_mode)

                filtered_probs = filter_and_normalize_scores(key_counts=next_key_counts,
                                                             candidate_keys=neighbors)

                for neighbor_key, score in filtered_probs.items():
                    candidate_keys = current_state.keys + [neighbor_key]
                    candidate_string = get_string_from_keys(candidate_keys)

                    visited_state = VisitedState(string=candidate_string, was_on_suggested=False)
                    should_aggregate_scores = len(candidate_keys) < target_length
    
                    # Make the next state
                    if visited_state not in visited:
                        next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                          mode=current_state.keyboard_mode)

                        string_score = dictionary.get_score_for_string(candidate_string, should_aggregate_scores) * move_score

                        next_state = SearchState(keys=candidate_keys,
                                                 score=string_score,
                                                 keyboard_mode=next_keyboard,
                                                 was_on_suggested=False,
                                                 current_key=neighbor_key,
                                                 center_key=neighbor_key)

                        candidate_queue.put((-1 * next_state.score, next_state))
                        visited.add(visited_state)

        # Consider suggested_keys
        if (prev_key in string.ascii_letters) and ((num_moves == 1) or (num_moves <= 3 and current_state.was_on_suggested)) and (len(current_state.keys) > 0):

            if len(current_state.keys) == 1:
                # Use the pre-collected autocomplete, biased towards the left
                autocomplete = single_suggestions[current_string[0]]

                scores = [0.25, 0.125, 0.125, 0.125]
                scores = scores[0:len(autocomplete)]

                sorted_keys = [(letter, score) for letter, score in zip(autocomplete, scores)]
            else:
                autocomplete_counts = dictionary.get_letter_counts(prefix=current_string, should_smooth=False, length=None)
                total_count = sum(autocomplete_counts.values())
                next_key_freq = { key: count / total_count for key, count in autocomplete_counts.items() }

                sorted_keys = list(reversed(sorted(next_key_freq.items(), key=lambda t: t[1])))
                sorted_keys = list(filter(lambda t: t[0] in string.ascii_letters, sorted_keys))

            # Use scores that take the max-length into account
            total_count = sum(next_key_counts.values())
            next_key_scores = { key: count / total_count for key, count in next_key_counts.items() }

            for (neighbor_key, _) in sorted_keys:
                score = next_key_scores.get(neighbor_key, 0.0)

                if score < SUGGESTION_THRESHOLD:
                    continue

                candidate_keys = current_state.keys + [neighbor_key]
                candidate_string = get_string_from_keys(candidate_keys)

                visited_state = VisitedState(string=candidate_string, was_on_suggested=True)
                should_aggregate_score = len(candidate_keys) < target_length

                if visited_state not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                  mode=current_state.keyboard_mode)

                    next_state = SearchState(keys=candidate_keys,
                                             score=dictionary.get_score_for_string(candidate_string, should_aggregate=should_aggregate_score),
                                             keyboard_mode=current_state.keyboard_mode,
                                             was_on_suggested=True,
                                             current_key=neighbor_key,
                                             center_key=current_state.center_key)

                    candidate_queue.put((-1 * next_state.score, next_state))
                    visited.add(visited_state)


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

    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves_autocomplete(move_sequence=args.moves_list, graph=graph, dictionary=dictionary, max_num_results=None, did_use_autocomplete=False)):
        if idx >= 100:
            break

        print('Guess: {}, Score: {}'.format(guess, score))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
