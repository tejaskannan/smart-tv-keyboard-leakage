import time
import string
import os.path
import numpy as np
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple, defaultdict, Counter
from typing import Set, List, Dict, Optional, Iterable, Tuple, DefaultDict

from smarttvleakage.audio import Move, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT, SAMSUNG_DELETE
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, SPACE, SAMSUNG_STANDARD
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SELECT_SOUND_KEYS, DELETE_SOUND_KEYS
from smarttvleakage.utils.constants import END_CHAR
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
from smarttvleakage.utils.file_utils import read_json


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode', 'was_on_suggested', 'current_key', 'center_key'])
VisitedState = namedtuple('VisitedState', ['string', 'was_on_suggested'])

INCORRECT_FACTOR = 1e-3
SUGGESTION_THRESHOLD = 1e-3
MAX_COUNT_PER_PREFIX = 10
TOP_KEYS = 2
TOP_KEY_FACTOR = 1e-1


def get_candidate_prefix(string: str) -> str:
    tokens = string.split()
    if len(tokens) == 0:
        return ''
    elif len(tokens) == 1:
        return tokens[-1]
    elif len(tokens[-1]) > 0:
        return tokens[-1]
    else:
        return tokens[-2]


def get_current_prefix(string: str) -> str:
    tokens = string.split()
    if (len(tokens) == 0) or string.endswith(' '):
        return ''
    else:
        return tokens[-1]


def apply_autocomplete(prefixes: List[str], dictionary: CharacterDictionary, min_length: int, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    for idx, (word, score) in enumerate(dictionary.get_words_for(prefixes, max_num_results=max_num_results, min_length=min_length, max_count_per_prefix=MAX_COUNT_PER_PREFIX)):
        if idx >= max_num_results:
            break

        yield word, score, 0  # TODO: Fix the candidates here


def get_words_from_moves_suggestions(move_sequence: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, did_use_autocomplete: bool, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    # Remove false positive zeros from the front of the move sequence (use the rule: q always followed by u)
    while (len(move_sequence) > 1) and (move_sequence[0].num_moves == 0) and (move_sequence[1].num_moves != 1):
        move_sequence.pop(0)

    target_length = len(move_sequence)

    # Split the string by assuming 'select' keys correspond to spaces (TODO: Fix dynamically if we don't take a space)
    # Note that we only use this routine for Samsung keyboards
    string_lengths = [None]
    if not did_use_autocomplete:
        string_lengths = []
        count = 0

        for move in move_sequence:
            if move.end_sound == SAMSUNG_SELECT:
                string_lengths.append(count)
                count = 0
            else:
                count += 1

        if move_sequence[-1].end_sound != SAMSUNG_SELECT:
            string_lengths.append(count)

    # Initialize a minimum priority queue
    candidate_queue = PriorityQueue()

    # Create the start state, which has reached zero keys
    init_state = SearchState(keys=[],
                             score=0.0,
                             keyboard_mode=SAMSUNG_STANDARD,
                             was_on_suggested=False,
                             current_key=None,
                             center_key=None)
    candidate_queue.put((-1 * init_state.score, init_state))

    keyboard_type = graph.get_keyboard_type()

    scores: Dict[str, float] = dict()
    visited: Set[VisitedState] = set()
    seen_strings: Set[str] = set()

    result_count = 0
    candidate_count = 0

    # Calculate the minimum string length (accounting for deletes)
    min_length = target_length - 2 * sum(int(move.end_sound == SAMSUNG_DELETE) for move in move_sequence)
    min_length += int(did_use_autocomplete)

    while not candidate_queue.empty():
        # Pop the minimum element off the queue
        pq_score, current_state = candidate_queue.get()

        # Make the string from the given keys
        current_string = get_string_from_keys(keys=current_state.keys)
        candidate_count += 1

        # Check the stopping condition (whether we reach the target number of keys)
        if len(current_state.keys) == target_length:
            end_score = dictionary.get_letter_counts(current_string, length=target_length).get(END_CHAR, 0.0)

            end_state = SearchState(keys=current_state.keys + [END_CHAR],
                                    score=current_state.score - np.log(end_score),
                                    keyboard_mode=current_state.keyboard_mode,
                                    current_key=current_state.current_key,
                                    center_key=current_state.center_key,
                                    was_on_suggested=current_state.was_on_suggested)
            candidate_queue.put((end_state.score, end_state))
            continue

        if current_string.endswith(END_CHAR):
            # Make sure we do not produce duplicate strings
            if current_string not in seen_strings:
                yield current_string.replace(END_CHAR, ''), current_state.score, candidate_count
                result_count += 1

                if (max_num_results is not None) and (result_count >= max_num_results):
                    return

            seen_strings.add(current_string)
            continue

        move_idx = len(current_state.keys)
        num_moves = move_sequence[move_idx].num_moves
        end_sound = move_sequence[move_idx].end_sound
        prev_key = current_state.center_key if len(current_state.keys) > 0 else START_KEYS[current_state.keyboard_mode]

        # Hold adjustment factors for move scores. If the last key was a letter (after the first move), then the keyboard
        # will suggest keys in the adjacent spaces. This reduces the amount of actual positions moved on the keyboard.
        suggestion_adjustment = 0

        if (len(current_state.keys) > 0) and (num_moves > 2) and (prev_key in string.ascii_letters):
            suggestion_adjustment = 1

            if current_state.center_key != current_state.current_key:
                suggestion_adjustment += 1

        move_score_factors: Dict[int, float] = dict()

        if (len(current_state.keys) > 0) and (num_moves >= 2) and (current_state.center_key in string.ascii_letters):
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

        string_length_idx = sum(int(sound == SAMSUNG_SELECT) for sound in move_sequence[0:move_idx + 1])
        string_length = string_lengths[string_length_idx] if (not did_use_autocomplete) else None

        # Get the unnormalized scores for the next keys
        current_prefix = get_current_prefix(current_string)
        next_key_counts = dictionary.get_letter_counts(prefix=current_prefix,
                                                       length=string_length)

        # If we didn't take 1 move (or are on the first move), then the user may be using the normal keyboard
        if (num_moves != 1) or (len(current_state.keys) == 0):
            # Get the top N suggested keys (to predict the suggestions)
            top_keys_counter: Counter = Counter()
            for key, count in next_key_counts.items():
                top_keys_counter[key] = count

            if len(current_state.keys) > 0:
                top_keys = set(map(lambda t: t[0], top_keys_counter.most_common(TOP_KEYS)))
            else:
                top_keys = set()

            for move_count, move_score in move_score_factors.items():
                # Get neighboring keys and normalize the resulting scores
                neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                          num_moves=move_count,
                                                          mode=current_state.keyboard_mode,
                                                          use_shortcuts=True,
                                                          use_wraparound=True)

                if end_sound == SAMSUNG_SELECT:
                    neighbors = list(filter(lambda n: (n in SELECT_SOUND_KEYS), neighbors))
                    filtered_probs = {n: (1.0 / len(neighbors)) for n in neighbors}
                elif end_sound == SAMSUNG_DELETE:
                    neighbors = list(filter(lambda n: (n in DELETE_SOUND_KEYS), neighbors))
                    filtered_probs = {n: (1.0 / len(neighbors)) for n in neighbors}
                else:
                    neighbors = list(filter(lambda n: (n not in SELECT_SOUND_KEYS) and (n not in DELETE_SOUND_KEYS), neighbors))
                    filtered_probs = filter_and_normalize_scores(key_counts=next_key_counts,
                                                                 candidate_keys=neighbors,
                                                                 should_renormalize=False)

                for neighbor_key, freq in filtered_probs.items():
                    candidate_keys = current_state.keys + [neighbor_key]
                    candidate_string = get_string_from_keys(candidate_keys)

                    visited_state = VisitedState(string=candidate_string, was_on_suggested=False)

                    #should_aggregate_score = (len(candidate_keys) < target_length) or (did_use_autocomplete)
                    score_factor = TOP_KEY_FACTOR if (neighbor_key in top_keys) else 1.0

                    # Make the next state
                    if visited_state not in visited:
                        next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                          mode=current_state.keyboard_mode,
                                                          keyboard_type=keyboard_type)

                        adjusted_score = score_factor * move_score * freq
                        string_score = current_state.score - np.log(adjusted_score)

                        next_state = SearchState(keys=candidate_keys,
                                                 score=string_score,
                                                 keyboard_mode=next_keyboard,
                                                 was_on_suggested=False,
                                                 current_key=neighbor_key,
                                                 center_key=neighbor_key)

                        candidate_queue.put((next_state.score, next_state))
                        visited.add(visited_state)

        # Consider suggested_keys
        if (prev_key in string.ascii_letters) and (num_moves <= 2) and (len(current_state.keys) > 0):
            autocomplete_counts = dictionary.get_letter_counts(prefix=current_prefix, length=None)
            total_count = sum(autocomplete_counts.values())
            next_key_freq = {key: count / total_count for key, count in autocomplete_counts.items()}

            sorted_keys = list(reversed(sorted(next_key_freq.items(), key=lambda t: t[1])))
            sorted_keys = list(filter(lambda t: (t[0] in string.ascii_letters), sorted_keys))

            # Use scores that take the max-length into account
            total_count = sum(next_key_counts.values())
            next_key_scores = {key: count / total_count for key, count in next_key_counts.items()}

            for (neighbor_key, _) in sorted_keys:
                freq = next_key_scores.get(neighbor_key, 0.0)

                if freq < SUGGESTION_THRESHOLD:
                    continue

                candidate_keys = current_state.keys + [neighbor_key]
                candidate_string = get_string_from_keys(candidate_keys)

                visited_state = VisitedState(string=candidate_string, was_on_suggested=True)

                if visited_state not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                      mode=current_state.keyboard_mode,
                                                      keyboard_type=keyboard_type)

                    string_score = current_state.score - np.log(freq)

                    next_state = SearchState(keys=candidate_keys,
                                             score=string_score,
                                             keyboard_mode=current_state.keyboard_mode,
                                             was_on_suggested=True,
                                             current_key=neighbor_key,
                                             center_key=current_state.center_key)

                    candidate_queue.put((next_state.score, next_state))
                    visited.add(visited_state)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
    parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
    parser.add_argument('--sounds-list', type=str, required=True, nargs='+', choices=[SAMSUNG_SELECT, SAMSUNG_KEY_SELECT], help='A space-separated sequence of sounds for each selection')
    parser.add_argument('--target', type=str, required=True, help='The target string.')
    args = parser.parse_args()

    graph = MultiKeyboardGraph()
    characters = graph.get_characters()

    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary(characters=characters)
    else:
        dictionary = EnglishDictionary.restore(characters=characters, path=args.dictionary_path)

    moves = [Move(num_moves=m, end_sound=s) for m, s in zip(args.moves_list, args.sounds_list)]

    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves_autocomplete(move_sequence=moves, graph=graph, dictionary=dictionary, max_num_results=None, did_use_autocomplete=False)):
        if idx >= 25:
            break

        print('Guess: {}, Score: {}'.format(guess, score))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
