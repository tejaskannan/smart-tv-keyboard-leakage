import time
import numpy as np
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.audio import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT, SAMSUNG_DELETE, APPLETV_KEYBOARD_DELETE
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, APPLETV_SEARCH_ALPHABET, SAMSUNG_STANDARD
from smarttvleakage.dictionary import CharacterDictionary, restore_dictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SPACE, SELECT_SOUND_KEYS, DELETE_SOUND_KEYS, DONE
from smarttvleakage.dictionary import NumericDictionary
from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.utils.constants import SmartTVType, KeyboardType, END_CHAR, Direction, SMALL_NUMBER
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
from smarttvleakage.utils.mistake_model import DecayingMistakeModel
from smarttvleakage.keyboard_utils.word_to_move import findPath


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode', 'current_key', 'move_idx'])
VisitedState = namedtuple('VisitedState', ['keys', 'current_key'])
CandidateMove = namedtuple('CandidateMove', ['num_moves', 'adjustment'])

MISTAKE_RATE = 1e-2
DECAY_RATE = 0.9
SCORE_THRESHOLD = 1e-4
MAX_NUM_CANDIDATES = 10000
MISTAKE_LIMIT = 3

SUGGESTION_THRESHOLD = 8
SUGGESTION_FACTOR = 2.0
CUTOFF = 0.05


def get_words_from_moves(move_sequence: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, tv_type: SmartTVType, max_num_results: Optional[int], precomputed: PasswordRainbow, includes_done: bool, start_key: str, is_searching_reverse: bool) -> Iterable[Tuple[str, float, int]]:
    # Variables to track progress
    guessed_strings: Set[str] = set()
    result_count = 0
    candidate_count = 0

    # If provided, use the precomputed index to look up this move sequence. We start with these guesses
    if precomputed is not None:
        rainbow_results = precomputed.get_strings_for_seq(move_sequence, tv_type=tv_type, max_num_results=max_num_results)

        for idx, result in enumerate(rainbow_results):
            if (max_num_results is not None) and (result_count > max_num_results):
                return

            result_count += 1 
            candidate_count += 1
            yield result.word, result.score, candidate_count

            guessed_strings.add(result.word)

    target_length = len(move_sequence)
    if includes_done:
        target_length -= 1

    candidate_queue = PriorityQueue()

    # Set the default keyboard mode
    delete_sound_name = ''
    if tv_type == SmartTVType.SAMSUNG:
        delete_sound_name = SAMSUNG_DELETE
    elif tv_type == SmartTVType.APPLE_TV:
        delete_sound_name = APPLETV_KEYBOARD_DELETE
    else:
        raise ValueError('Unknown TV type: {}'.format(tv_type.name))

    # Create initial state
    keyboard_mode = graph.get_start_keyboard_mode()
    keyboard_type = graph.get_keyboard_type()

    init_state = SearchState(keys=[],
                             score=0.0,
                             keyboard_mode=keyboard_mode,
                             current_key=start_key,
                             move_idx=0)
    candidate_queue.put((init_state.score, init_state))

    # Link the initial state (we can start in any one of the these spots)
    if not isinstance(dictionary, NumericDictionary):
        for linked_state in graph.get_linked_states(start_key, keyboard_mode=keyboard_mode):
            init_state = SearchState(keys=[],
                                     score=0.0,
                                     keyboard_mode=linked_state.mode,
                                     current_key=linked_state.key,
                                     move_idx=0)
            candidate_queue.put((init_state.score, init_state))

    scores: Dict[str, float] = dict()
    visited: Set[VisitedState] = set()
    guessed_strings: Set[str] = set()

    mistake_model = DecayingMistakeModel(mistake_rate=MISTAKE_RATE,
                                         decay_rate=DECAY_RATE,
                                         suggestion_threshold=SUGGESTION_THRESHOLD,
                                         suggestion_factor=SUGGESTION_FACTOR)

    while not candidate_queue.empty():
        _, current_state = candidate_queue.get()

        current_string = get_string_from_keys(keys=current_state.keys)
        candidate_count += 1
        move_idx = current_state.move_idx

        if len(current_state.keys) == target_length:
            if dictionary.is_valid(current_string):
                end_score = dictionary.get_letter_counts(current_string, length=target_length).get(END_CHAR, SMALL_NUMBER)
                next_score = current_state.score - np.log(end_score)

                end_state = SearchState(keys=current_state.keys + [END_CHAR],
                                        score=next_score,
                                        keyboard_mode=current_state.keyboard_mode,
                                        current_key=current_state.current_key,
                                        move_idx=current_state.move_idx)
                candidate_queue.put((end_state.score, end_state))

            continue

        if current_string.endswith(END_CHAR):
            is_valid = (current_string not in guessed_strings)

            if includes_done:
                moves_to_done = move_sequence[move_idx].num_moves
                candidate_keys = graph.get_keys_for_moves_from(start_key=current_state.current_key,
                                                               num_moves=moves_to_done,
                                                               use_shortcuts=True,
                                                               use_wraparound=True,
                                                               directions=Direction.ANY,
                                                               mode=current_state.keyboard_mode)
                is_valid = is_valid and (DONE in candidate_keys)

            if is_valid:
                yield current_string.replace(END_CHAR, ''), current_state.score, candidate_count

                result_count += 1
                guessed_strings.add(current_string)

                if (max_num_results is not None) and (result_count >= max_num_results):
                    return

            continue

        if candidate_count >= MAX_NUM_CANDIDATES:
            break

        if move_idx >= len(move_sequence):
            continue

        num_moves = move_sequence[move_idx].num_moves
        end_sound = move_sequence[move_idx].end_sound
        prev_key = current_state.current_key

        # This is a quirk of Apple TV. From the search bar, the first move is the move onto the keyboard. We remove this choice.
        if (keyboard_type == KeyboardType.APPLE_TV_SEARCH) and (move_idx == 0):
            num_moves = max(num_moves - 1, 0)

        move_candidates: List[CandidateMove] = [CandidateMove(num_moves=num_moves, adjustment=1.0)]

        if num_moves > 2:
            candidate_num_moves = num_moves - 1

            for num_mistakes in range(1, MISTAKE_LIMIT + 1):
                adjustment = mistake_model.get_mistake_prob(move_num=move_idx,
                                                            num_moves=num_moves,
                                                            num_mistakes=num_mistakes)

                candidate_move = CandidateMove(num_moves=candidate_num_moves, adjustment=adjustment)
                move_candidates.append(candidate_move)

                candidate_num_moves -= 1

            # Include one more in case we messed up the audio extraction (e.g., on double moves)
            candidate_num_moves = num_moves + 1
            adjustment = mistake_model.get_mistake_prob(move_num=move_idx,
                                                        num_moves=candidate_num_moves,
                                                        num_mistakes=1)

            candidate_move = CandidateMove(num_moves=candidate_num_moves, adjustment=adjustment)
            move_candidates.append(candidate_move)

        # Get the counts for the next keys
        next_key_counts = dictionary.get_letter_counts(prefix=current_string,
                                                       length=target_length)

        total_count = sum(next_key_counts.values())
        next_key_counts = {key: (count / total_count) for key, count in next_key_counts.items()}

        for candidate_move in move_candidates:
            # Get the neighboring keys for this number of moves
            directions = move_sequence[move_idx].directions if candidate_move.num_moves == num_moves else Direction.ANY

            if is_searching_reverse:
                neighbors = graph.get_keys_for_moves_to(end_key=prev_key,
                                                        num_moves=candidate_move.num_moves,
                                                        mode=current_state.keyboard_mode,
                                                        use_shortcuts=True,
                                                        use_wraparound=True)
            else:
                neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                          num_moves=candidate_move.num_moves,
                                                          mode=current_state.keyboard_mode,
                                                          use_shortcuts=True,
                                                          use_wraparound=True,
                                                          directions=directions)

            # Filter out any unclickable keys (could not have selected those)
            neighbors = list(filter(lambda n: (not graph.is_unclickable(n, current_state.keyboard_mode)), neighbors))

            if (end_sound == delete_sound_name):
                neighbors = list(filter(lambda n: (n in DELETE_SOUND_KEYS), neighbors))
                filtered_probs = {n: (1.0 / len(neighbors)) for n in neighbors}
            elif (tv_type == SmartTVType.SAMSUNG) and (end_sound == SAMSUNG_SELECT):
                neighbors = list(filter(lambda n: (n in SELECT_SOUND_KEYS), neighbors))
                filtered_probs = {n: (1.0 / len(neighbors)) for n in neighbors}
            else:
                if tv_type == SmartTVType.SAMSUNG:
                    neighbors = list(filter(lambda n: (n not in SELECT_SOUND_KEYS) and (n not in DELETE_SOUND_KEYS), neighbors))
                elif tv_type == SmartTVType.APPLE_TV:
                    neighbors = list(filter(lambda n: (n not in DELETE_SOUND_KEYS), neighbors))

                filtered_probs = filter_and_normalize_scores(key_counts=next_key_counts,
                                                             candidate_keys=neighbors,
                                                             should_renormalize=False)

            for neighbor_key, score in filtered_probs.items():
                adjusted_score = score * candidate_move.adjustment

                # Skip strings below a threshold. These are so low scoring, we are unlikely
                # to consider them later on.
                if adjusted_score < SCORE_THRESHOLD:
                    continue

                candidate_keys = current_state.keys + [neighbor_key]
                candidate_word = get_string_from_keys(candidate_keys)
                visited_str = ' '.join(candidate_keys)
                visited_state = VisitedState(keys=visited_str, current_key=neighbor_key)

                if (len(candidate_keys) == target_length) and (neighbor_key in ('<CHANGE>', '<SPACE>')):
                    continue

                if visited_state not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                      mode=current_state.keyboard_mode,
                                                      keyboard_type=keyboard_type)


                    next_state_score = current_state.score - np.log(adjusted_score)
                    next_move_idx = current_state.move_idx + 1

                    # Project the remaining score (as a heuristic for A* search)
                    estimated_remaining = dictionary.projected_remaining_log_prob(candidate_word, length=target_length)
                    priority = next_state_score + estimated_remaining

                    next_state = SearchState(keys=candidate_keys,
                                             score=next_state_score,
                                             keyboard_mode=next_keyboard,
                                             current_key=neighbor_key,
                                             move_idx=next_move_idx)

                    candidate_queue.put((priority, next_state))
                    visited.add(visited_state)

                    # Add any linked states (undetectable by keyboard audio alone)
                    if not isinstance(dictionary, NumericDictionary):
                        for linked_state in graph.get_linked_states(neighbor_key, keyboard_mode=next_keyboard):
                            next_state = SearchState(keys=candidate_keys,
                                                     score=next_state_score,
                                                     keyboard_mode=linked_state.mode,
                                                     current_key=linked_state.key,
                                                     move_idx=next_move_idx)

                            candidate_queue.put((priority, next_state))
                            visited_state = VisitedState(keys=visited_str, current_key=linked_state.key)


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

    print('Restoring dictionary...')
    dictionary = restore_dictionary(args.dictionary_path)
    dictionary.set_characters(characters)

    precomputed = None
    if args.precomputed_path is not None:
        precomputed = PasswordRainbow(args.precomputed_path)

    if keyboard_type == KeyboardType.SAMSUNG:
        tv_type = SmartTVType.SAMSUNG
    else:
        tv_type = SmartTVType.APPLE_TV

    print('Target String: {}'.format(args.target))

    #use_done = (keyboard_type == KeyboardType.APPLE_TV_PASSWORD)
    use_done = True
    start_key = START_KEYS[graph.get_start_keyboard_mode()]
    moves = findPath(args.target, use_shortcuts=True, use_wraparound=True, use_done=use_done, mistake_rate=0.0, decay_rate=1.0, max_errors=0, keyboard=graph, start_key=start_key)

    #if (args.sounds_list is None) or (len(args.sounds_list) == 0):
    #    moves = [Move(num_moves=num_moves, end_sound=default_sound) for num_moves in args.moves_list]
    #else:
    #    assert len(args.moves_list) == len(args.sounds_list), 'Must provide the same number of moves ({}) and sounds ({})'.format(len(args.moves_list), len(args.sounds_list))
    #    moves = [Move(num_moves=num_moves, end_sound=end_sound) for num_moves, end_sound in zip(args.moves_list, args.sounds_list)]

    print(moves)

    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves(moves, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=args.max_num_results, precomputed=precomputed, includes_done=use_done, is_searching_reverse=False, start_key=start_key)):
        print('Guess: {}, Score: {}'.format(guess, score))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
