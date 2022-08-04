import time
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.audio import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT, SAMSUNG_DELETE, APPLETV_KEYBOARD_DELETE
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, APPLETV_SEARCH_ALPHABET, SAMSUNG_STANDARD
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SPACE, SELECT_SOUND_KEYS, DELETE_SOUND_KEYS
from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
from smarttvleakage.utils.mistake_model import DecayingMistakeModel


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode', 'current_key', 'move_idx'])
VisitedState = namedtuple('VisitedState', ['keys', 'current_key'])
CandidateMove = namedtuple('CandidateMove', ['num_moves', 'adjustment', 'increment'])

MISTAKE_RATE = 1e-3
DECAY_RATE = 0.9
SUGGESTION_THRESHOLD = 8
SUGGESTION_FACTOR = 2.0
CUTOFF = 0.05


def get_words_from_moves(move_sequence: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, tv_type: SmartTVType, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    target_length = len(move_sequence)

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
                             score=1.0,
                             keyboard_mode=keyboard_mode,
                             current_key=START_KEYS[keyboard_mode],
                             move_idx=0)
    candidate_queue.put((-1 * init_state.score, init_state))

    scores: Dict[str, float] = dict()
    visited: Set[VisitedState] = set()
    guessed_strings: Set[str] = set()

    result_count = 0
    candidate_count = 0

    mistake_model = DecayingMistakeModel(mistake_rate=MISTAKE_RATE,
                                         decay_rate=DECAY_RATE,
                                         suggestion_threshold=SUGGESTION_THRESHOLD,
                                         suggestion_factor=SUGGESTION_FACTOR)

    while not candidate_queue.empty():
        _, current_state = candidate_queue.get()

        current_string = get_string_from_keys(keys=current_state.keys)
        candidate_count += 1

        if len(current_state.keys) == target_length:

            if current_string not in guessed_strings:
                yield current_string, current_state.score, candidate_count

                result_count += 1
                guessed_strings.add(current_string)

                if (max_num_results is not None) and (result_count >= max_num_results):
                    return

            continue

        move_idx = current_state.move_idx

        if move_idx >= len(move_sequence):
            continue

        num_moves = move_sequence[move_idx].num_moves
        end_sound = move_sequence[move_idx].end_sound
        prev_key = current_state.current_key

        # This is a quirk of Apple TV. From the search bar, the first move is the move onto the keyboard. We remove this choice.
        if (keyboard_type == KeyboardType.APPLE_TV_SEARCH) and (move_idx == 0):
            num_moves = max(num_moves - 1, 0)

        move_candidates: List[CandidateMove] = [CandidateMove(num_moves=num_moves, adjustment=1.0, increment=1)]

        if num_moves > 2:
            candidate_num_moves = num_moves - 1
            num_mistakes = 0

            while candidate_num_moves >= 1:
                adjustment = mistake_model.get_mistake_prob(move_num=move_idx,
                                                            num_moves=num_moves,
                                                            num_mistakes=num_mistakes)

                candidate_move = CandidateMove(num_moves=candidate_num_moves, adjustment=adjustment, increment=1)
                move_candidates.append(candidate_move)

                num_mistakes += 1
                candidate_num_moves -= 1

            # Include one more in case we messed up the audio extraction (e.g., on double moves)
            candidate_num_moves = num_moves + 1
            adjustment = mistake_model.get_mistake_prob(move_num=move_idx,
                                                        num_moves=candidate_num_moves,
                                                        num_mistakes=1)

            candidate_move = CandidateMove(num_moves=candidate_num_moves, adjustment=adjustment, increment=1)
            move_candidates.append(candidate_move)

        # Get the counts for the next keys
        next_key_counts = dictionary.get_letter_counts(prefix=current_string,
                                                       length=target_length,
                                                       should_smooth=True)

        for candidate_move in move_candidates:
            # Get the neighboring keys for this number of moves
            neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                      num_moves=candidate_move.num_moves,
                                                      mode=current_state.keyboard_mode,
                                                      use_shortcuts=True,
                                                      use_wraparound=True)

            # Filter out any unclickable keys (could not have selected those)
            neighbors = list(filter(lambda n: (not graph.is_unclickable(n, current_state.keyboard_mode)), neighbors))

            if (end_sound == delete_sound_name):
                neighbors = list(filter(lambda n: (n in DELETE_SOUND_KEYS), neighbors))
                filtered_probs = { n: (1.0 / len(neighbors)) for n in neighbors }
            elif (tv_type == SmartTVType.SAMSUNG) and (end_sound == SAMSUNG_SELECT):
                neighbors = list(filter(lambda n: (n in SELECT_SOUND_KEYS), neighbors))
                filtered_probs = { n: (1.0 / len(neighbors)) for n in neighbors }
            else:
                if tv_type == SmartTVType.SAMSUNG:
                    neighbors = list(filter(lambda n: (n not in SELECT_SOUND_KEYS), neighbors))
                elif tv_type == SmartTVType.APPLE_TV:
                    neighbors = list(filter(lambda n: (n not in DELETE_SOUND_KEYS), neighbors))

                filtered_probs = filter_and_normalize_scores(key_counts=next_key_counts,
                                                             candidate_keys=neighbors)

            for neighbor_key, score in filtered_probs.items():
                candidate_keys = current_state.keys + [neighbor_key]
                candidate_word = get_string_from_keys(candidate_keys)
                visited_str = ' '.join(candidate_keys)
                visited_state = VisitedState(keys=visited_str, current_key=neighbor_key)

                should_aggregate_score = len(candidate_keys) < target_length

                if visited_state not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                      mode=current_state.keyboard_mode,
                                                      keyboard_type=keyboard_type)

                    next_state_score = current_state.score * score * candidate_move.adjustment
                    next_move_idx = move_idx + candidate_move.increment

                    next_state = SearchState(keys=candidate_keys,
                                             score=next_state_score,
                                             keyboard_mode=next_keyboard,
                                             current_key=neighbor_key,
                                             move_idx=next_move_idx)

                    candidate_queue.put((-1 * next_state.score, next_state))
                    visited.add(visited_state)

                    # Add any linked states (undetectable by keyboard audio alone)
                    for linked_state in graph.get_linked_states(neighbor_key, keyboard_mode=next_keyboard):
                        next_state = SearchState(keys=candidate_keys,
                                                 score=next_state_score,
                                                 keyboard_mode=linked_state.mode,
                                                 current_key=linked_state.key,
                                                 move_idx=next_move_idx)

                        candidate_queue.put((-1 * next_state.score, next_state))
                        visited_state = VisitedState(keys=visited_str, current_key=linked_state.key)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
    parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
    parser.add_argument('--sounds-list', type=str, nargs='*', choices=[SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT], help='An optional space-separated sequence of end sounds. If none, we assume all sounds are `key_select`.')
    parser.add_argument('--tv-type', type=str, required=True, choices=[SmartTVType.SAMSUNG.name.lower(), SmartTVType.APPLE_TV.name.lower()], help='The name of the TV type to use.')
    parser.add_argument('--target', type=str, required=True, help='The target string.')
    args = parser.parse_args()

    tv_type = SmartTVType[args.tv_type.upper()]
    graph = MultiKeyboardGraph(tv_type=tv_type)
    characters = graph.get_characters()

    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary(characters=characters)
    else:
        dictionary = EnglishDictionary.restore(characters=characters, path=args.dictionary_path)

    if tv_type == SmartTVType.SAMSUNG:
        default_sound = SAMSUNG_KEY_SELECT
    elif tv_type == SmartTVType.APPLE_TV:
        default_sound = APPLETV_KEYBOARD_SELECT
    else:
        raise ValueError('Unknown TV type: {}'.format(args.tv_type))

    print('Target String: {}'.format(args.target))

    if (args.sounds_list is None) or (len(args.sounds_list) == 0):
        moves = [Move(num_moves=num_moves, end_sound=default_sound) for num_moves in args.moves_list]
    else:
        assert len(args.moves_list) == len(args.sounds_list), 'Must provide the same number of moves ({}) and sounds ({})'.format(len(args.moves_list), len(args.sounds_list))
        moves = [Move(num_moves=num_moves, end_sound=end_sound) for num_moves, end_sound in zip(args.moves_list, args.sounds_list)]

    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves(moves, graph=graph, dictionary=dictionary, tv_type=tv_type, max_num_results=10)):
        print('Guess: {}, Score: {}'.format(guess, score))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
