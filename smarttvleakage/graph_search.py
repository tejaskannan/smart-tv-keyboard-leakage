from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.graphs.english_dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode'])


def filter_and_normalize_scores(key_probs: Dict[str, float], candidate_keys: List[str]) -> Dict[str, float]:
    filtered_scores = { key: key_probs[key] for key in candidate_keys if key in key_probs }
    score_sum = sum(filtered_scores.values())
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
            characters.pop()
        elif key not in UNPRINTED_CHARACTERS:
            if caps_lock or ((idx > 0) and (keys[idx - 1] == '<CAPS>') and (not prev_turn_off_caps_lock)):
                character = key.upper()
            else:
                character = CHARACTER_TRANSLATION.get(key, key)

            characters.append(character)
            prev_turn_off_caps_lock = False

    return characters


def get_words_from_moves(num_moves: List[int], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, max_num_results: Optional[int]) -> Iterable[Tuple[str, float]]:
    target_length = len(num_moves)

    candidate_queue = PriorityQueue()

    init_state = SearchState(keys=[], score=1.0, keyboard_mode=KeyboardMode.STANDARD)
    candidate_queue.put((-1 * init_state.score, init_state))

    scores: Dict[str, float] = dict()
    visited: Set[str] = set()

    result_count = 0

    while not candidate_queue.empty():
        _, current_state = candidate_queue.get()

        current_characters = get_characters_from_keys(keys=current_state.keys)
        current_string = ''.join(current_characters)

        if len(current_state.keys) == target_length:
            yield current_string, current_state.score

            result_count += 1

            if (max_num_results is not None) and (result_count >= max_num_results):
                return

            continue

        move_idx = len(current_state.keys)
        prev_key = current_state.keys[-1] if move_idx > 0 else START_KEYS[current_state.keyboard_mode]

        neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                  num_moves=num_moves[move_idx],
                                                  mode=current_state.keyboard_mode)

        next_key_probs = dictionary.get_letter_freq(prefix=current_string, total_length=target_length)

        filtered_probs = filter_and_normalize_scores(key_probs=next_key_probs,
                                                     candidate_keys=neighbors)

        for neighbor_key, score in filtered_probs.items():
            candidate_keys = current_state.keys + [neighbor_key]
            candidate_word = ''.join(candidate_keys)

            if candidate_word not in visited:
                next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                  mode=current_state.keyboard_mode)

                next_state = SearchState(keys=candidate_keys,
                                         score=score * current_state.score,
                                         keyboard_mode=next_keyboard)

                candidate_queue.put((-1 * next_state.score, next_state))
                visited.add(candidate_word)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary JSON or text file.')
    parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
    parser.add_argument('--target', type=str, required=True, help='The target string.')
    args = parser.parse_args()

    graph = MultiKeyboardGraph()

    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary()
    else:
        dictionary = EnglishDictionary(path=args.dictionary_path)

    for idx, (guess, score) in enumerate(get_words_from_moves(num_moves=args.moves_list, graph=graph, dictionary=dictionary, max_num_results=None)):
        if args.target == guess:
            print('Found {}. Rank {}'.format(guess, idx + 1))
            break
