from argparse import ArgumentParser
from collections import deque, namedtuple
from typing import Set, List, Dict, Optional

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.graphs.english_dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary


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
        elif key != '<CHANGE>':
            if caps_lock or ((idx > 0) and (keys[idx - 1] == '<CAPS>') and (not prev_turn_off_caps_lock)):
                character = key.upper()
            else:
                character = key

            characters.append(character)
            prev_turn_off_caps_lock = False

    return characters


def get_words_from_moves(num_moves: List[int], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, max_num_results: Optional[int]) -> List[str]:
    target_length = len(num_moves)

    candidate_queue = deque()

    init_state = SearchState(keys=[], score=1.0, keyboard_mode=KeyboardMode.STANDARD)
    candidate_queue.append(init_state)

    scores: Dict[str, float] = dict()
    visited: Set[str] = set()

    while len(candidate_queue) > 0:
        current_state = candidate_queue.pop()

        current_characters = get_characters_from_keys(keys=current_state.keys)
        current_string = ''.join(current_characters)

        if len(current_state.keys) == target_length:
            scores[current_string] = current_state.score
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

                candidate_queue.append(next_state)
                visited.add(candidate_word)

    ranked_results = list(reversed(sorted(scores.items(), key=lambda t: t[1])))

    if max_num_results is not None:
        return ranked_results[0:max_num_results]

    return ranked_results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary JSON file.')
    parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
    args = parser.parse_args()

    graph = MultiKeyboardGraph()
    #dictionary = EnglishDictionary(path=args.dictionary_path)
    dictionary = UniformDictionary()

    results = get_words_from_moves(num_moves=args.moves_list, graph=graph, dictionary=dictionary, max_num_results=None)
    result_words = list(map(lambda t: t[0], results))

    target = 'Ns$a'
    print(target in result_words)
    print(len(result_words))
