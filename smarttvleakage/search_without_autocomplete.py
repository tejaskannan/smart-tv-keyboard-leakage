import time
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.audio.move_extractor import Move, Sound
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SPACE, SELECT_SOUND_KEYS
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
from smarttvleakage.utils.mistake_model import DecayingMistakeModel


SearchState = namedtuple('SearchState', ['keys', 'score', 'keyboard_mode'])
MISTAKE_RATE = 1e-3
DECAY_RATE = 0.9
SUGGESTION_THRESHOLD = 8
SUGGESTION_FACTOR = 2.0


def get_words_from_moves(move_sequence: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, max_num_results: Optional[int]) -> Iterable[Tuple[str, float, int]]:
    target_length = len(move_sequence)

    candidate_queue = PriorityQueue()

    init_state = SearchState(keys=[],
                             score=1.0,
                             keyboard_mode=KeyboardMode.STANDARD)
    candidate_queue.put((-1 * init_state.score, init_state))

    scores: Dict[str, float] = dict()
    visited: Set[str] = set()

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
            yield current_string, current_state.score, candidate_count

            result_count += 1

            if (max_num_results is not None) and (result_count >= max_num_results):
                return

            continue

        move_idx = len(current_state.keys)
        num_moves = move_sequence[move_idx].num_moves
        end_sound = move_sequence[move_idx].end_sound
        prev_key = current_state.keys[-1] if move_idx > 0 else START_KEYS[current_state.keyboard_mode]

        move_candidates: Dict[int, float] = {
            num_moves: 1.0
        }

        if num_moves > 3:
            move_candidates[num_moves - 1] = mistake_model.get_mistake_prob(move_num=move_idx,
                                                                            num_moves=num_moves,
                                                                            num_mistakes=1)

        if num_moves > 4:
            move_candidates[num_moves - 2] = mistake_model.get_mistake_prob(move_num=move_idx,
                                                                            num_moves=num_moves,
                                                                            num_mistakes=2)

        for candidate_moves, adjustment_factor in move_candidates.items():

            neighbors = graph.get_keys_for_moves_from(start_key=prev_key,
                                                      num_moves=candidate_moves,
                                                      mode=current_state.keyboard_mode,
                                                      use_space=(end_sound == Sound.SELECT) or (prev_key == SPACE))

            next_key_counts = dictionary.get_letter_counts(prefix=current_string,
                                                           length=target_length,
                                                           should_smooth=True)

            if end_sound == Sound.SELECT:
                neighbors = list(filter(lambda n: (n in SELECT_SOUND_KEYS), neighbors))
                filtered_probs = { n: (1.0 / len(neighbors)) for n in neighbors }
            else:
                neighbors = list(filter(lambda n: (n not in SELECT_SOUND_KEYS), neighbors))
                filtered_probs = filter_and_normalize_scores(key_counts=next_key_counts,
                                                             candidate_keys=neighbors)

            for neighbor_key, score in filtered_probs.items():
                candidate_keys = current_state.keys + [neighbor_key]
                candidate_word = get_string_from_keys(candidate_keys)

                if candidate_word not in visited:
                    next_keyboard = get_keyboard_mode(key=neighbor_key,
                                                      mode=current_state.keyboard_mode)

                    next_state = SearchState(keys=candidate_keys,
                                             score=score * current_state.score * adjustment_factor,
                                             keyboard_mode=next_keyboard)

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

    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves(args.moves_list, graph=graph, dictionary=dictionary, max_num_results=None)):
        print('Guess: {}, Score: {}'.format(guess, score))

        if args.target == guess:
            print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
            break
