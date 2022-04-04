from argparse import ArgumentParser
from collections import deque
from typing import Set, List, Dict

from smarttvleakage.graphs.keyboard_graph import KeyboardGraph
from smarttvleakage.graphs.english_dictionary import EnglishDictionary


def filter_and_normalize_scores(key_probs: Dict[str, float], candidate_keys: List[str]) -> Dict[str, float]:
    filtered_scores = { key: key_probs[key] for key in candidate_keys if key in key_probs }
    score_sum = sum(filtered_scores.values())
    return { key: (score / score_sum) for key, score in filtered_scores.items() }


def get_words_from_moves(num_moves: List[int], graph: KeyboardGraph, dictionary: EnglishDictionary, max_num_results: int) -> List[str]:
    target_length = len(num_moves)

    candidate_queue = deque()
    candidate_queue.append(('', 1.0))

    scores: Dict[str, float] = dict()
    visited: Set[str] = set()

    while len(candidate_queue) > 0:
        (current_string, current_score) = candidate_queue.pop()
        
        if len(current_string) == target_length:
            scores[current_string] = current_score
            continue

        move_idx = len(current_string)
        
        prev_key = current_string[-1] if move_idx > 0 else 'q'
        neighbors = graph.get_keys_for_moves_from(start_key=prev_key, num_moves=num_moves[move_idx])

        next_key_probs = dictionary.get_letter_freq(prefix=current_string, total_length=target_length)

        filtered_probs = filter_and_normalize_scores(key_probs=next_key_probs,
                                                     candidate_keys=neighbors)

        for neighbor_key, score in filtered_probs.items():
            candidate_word = '{}{}'.format(current_string, neighbor_key)
            if candidate_word not in visited:
                candidate_queue.append((candidate_word, score * current_score))
                visited.add(candidate_word)

    ranked_results = list(reversed(sorted(scores.items(), key=lambda t: t[1])))
    return ranked_results[0:max_num_results]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary JSON file.')
    parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
    args = parser.parse_args()

    graph = KeyboardGraph()
    dictionary = EnglishDictionary(path=args.dictionary_path)
    results = get_words_from_moves(num_moves=args.moves_list, graph=graph, dictionary=dictionary)
    print(results)


