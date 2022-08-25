from argparse import ArgumentParser
from typing import Tuple, List, Dict
import numpy as np

from smarttvleakage.utils.constants import SmartTVType, KeyboardType, Direction
from smarttvleakage.utils.file_utils import read_pickle_gz
from smarttvleakage.audio.constants import SAMSUNG_KEY_SELECT
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import restore_dictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_suggestions

from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms
from smarttvleakage.suggestions_model.manual_score_dict import build_ms_dict


def suggestion_from_id(i : int) -> str:
    """Translates a suggestion strategy ID into that suggestion"""
    if i == 0:
        return "assume standard"
    if i == 1:
        return "assume suggestions"
    if i == 2:
        return "predict"
    return "gt"

def recover_string(true_word : str, ms : List[int],
                    suggestions_model, suggestions : int, dictionary, max_num_results : int):
    """Attempts to recover a string, returns the rank and # candidates"""
    tv_type = SmartTVType.SAMSUNG
    did_use_autocomplete = False
    keyboard_type = KeyboardType.SAMSUNG
    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    dictionary.set_characters(graph.get_characters())

    move_sequence_vals = ms
    move_sequence = [Move(num_moves=num_moves,
                    end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in ms]

    move_sequence_vals = list(map(lambda m: m.num_moves, move_sequence))
    if suggestions == 0:
        use_suggestions = False
    elif suggestions == 1:
        use_suggestions = True
    elif suggestions == 2:
        use_suggestions = (classify_ms(suggestions_model, move_sequence_vals) == 1)

    if use_suggestions:
        ranked_candidates = get_words_from_moves_suggestions(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            did_use_autocomplete=did_use_autocomplete, max_num_results=max_num_results)
    else:
        ranked_candidates = get_words_from_moves(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            tv_type=tv_type,  max_num_results=max_num_results, precomputed=None)

    for rank, (guess, _, num_candidates) in enumerate(ranked_candidates):
        #print(rank, num_candidates)
        #print("Guess: {}, Score: {}".format(guess, score))
        if guess == true_word:
            return (rank+1, num_candidates)
    return (-1, -1)

def eval_results(rd : Dict[Tuple[str, str], Tuple[int, int]]) -> Tuple[
    List[Tuple[str, str, int, int]], List[Tuple[str, str, int, int]], List[Tuple[str, str]], int]:
    """Translates result dictionary into (found, top10, not found, total)"""
    found = []
    not_found = []
    top10 = []
    for (word, ty), (rank, cand) in rd.items():
        if rank > 0:
            found.append((word, ty, rank, cand))
            if rank <= 10:
                top10.append((word, ty, rank, cand))
        else:
            not_found.append((word, ty))

    return (found, top10, not_found, len(rd.items()))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False) #ms_auto_dict pkl.gz
    parser.add_argument("--ms-path-non", type=str, required=False) #ms_non_dict pkl.gz
    parser.add_argument("--ed-path", type=str, required=False) #ed pkl.gz, only build works rn
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    parser.add_argument("--results-path", type=str, required=False)
    args = parser.parse_args()

    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
    if args.ed_path is None:
        englishDictionary = restore_dictionary(
            "suggestions_model/local/dictionaries/ed.pkl.gz")
    if args.model_path is None:
        args.model_path = "suggestions_model/model_sim.pkl.gz"
    model = read_pickle_gz(args.model_path)
    # Tests

    max_num_results = 10
    take = 3
    dictionary = englishDictionary
    print("building test dicts")
    ms_dict_auto_test = build_ms_dict(args.ms_path_auto, take)
    ms_dict_non_test = build_ms_dict(args.ms_path_non, take)
    print("test dicts built")

    # ranks, candidates
    results = {}
    for s in [0, 1, 2, 3]:
        print("testing suggestions: " + suggestion_from_id(s))
        if s in [0, 1, 2]:
            sug = (s, s)
        else:
            sug = (1, 0)
        results[s] = {}

        print("testing autos")
        for key, val in ms_dict_auto_test.items():
            rank, candidates = recover_string(key, val, model, sug[0], dictionary, max_num_results)
            results[s][(key, "auto")] = (rank, candidates)
        print("testing nons")
        for key, val in ms_dict_non_test.items():
            rank, candidates = recover_string(key, val, model, sug[1], dictionary, max_num_results)
            results[s][(key, "non")] = (rank, candidates)

    lines = {}
    for s, rd in results.items():
        found, top10, not_found, total = eval_results(rd)
        rank_list = list(map((lambda x : x[2]), found))
        rank_list_adjusted = rank_list + [2*max_num_results for i in not_found]
        candidates_list = list(map((lambda x : x[3]), found))

        lines[s] = ["suggestions: " + suggestion_from_id(s) + "\n"]
        lines[s].append(
            f"{len(found)} found, Accuracy: ({len(found)/total})\n")
        lines[s].append(
            f"{len(top10)} top 10, Accuracy: ({len(top10) / total})\n")
        lines[s].append(
            f"Avg Rank Found: {np.average(rank_list),}, Median Rank Found: {np.median(rank_list)}\n")
        lines[s].append(
            f"Avg Rank Adjusted: {np.average(rank_list_adjusted)}, Median Rank Adjusted: {np.median(rank_list_adjusted)}\n")
        lines[s].append(
            f"Avg # Candidates Found: {np.average(candidates_list)}, Median # Candidates Found: {np.median(candidates_list)}\n")
        lines[s].append(f"Words not found: {list(map(lambda x : x[0], not_found))}\n")
        lines[s].append("\n")

        for l in lines[s]:
            print(l, end="")

    if args.results_path is not None:
        with open(args.results_path, "a", encoding="utf-8") as f:
            all_lines = []
            all_lines.append("max_num_results: " + str(max_num_results) + "\n")
            all_lines.append("words tested: " + str(take*2) + "\n\n")
            all_lines += [l for ls in lines.values() for l in ls]
            f.writelines(all_lines)
            f.close()
