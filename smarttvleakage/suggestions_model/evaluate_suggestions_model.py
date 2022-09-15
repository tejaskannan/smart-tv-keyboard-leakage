from argparse import ArgumentParser
from typing import Tuple, List, Dict
from os import getpid
import numpy as np
import multiprocessing as mp

from smarttvleakage.utils.constants import SmartTVType, KeyboardType, Direction
from smarttvleakage.utils.file_utils import read_pickle_gz, save_pickle_gz
from smarttvleakage.audio.constants import SAMSUNG_KEY_SELECT
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import restore_dictionary, NgramDictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_suggestions

from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms, classify_ms_with_msfd, classify_ms_with_msfd_full
from smarttvleakage.suggestions_model.manual_score_dict import build_ms_dict, build_msfd
from smarttvleakage.dictionary.rainbow import PasswordRainbow


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
                    suggestions_model, suggestions : int,
                    dictionary, max_num_results : int, msfd = None, db = None,
                    auto_cutoff : int = .5, non_cutoff : int = .5):
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
        if msfd is None:
            use_suggestions = (classify_ms(suggestions_model, move_sequence_vals)[0] == 1)
        else:
            use_suggestions = (classify_ms_with_msfd_full(
            suggestions_model, msfd, db, move_sequence_vals, auto_cutoff=auto_cutoff, non_cutoff=non_cutoff)[0] == 1)


    if use_suggestions:
        ranked_candidates = get_words_from_moves_suggestions(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            did_use_autocomplete=did_use_autocomplete, max_num_results=max_num_results)
    else:
        ranked_candidates = get_words_from_moves(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            tv_type=tv_type,  max_num_results=max_num_results, precomputed=None,
            includes_done=False, start_key="q", is_searching_reverse=False)

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


def worker(input_queue, output_queue, db_path):
    pid = str(getpid())
    db = PasswordRainbow(db_path)
    print(pid + " initialized db")
    while True:
        args = input_queue.get()
        if args == 0:
            break
        if args is None:
            continue
        (s, key, ty, _, val, model, sug,
                                    dictionary, max_num_results, msfd,
                                    ac, nc) = args
        rank, candidates = recover_string(key, val, model, sug, dictionary,
                            max_num_results, msfd, db=db,
                            auto_cutoff=ac, non_cutoff=nc)
        output_queue.put((s, (key, ty), rank, candidates))
        print(pid, end=" ")
        print("did: " + key)

    return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False) #ms_auto_dict pkl.gz
    parser.add_argument("--ms-path-non", type=str, required=False) #ms_non_dict pkl.gz
    parser.add_argument("--ed-path", type=str, required=False) #ed pkl.gz, only build works rn
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    parser.add_argument("--results-path", type=str, required=False)
    parser.add_argument("--text-path-rockyou", type=str, required=False) #rockyou.txt
    parser.add_argument("--ms-path-rockyou", type=str, required=False) #rockyou dict
    parser.add_argument("--ms-path-phpbb", type=str, required=False) #rockyou dict
    parser.add_argument("--msfd-path", type=str, required=False) #msfd pkl.gz
    parser.add_argument("--db-path", type=str, required=False) # rockyou-samsung.db

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
    if args.ms_path_rockyou is None:
        args.ms_path_rockyou = "suggestions_model/local/rockyou_d.pkl.gz"
        rockyouDictionary = read_pickle_gz(args.ms_path_rockyou)
    elif args.ms_path_rockyou == "build":
        args.ms_path_rockyou = "suggestions_model/local/rockyou.txt"
        rockyouDictionary = NgramDictionary()
        rockyouDictionary.build(args.ms_path_rockyou, 0, False)
        save_pickle_gz(rockyouDictionary, "suggestions_model/local/rockyou_d.pkl.gz")
    if args.ms_path_phpbb is None:
        args.ms_path_phpbb = "suggestions_model/local/phpbb_d.pkl.gz"
        phpbbDictionary = read_pickle_gz(args.ms_path_phpbb)
    elif args.ms_path_phpbb == "build":
        args.ms_path_phpbb = "suggestions_model/local/phpbb.txt"
        phpbbDictionary = NgramDictionary()
        phpbbDictionary.build(args.ms_path_phpbb, 0, False)
        save_pickle_gz(phpbbDictionary, "suggestions_model/local/phpbb_d.pkl.gz")
    if args.text_path_rockyou is None:
        args.text_path_rockyou = "suggestions_model/local/ms_dict_rockyou.pkl.gz"

    if args.msfd_path is None:
        args.msfd_path = "suggestions_model/local/msfd.pkl.gz"
        msfd = build_msfd(args.msfd_path)
    elif args.msfd_path == "exp":
        args.msfd_path = "suggestions_model/msfd_exp.pkl.gz"
        msfd = build_msfd(args.msfd_path)

    if args.db_path is None:
        args.db_path = "rockyou-samsung.db"
        db = PasswordRainbow(args.db_path)

    # Tests
    # 0 - english words
    # 1 - rockyou
    # 2 - english words (w/ msfd)
    # 3 - rockyou (w/ msfd)
    # 4 - full test, gt only
    # 5 - full test, parallel
    test = 4


    if test == 5:
        max_num_results = 50
        take_english = 1
        take_rockyou = 1
        take_phpbb = 1
        dictionary = englishDictionary
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto, take_english)
        ms_dict_non_test = build_ms_dict(args.ms_path_non, take_english)
        ms_dict_rockyou = build_ms_dict(args.text_path_rockyou, take_rockyou, 500)
        ms_dict_phpbb = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", take_phpbb)
        print("test dicts built")
        ac = .26
        nc = .32

        input_queue = mp.Queue(maxsize=4)
        output_queue = mp.Queue(maxsize=20)

        ps = {}
        pc = 4
        for i in range(pc):
            ps[i] = mp.Process(target=worker, args=(input_queue, output_queue, args.db_path))
            ps[i].start()
        

        results = {}
        for s in [3, 2]:
            print("testing suggestions: " + suggestion_from_id(s))
            if s == 3:
                sug = (1, 0, 0)
            else:
                sug = (2, 2, 2)
            results[s] = {}

            print("testing autos")
            for key, val in ms_dict_auto_test.items():
                input_queue.put((s, key, "auto", key, val, model, sug[0],
                                    dictionary, max_num_results, msfd,
                                    ac, nc), block=True, timeout=None)
            print("testing nons")
            for key, val in ms_dict_non_test.items():
                input_queue.put((s, key, "non", key, val, model, sug[1],
                                    dictionary, max_num_results, msfd,
                                    ac, nc), block=True, timeout=None)
            print("testing rockyous")
            for key, val in ms_dict_rockyou.items():
                input_queue.put((s, key, "rockyou", key, val, model, sug[2],
                                    rockyouDictionary, max_num_results, msfd,
                                    ac, nc), block=True, timeout=None)
            print("testing phpbbs")
            for key, val in ms_dict_phpbb.items():
                if key.lower() in ["vqsablpzla", "gznybxyj"]:
                    continue
                #print(key)
                input_queue.put((s, key, "phpbb", key, val, model, sug[2],
                                    phpbbDictionary, max_num_results, msfd,
                                    ac, nc), block=True, timeout=None)
        print("queue completed")
        for i in range(pc):
            input_queue.put(0)
        for i in range(pc):
            ps[i].join()
        print("joined final")
        while True:
            output = output_queue.get()
            if output is None:
                break
            results[output[0]][output[1]] = output[2:]


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
            lines[s].append(f"Words not found: {list(map(lambda x : x[0] + ' (' + x[1] + ')', not_found))}\n")
            lines[s].append("\n")

            for l in lines[s]:
                print(l, end="")

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("ac: " + str(ac) + ", nc: " + str(nc) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("words tested: " + str(take_english*2 + take_rockyou + take_phpbb) + "\n\n")
                all_lines += [l for ls in lines.values() for l in ls]
                f.writelines(all_lines)
                f.close()




    if test == 4:
        max_num_results = 50
        take_english = 1
        take_rockyou = 1
        take_phpbb = 1
        dictionary = englishDictionary
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto, take_english)
        ms_dict_non_test = build_ms_dict(args.ms_path_non, take_english)
        ms_dict_rockyou = build_ms_dict(args.text_path_rockyou, take_rockyou, 500)
        ms_dict_phpbb = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", take_phpbb)
        print("test dicts built")
        ac = .26
        nc = .32

        # ranks, candidates
        # "vqsablpzla"
        results = {}
        for s in [3, 2]:
            print("testing suggestions: " + suggestion_from_id(s))
            if s == 3:
                sug = (1, 0, 0)
            else:
                sug = (2, 2, 2)
            results[s] = {}

            print("testing autos")
            for key, val in ms_dict_auto_test.items():
                rank, candidates = recover_string(key, val, model, sug[0],
                                    dictionary, max_num_results, msfd, db=db,
                                    auto_cutoff=ac, non_cutoff=nc)
                results[s][(key, "auto")] = (rank, candidates)
            print("testing nons")
            for key, val in ms_dict_non_test.items():
                rank, candidates = recover_string(key, val, model, sug[1],
                                    dictionary, max_num_results, msfd, db=db,
                                    auto_cutoff=ac, non_cutoff=nc)
                results[s][(key, "non")] = (rank, candidates)
            print("testing rockyous")
            for key, val in ms_dict_rockyou.items():
                rank, candidates = recover_string(key, val, model, sug[2],
                                    rockyouDictionary, max_num_results, msfd, db=db,
                                    auto_cutoff=ac, non_cutoff=nc)
                results[s][(key, "rockyou")] = (rank, candidates)
            print("testing phpbbs")
            for key, val in ms_dict_phpbb.items():
                if key.lower() in ["vqsablpzla", "gznybxyj"]:
                    continue
                print(key)
                rank, candidates = recover_string(key, val, model, sug[2],
                                    phpbbDictionary, max_num_results, msfd, db=db,
                                    auto_cutoff=ac, non_cutoff=nc)
                results[s][(key, "phpbb")] = (rank, candidates)

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
            lines[s].append(f"Words not found: {list(map(lambda x : x[0] + ' (' + x[1] + ')', not_found))}\n")
            lines[s].append("\n")

            for l in lines[s]:
                print(l, end="")

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("ac: " + str(ac) + ", nc: " + str(nc) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("words tested: " + str(take_english*2 + take_rockyou + take_phpbb) + "\n\n")
                all_lines += [l for ls in lines.values() for l in ls]
                f.writelines(all_lines)
                f.close()


    if test == 3:
        max_num_results = 100
        dictionary = rockyouDictionary
        print("building test dicts")
        ms_dict_rockyou = build_ms_dict(args.text_path_rockyou, 2000, 500)
        print("test dicts built")
        print(len(ms_dict_rockyou.keys()))
        ac = .26
        nc = .32

        # ranks, candidates
        results = {}
        for s in [0, 2]:
            print("testing suggestions: " + suggestion_from_id(s))
            results[s] = {}

            print("testing rockyous")
            for key, val in ms_dict_rockyou.items():
                rank, candidates = recover_string(
                    key, val, model, s, dictionary, max_num_results,
                    msfd, auto_cutoff=ac, non_cutoff=nc)
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
            lines[s].append(f"Words not found: {list(map(lambda x : x[0] + ' (' + x[1] + ')', not_found))}\n")
            lines[s].append("\n")

            for l in lines[s]:
                print(l, end="")

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("ac: " + str(ac) + ", nc: " + str(nc) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("words tested: " + str(100) + "\n\n")
                all_lines += [l for ls in lines.values() for l in ls]
                f.writelines(all_lines)
                f.close()


    if test == 2:
        max_num_results = 200
        take = 0
        dictionary = englishDictionary
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto, take)
        ms_dict_non_test = build_ms_dict(args.ms_path_non, take)
        print("test dicts built")
        ac = .26
        nc = .32

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
                rank, candidates = recover_string(key, val, model, sug[0],
                                    dictionary, max_num_results, msfd,
                                    auto_cutoff=ac, non_cutoff=nc)
                results[s][(key, "auto")] = (rank, candidates)
            print("testing nons")
            for key, val in ms_dict_non_test.items():
                rank, candidates = recover_string(key, val, model, sug[1],
                                    dictionary, max_num_results, msfd,
                                    auto_cutoff=ac, non_cutoff=nc)
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
            lines[s].append(f"Words not found: {list(map(lambda x : x[0] + ' (' + x[1] + ')', not_found))}\n")
            lines[s].append("\n")

            for l in lines[s]:
                print(l, end="")

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("ac: " + str(ac) + ", nc: " + str(nc) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("words tested: " + str(105) + "\n\n")
                all_lines += [l for ls in lines.values() for l in ls]
                f.writelines(all_lines)
                f.close()


    if test == 1:
        max_num_results = 50
        dictionary = rockyouDictionary
        print("building test dicts")
        ms_dict_rockyou = build_ms_dict(args.text_path_rockyou, 600)
        print("test dicts built")

        # ranks, candidates
        results = {}
        for s in [0, 2]:
            print("testing suggestions: " + suggestion_from_id(s))
            results[s] = {}

            print("testing rockyous")
            psvr = 0
            for key, val in ms_dict_rockyou.items():
                if psvr < 500:
                    psvr += 1
                    continue
                rank, candidates = recover_string(key, val, model, s, dictionary, max_num_results)
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
            lines[s].append(f"Words not found: {list(map(lambda x : x[0] + ' (' + x[1] + ')', not_found))}\n")
            lines[s].append("\n")

            for l in lines[s]:
                print(l, end="")

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("words tested: " + str(100) + "\n\n")
                all_lines += [l for ls in lines.values() for l in ls]
                f.writelines(all_lines)
                f.close()



    if test == 0:
        max_num_results = 200
        take = 105
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
            lines[s].append(f"Words not found: {list(map(lambda x : x[0] + ' (' + x[1] + ')', not_found))}\n")
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
