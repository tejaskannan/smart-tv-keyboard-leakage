from traceback import print_last
from typing import List, Dict, Tuple
from argparse import ArgumentParser
import random

from smarttvleakage.suggestions_model.manual_score_dict import (build_msfd,
                                                            build_ms_dict)
from smarttvleakage.utils.file_utils import read_pickle_gz
from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms_with_msfd_full


from smarttvleakage.audio.sounds import SAMSUNG_KEY_SELECT
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_suggestions

from smarttvleakage.utils.constants import SmartTVType, KeyboardType, Direction

from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms_with_msfd_full

from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.dictionary import EnglishDictionary




def build_combo_dict(ms_dict_auto, ms_dict_non,
            count : int) -> Dict[Tuple[str, str], Tuple[List[int], List[int], int]]:
    """Builds a combined suggestions/non-suggestions ms dict"""

    print("building combo dict")
    random.seed(1)
    ms_dict_combo = {}
    for _ in range(count):
        auto_i = random.randint(0, len(ms_dict_auto.items())-1)
        non_i = random.randint(0, len(ms_dict_non.items())-1)
        auto_pair = list(ms_dict_auto.items())[auto_i]
        non_pair = list(ms_dict_non.items())[non_i]
        done_m = random.randint(0, 1)
        ms_dict_combo[(auto_pair[0], non_pair[0])] = (auto_pair[1], non_pair[1], done_m)
    print("combo dict built")
    return ms_dict_combo

def apply_heuristics_exp(ms_auto, ms_non,) -> Tuple[int, int, int, int]:
    """Applies confidences multipliers based on heuristics"""
    heur = []

    mult = 0
    for i, _ in enumerate(ms_non[1:]):
        if ms_non[i-1:i+1] == [0, 0]:
            mult += 1
    heur.append(mult)

    if ms_non[0] == 0:
        heur.append(1)
    else:
        heur.append(0)

    if ms_non[0] == 1:
        heur.append(1)
    else:
        heur.append(0)

    if len(ms_non) > 1:
        if (ms_non[0] == 0) and (ms_non[1] != 6):
            heur.append(1)
        else:
            heur.append(0)
    else:
        heur.append(0)

    return (heur[0], heur[1], heur[2], heur[3])


def apply_heuristics(ms_auto, ms_non,
                    mults : Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Applies confidences multipliers based on heuristics"""
    auto_mult = 1.0
    non_mult = 1.0

    for i, _ in enumerate(ms_non[1:]):
        if ms_non[i-1:i+1] == [0, 0]:
            non_mult *= mults[0]

    if ms_non[0] == 0:
        non_mult *= mults[1]

    if ms_non[0] == 1:
        non_mult *= mults[2]

    if len(ms_non) > 1:
        if (ms_non[0] == 0) and (ms_non[1] != 6):
            non_mult *= mults[3]

    return auto_mult, non_mult


def get_split_scores(model, msfd, db,
                ms_auto : List[int], ms_non : List[int],
                msg : bool = False) -> Tuple[float, float]:
    """Returns two confidences for a given auto/non split"""
    ac = .32
    nc = .26
    clas, _, _, conf = classify_ms_with_msfd_full(model, msfd, db, ms_auto,
                                            auto_cutoff=ac, non_cutoff=nc)
    if clas == 1:
        conf_auto = conf
    else:
        conf_auto = 1-conf
    clas, _, _, conf = classify_ms_with_msfd_full(model, msfd, db, ms_non,
                                            auto_cutoff=ac, non_cutoff=nc)
    if clas == 0:
        conf_non = conf
    else:
        conf_non = 1-conf

    if msg:
        print(str(len(ms_auto)) + "/" + str(len(ms_non)))
        print("conf_auto: " + str(conf_auto))
        print("conf_non: " + str(conf_non))
    return (conf_auto, conf_non)


# Write this!
# use apply heuristics exp too, which wil...
def evaluate_split_exp(model, msfd, db,
                ms_auto : List[int], ms_non : List[int],
                strategy : int,
                msg : bool = False,
                dictionary = None) -> Tuple[float, float, float, float,
                Tuple[int, int, int, int]]:
    """Evaluates a given auto/non ms split"""

    conf_auto, conf_non = get_split_scores(model, msfd, db, ms_auto, ms_non, msg=msg)
    # implement strategy and heurs

    mults = apply_heuristics_exp(ms_auto, ms_non)

    nrs = recovery_score(ms_non, model, 0, dictionary, msfd, db,
        auto_cutoff=.26, non_cutoff=.32)
    if nrs is None:
        nrs = 1/40
    ars = recovery_score(ms_auto, model, 1, dictionary, msfd, db,
        auto_cutoff=.26, non_cutoff=.32)
    if ars is None:
        ars = 1/40

    return (conf_auto, conf_non, ars, nrs, mults)




def evaluate_split(model, msfd, db,
                ms_auto : List[int], ms_non : List[int],
                strategy : int,
                mults : Tuple[float, float, float, float] = (1, 1, 1, 1),
                msg : bool = False,
                use_recovery_score : int = 0,
                dictionary = None) -> Tuple[float, float, float]:
    """Evaluates a given auto/non ms split"""

    conf_auto, conf_non = get_split_scores(model, msfd, db, ms_auto, ms_non, msg=msg)
    # implement strategy and heurs

    auto_mult, non_mult = apply_heuristics(ms_auto, ms_non, mults)
    conf_auto *= auto_mult
    conf_non *= non_mult

    if use_recovery_score > 0:
        nrs = recovery_score(ms_non, model, 0, dictionary, msfd, db,
            auto_cutoff=.26, non_cutoff=.32)
        ars = recovery_score(ms_auto, model, 1, dictionary, msfd, db,
            auto_cutoff=.26, non_cutoff=.32)
        
        
        #print("len of auto: " + str(len(ms_auto)))
        
        if use_recovery_score > 1:
            if ars is None:
                conf_auto *= (1/40)
            else:
                conf_auto *= ars
                #print("ars: " + str(40*ars))
        if nrs is None:
            conf_non *= (1/40)
        else:
            conf_non *= nrs
            #print("nrs: " + str(40*nrs))


    if strategy == 0:
        score = (conf_auto + conf_non)/2
    else:
        score = (pow(conf_auto, 2) + pow(conf_non, 2)) / 2

    return (score, conf_auto, conf_non)

# have to redo evaluate_split as well
def evaluate_all_splits_exp(model, msfd, db, ms : List[int],
            strategy : int,
            dictionary = None
            ) -> List[Tuple[int, Tuple[float, float, float, float,
                Tuple[int, int, int, int]]]]:

    """returns a ranked list of split evaluations"""
    split_scores = {}
    for i, m in enumerate(ms):
        #print("i: " + str(i))
        if i in [0, len(ms), len(ms)-1]:
            continue
        if m not in [0, 1]:
            continue
        ms_auto = ms[:i]
        ms_non = ms[i+1:]
        split_scores[i] = evaluate_split_exp(
            model, msfd, db, ms_auto, ms_non, strategy=strategy, msg=False, dictionary = dictionary)

    
    return list(split_scores.items())


def rank_splits(all_splits, mults, use_recovery_score):
    split_scores = {}

    for split, data in all_splits:
        base_auto, base_non, rs_auto, rs_non, heurs = data

        score_auto = base_auto
        score_non = base_non
        if use_recovery_score > 0:
            score_non *= rs_non
        if use_recovery_score > 1:
            score_auto *= rs_auto

        score = (score_auto + score_non)/2

        for i in range(len(mults)):
            if heurs[i] > 0:
                score *= mults[i]
        
        split_scores[split] = score

    ranked_splits = list(split_scores.items())
    ranked_splits.sort(key=lambda x : x[1], reverse=True)
    return ranked_splits
        


    



def evaluate_all_splits(model, msfd, db, ms : List[int],
            strategy : int,
            mults : Tuple[float, float, float, float] = (1, 1, 1, 1),
            use_recovery_score : int = 0,
            dictionary = None
            ) -> List[Tuple[int, float]]:
    """returns a ranked list of split evaluations"""
    split_scores = {}
    for i, m in enumerate(ms):
        #print("i: " + str(i))
        if i in [0, len(ms), len(ms)-1]:
            continue
        if m not in [0, 1]:
            continue
        ms_auto = ms[:i]
        ms_non = ms[i+1:]
        split_scores[i] = evaluate_split(
            model, msfd, db, ms_auto, ms_non, strategy=strategy, msg=False, mults=mults,
            use_recovery_score=use_recovery_score, dictionary = dictionary)

    ranked_splits = list(split_scores.items())
    ranked_splits.sort(key=lambda x : x[1][0], reverse=True)
    return ranked_splits
    #return ms[:ranked_splits[0][0]], ms[ranked_splits[0][0]:]
    # zip the return with each score?



def recovery_score(ms : List[int],
                    suggestions_model, use_suggestions,
                    dictionary, msfd = None, db = None,
                    auto_cutoff : int = .5, non_cutoff : int = .5, peak : int = 30):
    """Gets a score (for the split) by the score of the first recovered string"""

    tv_type = SmartTVType.SAMSUNG
    did_use_autocomplete = False
    keyboard_type = KeyboardType.SAMSUNG
    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    dictionary.set_characters(graph.get_characters())

    move_sequence_vals = ms
    move_sequence = [Move(num_moves=num_moves,
                    end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in ms]

    move_sequence_vals = list(map(lambda m: m.num_moves, move_sequence))


    if use_suggestions:
        ranked_candidates = get_words_from_moves_suggestions(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            did_use_autocomplete=did_use_autocomplete, max_num_results=1)
    else:
        ranked_candidates = get_words_from_moves(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            tv_type=tv_type,  max_num_results=1, precomputed=None,
            includes_done=False, start_key="q", is_searching_reverse=False)

    for _, (_, score, _) in enumerate(ranked_candidates): 
        return (40 - score)/40



def recover_string(true_word : str, ms : List[int],
                    suggestions_model, use_suggestions,
                    dictionary, max_num_results : int, msfd = None, db = None,
                    auto_cutoff : int = .5, non_cutoff : int = .5, peak : int = 30):
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


def recover_combo(auto_word : str, non_word : str, ms : Tuple[List[int], List[int], int],
                suggestions_model, dictionary, max_num_results : int, msfd = None, db = None,
                auto_cutoff : int = .5, non_cutoff : int = .5, peak : int = 30):

    rank_auto, nc_auto = recover_string(auto_word, ms[0], suggestions_model, 1,
        dictionary, max_num_results, msfd, db, auto_cutoff, non_cutoff, peak)

    rank_non, nc_non = recover_string(non_word, ms[1], suggestions_model, 0,
        dictionary, max_num_results, msfd, db, auto_cutoff, non_cutoff, peak)    

    return (rank_auto, nc_auto, rank_non, nc_non)

# KEEP GOING W THIS, test it 
















if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False) #ms_auto_dict pkl.gz
    parser.add_argument("--ms-path-non", type=str, required=False) #ms_non_dict pkl.gz
    parser.add_argument("--msfd-path", type=str, required=False) #msfd pkl.gz
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    parser.add_argument("--ed-path", type=str, required=False) #ed pkl.gz, only build works rn
    parser.add_argument("--results-path", type=str, required=False)
    args = parser.parse_args()

    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
        ms_dict_auto = build_ms_dict(args.ms_path_auto)

    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_non)
    elif args.ms_path_non == "phpbb":
        args.ms_path_non = "suggestions_model/local/ms_dict_phpbb.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_non, 100, 500)

    if args.msfd_path is None:
        args.msfd_path = "suggestions_model/local/msfd.pkl.gz"
        msfd = build_msfd(args.msfd_path)
    elif args.msfd_path == "exp":
        args.msfd_path = "suggestions_model/msfd_exp.pkl.gz"
        msfd = build_msfd(args.msfd_path)

    if args.model_path is None:
        args.model_path = "suggestions_model/model_sim.pkl.gz"
    model = read_pickle_gz(args.model_path)

    if args.ed_path is None:
        englishDictionary = EnglishDictionary.restore(
            "suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "build":
        englishDictionary = EnglishDictionary(50)
        englishDictionary.build(
            "suggestions_model/local/dictionaries/enwiki-20210820-words-frequency.txt", 50, True, False)
        englishDictionary.save("suggestions_model/local/dictionaries/ed.pkl.gz")
    # TESTS
    # 0 - accuracy test
    # 2 - analysis test
    # 3 - heurs test
    # 4 - ranking test
    # 5 - 3+4
    # 6 - test recovery
    # 7 - test using recover yscore

    # 8 - test recovery w score strat

    test = 9

    # use exp to speed up
    if test == 9:
        print("test 9")
       
        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        count = 100
        max_num_results = 20
        mults = (.9, .8, .8, .8)

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, count)

        wrong_gt = (0, 0)
        score_dict = {}
        recovery_score_strats = [0, 1, 2]
        for i in recovery_score_strats:
            score_dict[i] = {}
        
        print("testing")
        done = 0
        for key, val in ms_dict_combo.items():
            auto_word = key[0]
            non_word = key[1]
            print("aw: " + auto_word + ", nw: " + non_word)
            if (done % 4 == 0):
                print("done: " + str(done))
            done += 1

            
            ms = val[0] + [val[2]] + val[1]
            ranks = {}
            unranked_splits = evaluate_all_splits_exp(model, msfd, db, ms, 0, englishDictionary)
            print("got unr splits")

            for rss in recovery_score_strats:
                print("rss: " + str(rss))
                splits = rank_splits(unranked_splits, mults, rss)

                for i, ss in enumerate(splits):
                    split, _ = ss
                    if ms[:split] == val[0]:
                        ranks[rss] = i+1
                        if ranks[rss] in score_dict[rss]:
                            score_dict[rss][ranks[rss]] =(
                                score_dict[rss][ranks[rss]][0]+1, score_dict[rss][ranks[rss]][1],
                                score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3],
                                score_dict[rss][ranks[rss]][4]) 
                        else:
                            score_dict[rss][ranks[rss]] = (1, 0, 0, 0, 0)
                        break

            
            print("testing recovery")

            rank_auto, nc_auto, rank_non, nc_non = recover_combo(auto_word, non_word,
                val, model, englishDictionary, max_num_results, msfd, db, auto_cutoff=.26, non_cutoff=.32)

            if (rank_auto >= 0):
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1]+1,
                            score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3],
                            score_dict[rss][ranks[rss]][4]) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 1, 0, 0, 0)
            else:
                wrong_gt = (wrong_gt[0]+1, wrong_gt[1])
                
            if (rank_non >= 0):
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1],
                            score_dict[rss][ranks[rss]][2]+1, score_dict[rss][ranks[rss]][3],
                            score_dict[rss][ranks[rss]][4]) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 0, 1, 0, 0)
            else:
                wrong_gt = (wrong_gt[0], wrong_gt[1]+1)

            if (rank_non >= 0 and rank_auto >= 0):
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1],
                            score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3]+1,
                            score_dict[rss][ranks[rss]][4]) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 0, 0, 1, 0)

            else:
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1],
                            score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3],
                            score_dict[rss][ranks[rss]][4]+1) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 0, 0, 0, 1)


        print("Unrecovered Suggestions: " + str(wrong_gt[0]))
        print("Unrecovered Nons: " + str(wrong_gt[1]))
        for rss in recovery_score_strats:
            score_dict_rss = score_dict[rss]
            print("recovery score strat: " + str(rss))

            print("all ranks:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[0]))
            print("\nSuggestions Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[1]))
            print("\nNon Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[2]))
            print("\nBoth Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[3]))
            print("\nNeither Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[4]))



        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("count: " + str(count) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("Unrecovered Suggestions: " + str(wrong_gt[0]) + "\n")
                all_lines.append("Unrecovered Nons: " + str(wrong_gt[1]) + "\n")
                
                for rss in recovery_score_strats:
                    score_dict_rss = score_dict[rss]
                    all_lines.append("\nrecovery score strat: " + str(rss) + "\n")

                    all_lines.append("\nall ranks:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[0]) + "\n")
                    all_lines.append("\nSuggestions Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[1]) + "\n")
                    all_lines.append("\nNon Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[2]) + "\n")
                    all_lines.append("\nBoth Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[3]) + "\n")
                    all_lines.append("\nNeither Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[4]) + "\n")

                
                f.writelines(all_lines)
                f.close()
            





    if test == 8:
        print("test 8")
       
        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        count = 20
        max_num_results = 30

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, count)

        wrong_gt = (0, 0)
        score_dict = {}
        recovery_score_strats = [0, 1, 2]
        for i in recovery_score_strats:
            score_dict[i] = {}
        
        print("testing")
        done = 0
        for key, val in ms_dict_combo.items():
            auto_word = key[0]
            non_word = key[1]
            print("aw: " + auto_word + ", nw: " + non_word)
            if (done % 4 == 0):
                print("done: " + str(done))
            done += 1

            
            ms = val[0] + [val[2]] + val[1]
            ranks = {}
            for rss in recovery_score_strats:
                splits = evaluate_all_splits(model, msfd, db, ms, 0, (.9, .8, .8, .8),
                    use_recovery_score=True, dictionary=englishDictionary)

                for i, ss in enumerate(splits):
                    split, _ = ss
                    if ms[:split] == val[0]:
                        ranks[rss] = i+1
                        if ranks[rss] in score_dict[rss]:
                            score_dict[rss][ranks[rss]] =(
                                score_dict[rss][ranks[rss]][0]+1, score_dict[rss][ranks[rss]][1],
                                score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3],
                                score_dict[rss][ranks[rss]][4]) 
                        else:
                            score_dict[rss][ranks[rss]] = (1, 0, 0, 0, 0)
                        break

            
            print("testing recovery")

            rank_auto, nc_auto, rank_non, nc_non = recover_combo(auto_word, non_word,
                val, model, englishDictionary, max_num_results, msfd, db, auto_cutoff=.26, non_cutoff=.32)

            if (rank_auto >= 0):
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1]+1,
                            score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3],
                            score_dict[rss][ranks[rss]][4]) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 1, 0, 0, 0)
            else:
                wrong_gt = (wrong_gt[0]+1, wrong_gt[1])
                
            if (rank_non >= 0):
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1],
                            score_dict[rss][ranks[rss]][2]+1, score_dict[rss][ranks[rss]][3],
                            score_dict[rss][ranks[rss]][4]) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 0, 1, 0, 0)
            else:
                wrong_gt = (wrong_gt[0], wrong_gt[1]+1)

            if (rank_non >= 0 and rank_auto >= 0):
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1],
                            score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3]+1,
                            score_dict[rss][ranks[rss]][4]) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 0, 0, 1, 0)

            else:
                for rss in recovery_score_strats:
                    if ranks[rss] in score_dict[rss]:
                        score_dict[rss][ranks[rss]] =(
                            score_dict[rss][ranks[rss]][0], score_dict[rss][ranks[rss]][1],
                            score_dict[rss][ranks[rss]][2], score_dict[rss][ranks[rss]][3],
                            score_dict[rss][ranks[rss]][4]+1) 
                    else:
                        score_dict[rss][ranks[rss]] = (0, 0, 0, 0, 1)


        print("Unrecovered Suggestions: " + str(wrong_gt[0]))
        print("Unrecovered Nons: " + str(wrong_gt[1]))
        for rss in recovery_score_strats:
            score_dict_rss = score_dict[rss]
            print("recovery score strat: " + str(rss))

            print("all ranks:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[0]))
            print("\nSuggestions Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[1]))
            print("\nNon Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[2]))
            print("\nBoth Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[3]))
            print("\nNeither Recovered:")
            for rank, scores in score_dict_rss.items():
                print(str(rank) + ": " + str(scores[4]))



        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("count: " + str(count) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("Unrecovered Suggestions: " + str(wrong_gt[0]) + "\n")
                all_lines.append("Unrecovered Nons: " + str(wrong_gt[1]) + "\n")
                
                for rss in recovery_score_strats:
                    score_dict_rss = score_dict[rss]
                    all_lines.append("\nrecovery score strat: " + str(rss) + "\n")

                    all_lines.append("\nall ranks:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[0]) + "\n")
                    all_lines.append("\nSuggestions Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[1]) + "\n")
                    all_lines.append("\nNon Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[2]) + "\n")
                    all_lines.append("\nBoth Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[3]) + "\n")
                    all_lines.append("\nNeither Recovered:\n")
                    for rank, scores in score_dict_rss.items():
                        all_lines.append(str(rank) + ": " + str(scores[4]) + "\n")

                
                f.writelines(all_lines)
                f.close()
            







    if test == 7:
        print("test 7")


        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        count = 50
        max_num_results = 100

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, count)


        for key, val in ms_dict_combo.items():
            auto_word = key[0]
            non_word = key[1]
            print("aw: " + auto_word + ", nw: " + non_word)

            
            ms = val[0] + [val[2]] + val[1]
            splits = evaluate_all_splits(model, msfd, db, ms, 0, (.8, .6, .6, .6),
                use_recovery_score=True, dictionary=englishDictionary)



    #correct_gt: 74
    #correct_model: 74
    #/100 !!  
    # False Results

    # Add check for 2nd best, #3rd best result?
    # git pull and check


    if test == 6:
        print("test 6")
       
        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        count = 500
        max_num_results = 100

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, count)

        wrong_gt = 0
        score_dict = {}

        print("testing")
        for key, val in ms_dict_combo.items():
            auto_word = key[0]
            non_word = key[1]
            print("aw: " + auto_word + ", nw: " + non_word)

            
            ms = val[0] + [val[2]] + val[1]
            splits = evaluate_all_splits(model, msfd, db, ms, 0, (.8, .6, .6, .6))

            for i, ss in enumerate(splits):
                split, _ = ss
                if ms[:split] == val[0]:
                    rank = i+1
                    if rank in score_dict:
                        score_dict[rank] = (score_dict[rank][0]+1, score_dict[rank][1], score_dict[rank][2]) 
                    else:
                        score_dict[rank] = (1, 0, 0)
                    break

            
            print("testing recovery")

            rank_auto, nc_auto, rank_non, nc_non = recover_combo(auto_word, non_word,
                val, model, englishDictionary, max_num_results, msfd, db, auto_cutoff=.26, non_cutoff=.32)

            if (rank_auto < 0) or (rank_non < 0):
                wrong_gt += 1
                if rank in score_dict:
                    score_dict[rank] = (score_dict[rank][0], score_dict[rank][1], score_dict[rank][2]+1) 
                else:
                    score_dict[rank] = (0, 0, 1)
            else:
                if rank in score_dict:
                    score_dict[rank] = (score_dict[rank][0], score_dict[rank][1]+1, score_dict[rank][2]) 
                else:
                    score_dict[rank] = (0, 1, 0)
           

        print("general ranks:")
        for rank, scores in score_dict.items():
            print(str(rank) + ": " + str(scores[0]))

        print("\nrelevant ranks:")
        for rank, scores in score_dict.items():
            print(str(rank) + ": " + str(scores[1]))


        print("\nwrong_gt: " + str(wrong_gt))

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                all_lines = []
                all_lines.append("count: " + str(count) + "\n")
                all_lines.append("max_num_results: " + str(max_num_results) + "\n")
                all_lines.append("\ngeneral ranks:\n")
                for rank, scores in score_dict.items():
                    all_lines.append(str(rank) + ": " + str(scores[0]) + "\n")
                all_lines.append("\nrelevant ranks:\n")
                for rank, scores in score_dict.items():
                    all_lines.append(str(rank) + ": " + str(scores[1]) + "\n")
                all_lines.append("\nirrelevant ranks:\n")
                for rank, scores in score_dict.items():
                    all_lines.append(str(rank) + ": " + str(scores[2]) + "\n")

                all_lines.append("\nwrong_gt: " + str(wrong_gt))
                f.writelines(all_lines)
                f.close()
            



    if test == 5:
        print("test 5")

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, 200)
        mults = [[.6, .8, 1], [.4, .6], [.4, .6], [.6, .9]]
        results = {}

        for i in mults[0]:
            for j in mults[1]:
                for k in mults[2]:
                    for l in mults[3]:
                        hs = (i, j, k, l)
                        results[hs] = [0, 0, 0, 0]
                        print("testing", end=": ")
                        print(hs)

                        for keys, vals in ms_dict_combo.items():
                            ms = vals[0] + [vals[2]] + vals[1]
                            guesses = evaluate_all_splits(model, msfd, ms, strategy=0, mults=hs)
                            if guesses[0][0] == len(keys[0]):
                                results[hs][0] += 1
                            elif len(guesses) > 0 and guesses[1][0] == len(keys[0]):
                                results[hs][1] += 1
                            elif len(guesses) > 1 and guesses[2][0] == len(keys[0]):
                                results[hs][2] += 1
                            else:
                                results[hs][3] += 1

                        print(results[hs])

        res = list(results.items())
        res_weighted = list(map(lambda x : (x[0], x[1][0] + .5*x[1][1] + .25*x[1][2]), res))
        res_weighted.sort(key = lambda x : x[1])
        for hs, correct in res_weighted:
            print(hs, end=": ")
            print(correct, end="; weighted acc: ")
            print(correct / len(list(ms_dict_combo.items())))




    if test == 4:
        print("test 4")

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, 1000)
        hs = (.85, .5, .6, .7)
        results = {}
        for i in range(10):
            results[i] = 0
        print("testing", end=": ")
        print(hs)

        for keys, vals in ms_dict_combo.items():
            ms = vals[0] + [vals[2]] + vals[1]
            guesses = evaluate_all_splits(model, msfd, ms, strategy=0, mults=hs)
            for i, guess in enumerate(guesses):
                if guess[0] == len(keys[0]):
                    results[i] += 1

        for key, val in results.items():
            print(key, end=": ")
            print(val)



    # think best for 4[.7, .85, 1] was [x, .7, .7!!, .7]?
    if test == 3:
        print("test 3")

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, 500)
        mults = [.3, .4, .5]
        results = {}

        for i in mults:
            hs = (.85, i, .6, .7)
            results[hs] = 0
            print("testing", end=": ")
            print(hs)

            for keys, vals in ms_dict_combo.items():
                ms = vals[0] + [vals[2]] + vals[1]
                guess = evaluate_all_splits(model, msfd, ms, strategy=0, mults=hs)[0][0]
                if guess == len(keys[0]):
                    results[hs] += 1

        res = list(results.items())
        res.sort(key = lambda x : x[1])
        for hs, correct in res:
            print(hs, end=": ")
            print(correct, end="; acc: ")
            print(correct / len(list(ms_dict_combo.items())))


    if test == 2:
        print("test 2")
        hs = (.85, 1, 1, .7)

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, 10)
        results = {}
        for keys, vals in ms_dict_combo.items():
            ms = vals[0] + [vals[2]] + vals[1]
            results[(keys)] = list(map(
                            lambda x : (x[0], (x[1][1], x[1][2])), 
                            evaluate_all_splits(model, msfd, ms, strategy=1, mults=hs)))

        for keys, rs in results.items():
            print(keys[0] + "/" + keys[1])
            vals = ms_dict_combo[keys]
            print(vals[0], end="/")
            print(vals[2], end="/")
            print(vals[1])
            ms = vals[0] + [vals[2]] + vals[1]
            for i, score in rs:
                print(ms[:i], end="/")
                print(ms[i], end="/")
                print(ms[i+1:], end=": ")
                print(score[0], end=", ")
                print(score[1])
            print("\n")
