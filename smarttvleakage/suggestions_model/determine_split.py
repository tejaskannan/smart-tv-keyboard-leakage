from typing import List, Dict, Tuple
from argparse import ArgumentParser
import random

from smarttvleakage.suggestions_model.manual_score_dict import (build_msfd,
                                                            build_ms_dict)
from smarttvleakage.utils.file_utils import read_pickle_gz
from smarttvleakage.suggestions_model.determine_autocomplete import classify_ms_with_msfd



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


def get_split_scores(model, msfd,
                ms_auto : List[int], ms_non : List[int],
                msg : bool = False) -> Tuple[float, float]:
    """Returns two confidences for a given auto/non split"""
    ac = .32
    nc = .26
    clas, _, _, conf = classify_ms_with_msfd(model, msfd, ms_auto,
                                            auto_cutoff=ac, non_cutoff=nc)
    if clas == 1:
        conf_auto = conf
    else:
        conf_auto = 1-conf
    clas, _, _, conf = classify_ms_with_msfd(model, msfd, ms_non,
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



def evaluate_split(model, msfd,
                ms_auto : List[int], ms_non : List[int],
                strategy : int,
                mults : Tuple[float, float, float, float] = (1, 1, 1, 1),
                msg : bool = False) -> Tuple[float, float, float]:
    """Evaluates a given auto/non ms split"""

    conf_auto, conf_non = get_split_scores(model, msfd, ms_auto, ms_non, msg=msg)
    # implement strategy and heurs

    auto_mult, non_mult = apply_heuristics(ms_auto, ms_non, mults)
    conf_auto *= auto_mult
    conf_non *= non_mult

    if strategy == 0:
        score = (conf_auto + conf_non)/2
    else:
        score = (pow(conf_auto, 2) + pow(conf_non, 2)) / 2

    return (score, conf_auto, conf_non)



def evaluate_all_splits(model, msfd, ms : List[int],
            strategy : int,
            mults : Tuple[float, float, float, float] = (1, 1, 1, 1)
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
            model, msfd, ms_auto, ms_non, strategy=strategy, msg=False, mults=mults)

    ranked_splits = list(split_scores.items())
    ranked_splits.sort(key=lambda x : x[1][0], reverse=True)
    return ranked_splits
    #return ms[:ranked_splits[0][0]], ms[ranked_splits[0][0]:]
    # zip the return with each score?






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False) #ms_auto_dict pkl.gz
    parser.add_argument("--ms-path-non", type=str, required=False) #ms_non_dict pkl.gz
    parser.add_argument("--msfd-path", type=str, required=False) #msfd pkl.gz
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    args = parser.parse_args()

    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
        ms_dict_auto = build_ms_dict(args.ms_path_auto)

    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_non)

    if args.msfd_path is None:
        args.msfd_path = "suggestions_model/local/msfd.pkl.gz"
        msfd = build_msfd(args.msfd_path)
    elif args.msfd_path == "exp":
        args.msfd_path = "suggestions_model/msfd_exp.pkl.gz"
        msfd = build_msfd(args.msfd_path)

    if args.model_path is None:
        args.model_path = "suggestions_model/model_sim.pkl.gz"
    model = read_pickle_gz(args.model_path)

    # TESTS
    # 0 - accuracy test
    # 2 - analysis test
    # 3 - heurs test
    test = 3

    # think best for 4[.7, .85, 1] was [x, .7, .7!!, .7]?
    if test == 3:
        print("test 3")

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, 2000)
        mults = [.3, .4, .5, .6]
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

        ms_dict_combo = build_combo_dict(ms_dict_auto, ms_dict_non, 10)
        results = {}
        for keys, vals in ms_dict_combo.items():
            ms = vals[0] + [vals[2]] + vals[1]
            results[(keys)] = list(map(
                            lambda x : (x[0], (x[1][1], x[1][2])), 
                            evaluate_all_splits(model, msfd, ms, strategy=1)))

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
