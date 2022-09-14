from typing import List, Dict, Tuple
from argparse import ArgumentParser

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from smarttvleakage.dictionary import EnglishDictionary
from smarttvleakage.suggestions_model.alg_determine_autocomplete import get_score_from_ms
from smarttvleakage.suggestions_model.manual_score_dict import (build_msfd,
                                                            build_ms_dict, build_rockyou_ms_dict)
from smarttvleakage.suggestions_model.simulate_ms import grab_words, simulate_ms
from smarttvleakage.utils.file_utils import read_pickle_gz, save_pickle_gz

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType, SmartTVType, Direction

from smarttvleakage.suggestions_model.msfd_math import combine_confidences, normalize_msfd_score

from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.audio.constants import SAMSUNG_DELETE, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT

from smarttvleakage.dictionary.dictionaries import NgramDictionary



# Make Data Structures
def make_column_titles_dist(bins : List[int]) -> List[str]:
    """makes titles for distance bins"""
    if bins == []:
        return []
    return list(map(lambda n: "d " + n, map(str, bins)))


def move_weight(i : int, strategy : int) -> int:
    """Returns a move's weight given a strategy"""
    if strategy in [0, 2]:
        return 1
    if strategy in [1, 3]:
        return i
    if strategy in [4, 5]:
        return i*i
    return 0

def moves_to_hist_dist(moves : List[int], bins : List[int], weighted : int) -> Dict[int, int]:
    """Turns move list into histogram dictionary given bins and weighting"""
    hist = {}
    for b in bins:
        hist[b] = 0

    for i, move in enumerate(moves):
        weight = move_weight(i, weighted)
        if move in bins:
            hist[move] += weight
        else:
            hist[bins[len(bins) - 1]] += weight

    if weighted in [0, 1, 4]:
        norm = sum([hist[b] for b in bins])
        for b in bins:
            hist[b] = hist[b] / max(1, norm)
    return hist

def hist_to_list(hist : Dict[int, int], bins : int) -> List[int]:
    """Turns histogram dictionary into list for features"""
    lst = []
    for b in bins:
        lst.append(hist[b])
    return lst


def make_row_dist(moves : List[int], bins : List[int], weighted : int) -> List[int]:
    """Turns a list of moves into a feature list"""
    if bins == []:
        return []
    return hist_to_list(moves_to_hist_dist(moves, bins, weighted), bins)

def make_df(bins_dist : List[int], weighted : int,
            ms_dict_auto : Dict[str, List[int]], ms_dict_non : Dict[str, List[int]]):
    """Makes a datagrame given a bin distribution, weighting, and auto and non dictionaries"""
    
    data = np.empty((0, len(bins_dist) + 2), dtype=float)
    id = 0
    for key in ms_dict_auto.keys():
        list_dist = make_row_dist(ms_dict_auto[key], bins_dist, weighted)
        new_row = np.array([[id] + [1] + list_dist], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1
    for key in ms_dict_non.keys():
        list_dist = make_row_dist(ms_dict_non[key], bins_dist, weighted)
        new_row = np.array([[id] + [0] + list_dist], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1

    column_titles = ["id"] + ["ac"] + make_column_titles_dist(bins_dist)
    df = pd.DataFrame(data = data, columns = column_titles)
    print(df)
    return df




def id_to_weight(id : int) -> str:
    """Describes a weight ID"""
    weight_names = []
    weight_names.append("unweighted, normalized")
    weight_names.append("weighted, normalized")
    weight_names.append("unweighted, not normalized")
    weight_names.append("weighted, not normalized")
    weight_names.append("double weighted, normalized")
    weight_names.append("double weighted, not normalized")
    return weight_names[id]





# builds model
def build_model(ms_dict_auto, ms_dict_non, ms_dict_rockyou,
                include_rockyou : bool = False, bins : int = 4, weight : int = 3):
    """Build a model on GT data"""

    if include_rockyou:
        for key in ms_dict_rockyou:
            ms_dict_non[key] = ms_dict_rockyou[key]

    df = make_df(range(bins), weight, ms_dict_auto, ms_dict_non)
    model = RandomForestClassifier()

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model

def build_model_sim(ms_dict_rockyou,
                    englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, bins : int = 4, weight : int = 3):
    """Builds a model on simulated data"""

    ms_dict_auto = {}
    ms_dict_non = {}
    for word in words_auto:
        ms_dict_auto[word] = simulate_ms(englishDictionary, ss_path, word, True)
    for word in words_non:
        ms_dict_non[word] = simulate_ms(englishDictionary, ss_path, word, False)

    if include_rockyou:
        for key in ms_dict_rockyou:
            ms_dict_non[key] = ms_dict_rockyou[key]

    df = make_df(range(bins), weight, ms_dict_auto, ms_dict_non)
    model = RandomForestClassifier()

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model

def save_model(path : str, model):
    """Saves a model"""
    save_pickle_gz(model, path)


# Classify move sequences


# takes in a model and a move sequence, returns an int; 1 for auto, 0 for non
def classify_ms(model, ms : List[int],
                bins : int = 3, weight : int = 3) -> Tuple[int, float]:
    """Classifies a move sequence, returns class and confidence"""
    data = np.empty((0, bins), dtype=float)
    list_dist = make_row_dist(ms, range(bins), weight)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)
    column_titles = make_column_titles_dist(range(bins))
    df = pd.DataFrame(data = data, columns = column_titles)

    pred_probas = model.predict_proba(df)[0]
    if pred_probas[0] >= .5:
        return (0, pred_probas[0])
    return (1, pred_probas[1])


def classify_ms_with_msfd(model, msfd,
                ms : List[int],
                bins : int = 3, weight : int = 3,
                auto_cutoff : float = .5, non_cutoff : float = .5) -> Tuple[int, float, float, float]:
    """Classifies a move sequence using ML and algorithmic method,
    returns class, ML confidence, manual score, combined score"""

    data = np.empty((0, bins), dtype=float)
    list_dist = make_row_dist(ms, range(bins), weight)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)

    column_titles = make_column_titles_dist(range(bins))
    df = pd.DataFrame(data = data, columns = column_titles)

    # now predict from the dataframe, and then add manual

    pred_probas = model.predict_proba(df)[0]
    if pred_probas[0] >= 1-non_cutoff:
        return (0, pred_probas[0], -1, pred_probas[0])
    if pred_probas[1] > 1-auto_cutoff:
        return (1, pred_probas[1], -1, pred_probas[1])

    # go into manual scoring, use strategy = 2!
    manual_score = get_score_from_ms(ms, 2, msfd)[0][1]
    combined_score, normalized_manual_score = combine_confidences(pred_probas[0], manual_score, 0)
    if combined_score > .5:
        return (0, pred_probas[0], normalized_manual_score, combined_score)
    return (1, pred_probas[1], normalized_manual_score, 1-combined_score)

def classify_ms_with_msfd_full(model, msfd, db,
                ms : List[int],
                bins : int = 3, weight : int = 3, peak : float = 30,
                auto_cutoff : float = .5, non_cutoff : float = .5) -> Tuple[int, float, float, float]:
    """Classifies a move sequence using ML and algorithmic method,
    returns class, ML confidence, manual score, combined score"""

    data = np.empty((0, bins), dtype=float)
    list_dist = make_row_dist(ms, range(bins), weight)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)

    column_titles = make_column_titles_dist(range(bins))
    df = pd.DataFrame(data = data, columns = column_titles)

    # now predict from the dataframe, and then add manual

    pred_probas = model.predict_proba(df)[0]
    if pred_probas[0] >= 1-non_cutoff:
        return (0, pred_probas[0], -1, pred_probas[0])
    if pred_probas[1] > 1-auto_cutoff:
        return (1, pred_probas[1], -1, pred_probas[1])

    # go into manual scoring, use strategy = 2!
    manual_score_msfd = get_score_from_ms(ms, 2, msfd)[0][1]

    moves = [Move(num_moves=num_moves, directions=Direction.ANY,
                    end_sound=SAMSUNG_KEY_SELECT) for num_moves in ms]
    db_strings = db.get_strings_for_seq(moves, SmartTVType.SAMSUNG, max_num_results=None)
    if db_strings == []:
        manual_score_db = 0
    else:
        manual_score_db = db_strings[0][1]

    combined_score, normalized_manual_score = combine_confidences(pred_probas[0], manual_score_msfd, manual_score_db, peak=peak)
    if combined_score > .5:
        return (0, pred_probas[0], normalized_manual_score, combined_score)
    return (1, pred_probas[1], normalized_manual_score, 1-combined_score)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False) #ms_auto_dict pkl.gz
    parser.add_argument("--ms-path-non", type=str, required=False) #ms_non_dict pkl.gz
    parser.add_argument("--ms-path-rockyou", type=str, required=False) #rockyou.txt
    parser.add_argument("--ed-path", type=str, required=False) #ed pkl.gz, only build works rn
    parser.add_argument("--words-path", type=str, required=False) #big.txt
    parser.add_argument("--msfd-path", type=str, required=False) #msfd pkl.gz
    parser.add_argument("--results-path", type=str, required=False) #where to save results
    parser.add_argument("--save-path", type=str, required=False) #where to save model
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    parser.add_argument("--ss-path", type=str, required=False) #single suggestions .json
    parser.add_argument("--move-sequence", type=int, nargs='+', required=False)
    parser.add_argument("--target", type=str, required=False)
    parser.add_argument("--classify", type=str, required=False)
    args = parser.parse_args()

    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
        ms_dict_auto = build_ms_dict(args.ms_path_auto)

    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_non)

    if args.ms_path_rockyou is None:
        args.ms_path_rockyou = "suggestions_model/local/ms_dict_rockyou.pkl.gz"
        ms_dict_rockyou = build_ms_dict(args.ms_path_rockyou, 500)
    elif args.ms_path_rockyou == "test":
        ms_dict_rockyou = build_ms_dict("suggestions_model/local/ms_dict_rockyou.pkl.gz")

    if args.msfd_path is None:
        args.msfd_path = "suggestions_model/msfd_exp.pkl.gz"
        msfd = build_msfd(args.msfd_path)
    elif args.msfd_path == "base":
        args.msfd_path = "suggestions_model/msfd_exp.pkl.gz"
        msfd = build_msfd(args.msfd_path)
    elif args.msfd_path == "both":
        args.msfd_path = "suggestions_model/local/msfd.pkl.gz"
        msfd_base = build_msfd(args.msfd_path)
        args.msfd_path = "suggestions_model/msfd_exp.pkl.gz"
        msfd_exp = build_msfd(args.msfd_path)

    if args.ed_path is None:
        englishDictionary = EnglishDictionary.restore(
            "suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "build":
        englishDictionary = EnglishDictionary(50)
        englishDictionary.build(
            "suggestions_model/local/dictionaries/enwiki-20210820-words-frequency.txt", 50, True)
        englishDictionary.save("suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "ry":
        rockyouDictionary = NgramDictionary.restore("suggestions_model/local/rockyou_d.pkl.gz")
    elif args.ed_path != "skip":
        englishDictionary = EnglishDictionary.restore(args.ed_path)

    if args.words_path is None:
        args.words_path = "suggestions_model/local/dictionaries/big.txt"

    if args.ss_path is None:
        args.ss_path = "graphs/autocomplete.json"

    # Tests
    # 0 - save model
    # 1 - test all dicts
    # 2 - test all dicts configs
    # 3 - (1) with sim model
    # 4 - (2) with sim model
    # 5 - reveal mistake words ADD
    # 6 - (5) with build model (sim)
    # 7 - test loaded model with 3 class breakdown
    # 8 - test a move sequence with a model
    # 9 - find best cutoffs
    # 10 - test msfd vs exp
    # 11 - sql test
    # 12 - test msfd vs exp vs sql
    test = 13

    if test == 13:
        print("test 13")
        
        model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 500)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 100)
        print("test dicts built")
        db_path = "rockyou-samsung.db"
        db = PasswordRainbow(db_path)

        ac = .26
        nc = .32
        results = {}
        peaks = [25, 30, 35, 40]
        total = len(
            ms_dict_auto_test.items()) + len(
            ms_dict_non_test.items()) + len(
            ms_dict_rockyou_test.items()) + len(
            ms_dict_phpbb_test.items())
        for peak in peaks:
            results[peak] = []
            msfd = msfd_base

            print("classifying autos")
            for key, val in ms_dict_auto_test.items():
                pred = classify_ms_with_msfd_full(model, msfd, db,
                    val, bins=3, weight=3, peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 0:
                    results[peak].append((key, "auto"))
            print("classifying nons")
            for key, val in ms_dict_non_test.items():
                pred = classify_ms_with_msfd_full(model, msfd, db,
                    val, bins=3, weight=3, peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[peak].append((key, "non"))
            print("classifying rockyous")
            for key, val in ms_dict_rockyou_test.items():
                pred = classify_ms_with_msfd_full(model, msfd, db,
                    val, bins=3, weight=3, peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[peak].append((key, "rockyou"))
            print("classifying phpbbs")
            for key, val in ms_dict_phpbb_test.items():
                pred = classify_ms_with_msfd_full(model, msfd, db,
                    val, bins=3, weight=3, peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[peak].append((key, "phpbb"))
            #print("classified")


        for peak in peaks:
            non_list = list(filter(lambda x : x[1] == "non", results[peak]))
            non_acc = 1-len(non_list)/len(ms_dict_non_test.items())
            auto_list = list(filter(lambda x : x[1] == "auto", results[peak]))
            auto_acc = 1-len(auto_list)/len(ms_dict_auto_test.items())
            rockyou_list = list(filter(lambda x : x[1] == "rockyou", results[peak]))
            rockyou_acc = 1-len(rockyou_list)/len(ms_dict_rockyou_test.items())
            phpbb_list = list(filter(lambda x : x[1] == "phpbb", results[peak]))
            phpbb_acc = 1-len(phpbb_list)/len(ms_dict_phpbb_test.items())

            print("(peak = " + str(peak) + ") ", end="")
            print("weighted all: ", end="")
            acc = 0
            acc += non_acc * (len(ms_dict_non_test.items())/total)
            acc += auto_acc * (len(ms_dict_auto_test.items())/total)
            acc += rockyou_acc * (len(ms_dict_rockyou_test.items())/total)
            acc += phpbb_acc * (len(ms_dict_phpbb_test.items())/total)
            print(acc, end="| ")

            print("non: ", end="")
            print(non_acc, end="; ")
            print("auto: ", end="")
            print(auto_acc, end="; ")
            print("rockyou: ", end="")
            print(rockyou_acc, end="; ")
            print("phpbb: ", end="")
            print(phpbb_acc)



    if test == 12:
        print("test 12")
        
        model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 1000, 500)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 1000)
        print("test dicts built")
        db_path = "rockyou-samsung.db"
        db = PasswordRainbow(db_path)

        ac = .26
        nc = .32
        results = {}
        ds = ["base", "exp", "sql"]
        for d in ds:
            results[d] = []
            if d == "exp":
                msfd = msfd_exp
            else:
                msfd = msfd_base

            print("classifying autos")
            for key, val in ms_dict_auto_test.items():
                if d == "sql":
                    pred = classify_ms_with_msfd_full(model, msfd, db,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                else:
                    pred = classify_ms_with_msfd(model, msfd,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 0:
                    results[d].append((key, "auto"))
            print("classifying nons")
            for key, val in ms_dict_non_test.items():
                if d == "sql":
                    pred = classify_ms_with_msfd_full(model, msfd, db,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                else:
                    pred = classify_ms_with_msfd(model, msfd,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[d].append((key, "non"))
            print("classifying rockyous")
            for key, val in ms_dict_rockyou_test.items():
                if d == "sql":
                    pred = classify_ms_with_msfd_full(model, msfd, db,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                else:
                    pred = classify_ms_with_msfd(model, msfd,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[d].append((key, "rockyou"))
            print("classifying phpbbs")
            for key, val in ms_dict_phpbb_test.items():
                if d == "sql":
                    pred = classify_ms_with_msfd_full(model, msfd, db,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                else:
                    pred = classify_ms_with_msfd(model, msfd,
                        val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[d].append((key, "phpbb"))
            #print("classified")


        for d in ds:
            print(d)
            print("all: ", end="")
            print(len(results[d]), end="| ")
            non_list = list(filter(lambda x : x[1] == "non", results[d]))
            auto_list = list(filter(lambda x : x[1] == "auto", results[d]))
            rockyou_list = list(filter(lambda x : x[1] == "rockyou", results[d]))
            phpbb_list = list(filter(lambda x : x[1] == "phpbb", results[d]))
            print("non: ", end="")
            print(len(non_list), end="; ")
            print("auto: ", end="")
            print(len(auto_list), end="; ")
            print("rockyou: ", end="")
            print(len(rockyou_list), end="; ")
            print("phpbb: ", end="")
            print(len(phpbb_list))




    if test == 11:
        print("test 11")
        #model = read_pickle_gz(args.model_path)
        db_path = "rockyou-samsung.db"
        db = PasswordRainbow(db_path)
        ms = args.move_sequence
        print(ms)

        moves = [Move(num_moves=num_moves, directions=Direction.ANY,
                    end_sound=SAMSUNG_KEY_SELECT) for num_moves in ms]
        print(moves)

        print("test without")
        db_strings = db.get_strings_for_seq(moves, tv_type=SmartTVType.SAMSUNG, max_num_results=None)
        print(db_strings)


        print("\n\ntest reverse")
        db_output = db.get_seq_for_string("12345", tv_type=SmartTVType.SAMSUNG)
        print(db_output)

        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 1000)
        scores = []
        for ms in ms_dict_rockyou_test.values():
            moves = [Move(num_moves=num_moves, directions=Direction.ANY,
                    end_sound=SAMSUNG_KEY_SELECT) for num_moves in ms]
            scores.append(db.get_strings_for_seq(moves, tv_type=SmartTVType.SAMSUNG, max_num_results=None)[0][1])
        scores.sort(key=lambda x : x, reverse=True)
        print(scores[0]) # top score from 1000 - 29.5
        scores.sort(key=lambda x : x, reverse=False)
        print(scores[0]) # 13.47
        # For now, just divide by 30?

    #msfd: all: 353, phpbb: 344
    #exp: all: 266, phpbb: 258
    if test == 10:
        print("test 10")
        model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 500)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 500)
        print("test dicts built")

        ac = .26
        nc = .32
        results = {}
        ds = ["msfd", "exp"]
        for d in ds:
            results[d] = []
            if d == "msfd":
                msfd = msfd_base
            else:
                msfd = msfd_exp

            print("classifying autos")
            for key, val in ms_dict_auto_test.items():
                pred = classify_ms_with_msfd(
                    model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 0:
                    results[d].append((key, "auto"))
            print("classifying nons")
            for key, val in ms_dict_non_test.items():
                pred = classify_ms_with_msfd(
                    model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[d].append((key, "non"))
            print("classifying rockyous")
            for key, val in ms_dict_rockyou_test.items():
                pred = classify_ms_with_msfd(
                    model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[d].append((key, "rockyou"))
            print("classifying phpbbs")
            for key, val in ms_dict_phpbb_test.items():
                pred = classify_ms_with_msfd(
                    model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                if pred == 1:
                    results[d].append((key, "phpbb"))
            print("classified")

        for d in ds:
            print(d)
            for word, ty in results[d]:
                print(word + ", " + ty)
        for d in ds:
            print(d)
            print("all: ", end="")
            print(len(results[d]))
            phpbb_list = list(filter(lambda x : x[1] == "phpbb", results[d]))
            print("phpbb: ", end="")
            print(len(phpbb_list))



    if test == 9:
        print("test 9")
        model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 500)
        print("test dicts built")

        acs = [x / 50 for x in range(11, 21)]
        ncs = [x / 50 for x in range(11, 21)]
        results = {}
        for ac in acs:
            for nc in ncs:
                results[(ac, nc)] = []
                print("classifying autos")
                for key, val in ms_dict_auto_test.items():
                    pred = classify_ms_with_msfd(
                        model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                    if pred == 0:
                        results[(ac, nc)].append((key, "auto"))
                print("classifying nons")
                for key, val in ms_dict_non_test.items():
                    pred = classify_ms_with_msfd(
                        model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                    if pred == 1:
                        results[(ac, nc)].append((key, "non"))
                print("classifying rockyous")
                for key, val in ms_dict_rockyou_test.items():
                    pred = classify_ms_with_msfd(
                        model, msfd, val, bins=3, weight=3, auto_cutoff=ac, non_cutoff=nc)[0]
                    if pred == 1:
                        results[(ac, nc)].append((key, "rockyou"))
                print("classified")
        
        items = list(results.items())
        items.sort(key = lambda x : len(x[1]), reverse=True)
        lines = []
        for params, wrongs in items:
            ac, nc = params
            lines.append("ac: " + str(ac) + "; nc: " + str(nc) + "\n")
            print("ac: " + str(ac) + "; nc: " + str(nc))
            
            for ty in ["non", "auto", "rockyou"]:
                lines.append(ty + ": ")
                for word, _ in filter(lambda x : x[1] == ty, wrongs):
                    lines.append(word + ", ")
                    print(word + ", " + ty)
                lines.append("\n")
            lines.append("\n")

        if args.results_path is not None:
            with open(args.results_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                f.close()

    if test == 8:
        print("test 8")
        model = read_pickle_gz(args.model_path)
        if args.move_sequence is None:
            keyboard_type = KeyboardType.SAMSUNG
            graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
            ms = findPath(args.target, False, False, False, False, 0, 0, 0, graph)
            ms = list(map(lambda x : x.num_moves, ms))
            print(ms)
        else:
            ms = args.move_sequence

        if args.classify is None:
            clas, proba = classify_ms(model, ms)
            print(clas)
            print(proba)
        elif args.classify == "msfd":
            clas, conf, manual, combined = classify_ms_with_msfd(
                model, msfd, ms, auto_cutoff=.32, non_cutoff=.26, weight=1)
            print(clas)
            print(conf)
            print(manual)
            print(combined)

        msfd_word, msfd_score = get_score_from_ms(ms, 2, msfd)[0]
        print("msfd word: " + msfd_word)
        print("msfd score: " + str(msfd_score))


    if test == 7:
        print("test 7")
        model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = ms_dict_rockyou
        print("test dicts built")

        results = {}
        print("classifying autos")
        for key, val in ms_dict_auto_test.items():
            pred = classify_ms(model, val, bins=3, weight=3)[0]
            results[(key, "auto")] = pred
        print("classifying nons")
        for key, val in ms_dict_non_test.items():
            pred = classify_ms(model, val, bins=3, weight=3)[0]
            results[(key, "non")] = pred
        print("classifying rockyous")
        for key, val in ms_dict_rockyou_test.items():
            pred = classify_ms(model, val, bins=3, weight=3)[0]
            results[(key, "rockyou")] = pred
        print("classified")

        gt = ([], [])
        preds = ([], [])
        cm_data = [[0, 0, 0], [0, 0, 0]]
        for key, pred in results.items():
            if key[1] == "auto":
                gt[0].append(1)
                preds[0].append(pred)
                cm_data[pred][0] += 1
            elif key[1] == "non":
                gt[0].append(0)
                preds[0].append(pred)
                cm_data[pred][1] += 1
            else:
                gt[1].append(0)
                preds[1].append(pred)
                cm_data[pred][2] += 1

        # accuracy and f1 for english and rockyou sets
        acc_english = accuracy_score(gt[0], preds[0])
        f1_english = f1_score(gt[0], preds[0])
        acc_rockyou = accuracy_score(gt[1], preds[1])
        f1_rockyou = f1_score(gt[1], preds[1])
        graph_texts = []
        graph_texts.append("English Accuracy:" + str(acc_english)[:6])
        graph_texts.append("English F1: " + str(f1_english)[:6])
        graph_texts.append("Rockyou Accuracy: " + str(acc_rockyou)[:6])
        graph_texts.append("Rockyou F1: " + str(f1_rockyou)[:6])

        # CM
        cm_array = np.array(cm_data)
        inputs = ["auto", "non", "rockyou"]
        outputs = ["auto", "non"]
        fig, ax = plt.subplots()
        im = ax.imshow(cm_array)
        plt.subplots_adjust(bottom=0.3)
        ax.set_xticks(np.arange(len(inputs)), labels=inputs, color="b")
        ax.set_yticks(np.arange(len(outputs)), labels=outputs, color="b")
        plt.xlabel("inputs", fontsize=16)
        plt.ylabel("outputs", fontsize=16)
        for i in range(len(inputs)):
            for j in range(len(outputs)):
                text = ax.text(i, j, cm_array[j, i], ha="center", va="center", color="w")
        ax.set_title("suggestions classifications", fontsize=20)
        # print textstr
        for i, t in enumerate(graph_texts):
            fig.text(.05, .2 - (.05 * i), t, fontsize=12)
        #fig.tight_layout()
        plt.show()

        if args.save_path is not None:
            fig.savefig(args.save_path)

    if test == 6:
        print("test 6")

        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_rockyou_ms_dict(args.ms_path_rockyou, 100, 500)

        model = read_pickle_gz(args.model_path)
        errors = []
        for key, val in ms_dict_auto_test.items():
            pred = classify_ms_with_msfd(
                model, msfd, val, bins=3, weight=3, auto_cutoff=.25, non_cutoff=.25)[0]
            if pred != 1:
                errors.append((key, "auto", val))

        for key, val in ms_dict_non_test.items():
            pred = classify_ms_with_msfd(
                model, msfd, val, bins=3, weight=3, auto_cutoff=.25, non_cutoff=.25)[0]
            if pred != 0:
                errors.append((key, "non", val))

        for key, val in ms_dict_rockyou_test.items():
            pred = classify_ms_with_msfd(
                model, msfd, val, bins=3, weight=3, auto_cutoff=.25, non_cutoff=.25)[0]
            if pred != 0:
                errors.append((key, "rockyou", val))

        print("errors:")
        for word, ty, ms in errors:
            print(word + ", " + ty, end=": ")
            print(ms)

    if test == 5:
        print("test 5")
        split = 53
        ms_dict_auto_train = build_ms_dict(args.ms_path_auto, take=split)
        ms_dict_non_train = build_ms_dict(args.ms_path_non, take=split)
        ms_dict_rockyou_train = build_rockyou_ms_dict(args.ms_path_rockyou, 500)
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto, take=(0-split))
        ms_dict_non_test = build_ms_dict(args.ms_path_non, take=(0-split))
        ms_dict_rockyou_test = build_rockyou_ms_dict(args.ms_path_rockyou, 100, 500)

        results = {}
        model = build_model(include_rockyou=True, bins=3, weight=3,
                            ms_dict_auto=ms_dict_auto_train, ms_dict_non=ms_dict_non_train,
                            ms_dict_rockyou=ms_dict_rockyou_train)

        errors = []
        for key, val in ms_dict_auto_test.items():
            pred = classify_ms_with_msfd(
                model, msfd, val, bins=3, weight=3, auto_cutoff=.25, non_cutoff=.25)[0]
            if pred != 1:
                errors.append((key, "auto", val))

        for key, val in ms_dict_non_test.items():
            pred = classify_ms_with_msfd(
                model, msfd, val, bins=3, weight=3, auto_cutoff=.25, non_cutoff=.25)[0]
            if pred != 0:
                errors.append((key, "non", val))

        for key, val in ms_dict_rockyou_test.items():
            pred = classify_ms_with_msfd(
                model, msfd, val, bins=3, weight=3, auto_cutoff=.25, non_cutoff=.25)[0]
            if pred != 0:
                errors.append((key, "rockyou", val))

        print("errors:")
        for word, ty, ms in errors:
            print(word + ", " + ty, end=": ")
            print(ms)
    



    if test == 4:
        print("test 4")
        auto_words = grab_words(200, args.words_path)
        non_words = grab_words(200, args.words_path)
        ms_dict_rockyou_train = build_rockyou_ms_dict(args.ms_path_rockyou, 500)
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_rockyou_ms_dict(args.ms_path_rockyou, 100, 500)

        bss = [3, 4, 5]
        weights = [3, 4, 5]
        certainty_cutoffs = [.25, .3, .35, .4, .45, .5]

        print("building rockyou dict")
        print("build rockyou dict")

        results = {}
        for bs in bss:
            for w in weights:
                model = build_model_sim(ms_dict_rockyou=ms_dict_rockyou_train,
                                        englishDictionary=englishDictionary, ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, bins=bs, weight=w)
                    
                for cc in certainty_cutoffs:
                    print("testing bins: " + str(bs) + "; cc: " + str(cc) + "; " + id_to_weight(w))

                    gt_array = []
                    label_array = []

                    for _, val in ms_dict_auto_test.items():
                        pred = classify_ms_with_msfd(
                            model, msfd, val, bins=bs, weight=w, auto_cutoff=cc, non_cutoff=cc)[0]
                        gt_array.append(1)
                        label_array.append(pred)

                    for _, val in ms_dict_non_test.items():
                        pred = classify_ms_with_msfd(
                            model, msfd, val, bins=bs, weight=w, auto_cutoff=cc, non_cutoff=cc)[0]
                        gt_array.append(0)
                        label_array.append(pred)

                    for _, val in ms_dict_rockyou_test.items():
                        pred = classify_ms_with_msfd(
                            model, msfd, val, bins=bs, weight=w, auto_cutoff=cc, non_cutoff=cc)[0]
                        gt_array.append(0)
                        label_array.append(pred)

                    cm = confusion_matrix(gt_array, label_array, labels=[0, 1])
                    acc = accuracy_score(gt_array, label_array)
                    f1 = f1_score(gt_array, label_array)

                    results[(bs, w, cc)] = (cm, acc, f1)

        results_list = []
        for bs, w, cc in results:
            cm, acc, f1 = results[(bs, w, cc)]
            results_list.append((bs, w, cc, cm, acc, f1))

        results_list.sort(key=(lambda x : x[4]), reverse=True)
        lines = []
        for bs, w, cc, cm, acc, f1 in results_list:
            print("bins: " + str(bs) + "; weight: " + str(w) + "; cc: " + str(cc))
            print("acc: " + str(acc))
            print("f1: " + str(f1))
            #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non", "auto"])
            #disp.plot()
            #plt.show()
            print("\n")

            lines.append("bins: " + str(bs) + "; weight: " + str(w) + "; cc: " + str(cc) + "\n")
            lines.append("acc: " + str(acc))
            lines.append(", f1: " + str(f1) + "\n")
            lines.append("cm:\n" + str(cm))
            lines.append("\n\n")

        if args.results_path is not None:
            with open(args.results_path, "a", encoding="utf-8") as f:
                f.writelines(lines)
                f.close()


    if test == 3:
        print("test 3")
        model = build_model_sim(build_rockyou_ms_dict(args.ms_path_rockyou, 500), englishDictionary,
                                ss_path=args.ss_path,
                                words_auto=grab_words(2000, args.words_path),
                                words_non=grab_words(2000, args.words_path),
                                include_rockyou=True)

        ms_dict_rockyou = build_rockyou_ms_dict(args.ms_path_rockyou, 105, 200)

        correct_auto = 0
        correct_non = 0
        correct_rockyou = 0
        total_auto = 0
        total_non = 0
        total_rockyou = 0

        for key, val in ms_dict_auto.items():
            print(key)
            if classify_ms_with_msfd(model, msfd, val, auto_cutoff=.5, non_cutoff=.5)[0] == 1:
                correct_auto += 1
            total_auto += 1
        for key, val in ms_dict_non.items():
            print(key)
            if classify_ms_with_msfd(model, msfd, val, auto_cutoff=.5, non_cutoff=.5)[0] == 0:
                correct_non += 1
            total_non += 1
        for key, val in ms_dict_rockyou.items():
            print(key)
            if classify_ms_with_msfd(model, msfd, val, auto_cutoff=.5, non_cutoff=.5)[0] == 0:
                correct_rockyou += 1
            total_rockyou += 1

        print("correct auto: " + str(correct_auto))
        print("total auto: " + str(total_auto))
        print("accuracy auto: " + str(correct_auto/total_auto))
        print("\n")
        print("correct non: " + str(correct_non))
        print("total non: " + str(total_non))
        print("accuracy non: " + str(correct_non/total_non))
        print("\n")
        print("correct rockyou: " + str(correct_rockyou))
        print("total rockyou: " + str(total_rockyou))
        print("accuracy rockyou: " + str(correct_rockyou/total_rockyou))
        print("\n")
        correct = correct_auto + correct_non + correct_rockyou
        total = total_auto + total_non + total_rockyou
        print("correct: " + str(correct))
        print("total auto: " + str(total))
        print("accuracy auto: " + str(correct/total))

    if test == 2:
        split = 53
        ms_dict_auto_train = build_ms_dict(args.ms_path_auto, take=split)
        ms_dict_non_train = build_ms_dict(args.ms_path_non, take=split)
        ms_dict_rockyou_train = build_rockyou_ms_dict(args.ms_path_rockyou, 500)
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto, take=(0-split))
        ms_dict_non_test = build_ms_dict(args.ms_path_non, take=(0-split))
        ms_dict_rockyou_test = build_rockyou_ms_dict(args.ms_path_rockyou, 100, 500)

        bss = [3, 4, 5]
        weights = [3, 4, 5]
        certainty_cutoffs = [.25, .3, .35, .4, .45, .5]
        print("test 2")

        results = {}
        for bs in bss:
            for w in weights:
                model = build_model(include_rockyou=True, bins=bs, weight=w, ms_dict_auto=ms_dict_auto_train, ms_dict_non=ms_dict_non_train, ms_dict_rockyou=ms_dict_rockyou_train)
                    
                for cc in certainty_cutoffs:
                    print("testing bins: " + str(bs) + "; cc: " + str(cc) + "; " + id_to_weight(w))

                    gt_array = []
                    label_array = []

                    for _, val in ms_dict_auto_test.items():
                        pred = classify_ms_with_msfd(model, msfd, val, bins=bs, weight=w, auto_cutoff=cc, non_cutoff=cc)[0]
                        gt_array.append(1)
                        label_array.append(pred)

                    for _, val in ms_dict_non_test.items():
                        pred = classify_ms_with_msfd(model, msfd, val, bins=bs, weight=w, auto_cutoff=cc, non_cutoff=cc)[0]
                        gt_array.append(0)
                        label_array.append(pred)

                    for _, val in ms_dict_rockyou_test.items():
                        pred = classify_ms_with_msfd(model, msfd, val, bins=bs, weight=w, auto_cutoff=cc, non_cutoff=cc)[0]
                        gt_array.append(0)
                        label_array.append(pred)

                    cm = confusion_matrix(gt_array, label_array, labels=[0, 1])
                    acc = accuracy_score(gt_array, label_array)
                    f1 = f1_score(gt_array, label_array)

                    results[(bs, w, cc)] = (cm, acc, f1)

        results_list = []
        for bs, w, cc in results:
            cm, acc, f1 = results[(bs, w, cc)]
            results_list.append((bs, w, cc, cm, acc, f1))
        
        results_list.sort(key=(lambda x : x[4]), reverse=True)
        
        lines = []
        for bs, w, cc, cm, acc, f1 in results_list:
            print("bins: " + str(bs) + "; weight: " + str(w) + "; cc: " + str(cc))
            print("acc: " + str(acc))
            print("f1: " + str(f1))
            #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["non", "auto"])
            #disp.plot()
            #plt.show()
            print("\n")

            lines.append("bins: " + str(bs) + "; weight: " + str(w) + "; cc: " + str(cc) + "\n")
            lines.append("acc: " + str(acc))
            lines.append(", f1: " + str(f1) + "\n")
            lines.append("cm:\n" + str(cm))
            lines.append("\n\n")
        if args.results_path is not None:
            with open(args.results_path, "w") as f:
                f.writelines(lines)
                f.close()

    if test == 1:
        print("test 1")
        #model = read_pickle_gz("max/model_rockyou.pkl.gz")
        model = build_model(include_rockyou=True,
                        ms_dict_auto=build_ms_dict(args.ms_path_auto, take=53),
                        ms_dict_non=build_ms_dict(args.ms_path_non, take=70),
                        ms_dict_rockyou=ms_dict_rockyou)
        ms_dict_auto = build_ms_dict(args.ms_path_auto, take=-53)
        ms_dict_non = build_ms_dict(args.ms_path_non, take=-53)

        ms_dict_rockyou = build_rockyou_ms_dict(args.ms_path_rockyou, 100, 500)
        
        correct_auto = 0
        correct_non = 0
        correct_rockyou = 0
        total_auto = 0
        total_non = 0
        total_rockyou = 0

        gt_array = []
        label_array = []

        for key, val in ms_dict_auto.items():
            print(key)
            pred = classify_ms_with_msfd(model, msfd, val, auto_cutoff=.3, non_cutoff=.3)[0]
            if pred == 1:
                correct_auto += 1
            total_auto += 1

            gt_array.append(1)
            label_array.append(pred)
        for key, val in ms_dict_non.items():
            print(key)
            pred = classify_ms_with_msfd(model, msfd, val, auto_cutoff=.3, non_cutoff=.3)[0]
            if pred == 0:
                correct_non += 1
            total_non += 1

            gt_array.append(0)
            label_array.append(pred)
        for key, val in ms_dict_rockyou.items():
            print(key)
            pred = classify_ms_with_msfd(model, msfd, val, auto_cutoff=.3, non_cutoff=.3)[0]
            if pred == 0:
                correct_rockyou += 1
            total_rockyou += 1

            gt_array.append(0)
            label_array.append(pred)

        print("correct auto: " + str(correct_auto))
        print("total auto: " + str(total_auto))
        print("accuracy auto: " + str(correct_auto/total_auto))
        print("\n")
        print("correct non: " + str(correct_non))
        print("total non: " + str(total_non))
        print("accuracy non: " + str(correct_non/total_non))
        print("\n")
        print("correct rockyou: " + str(correct_rockyou))
        print("total rockyou: " + str(total_rockyou))
        print("accuracy rockyou: " + str(correct_rockyou/total_rockyou))
        print("\n")
        correct = correct_auto + correct_non + correct_rockyou
        total = total_auto + total_non + total_rockyou
        print("correct: " + str(correct))
        print("total auto: " + str(total))
        print("accuracy auto: " + str(correct/total))

        # Metrics
        print("confusion matrix:")
        print(confusion_matrix(gt_array, label_array))

        print("f1 score: ", end="")
        print(f1_score(gt_array, label_array))

        print("accuracy: ", end="")
        print(accuracy_score(gt_array, label_array))


    # build and save model
    elif test == 0:
        if args.save_path is None:
            print("no save path")
        else:
            ms_dict_auto_train = build_ms_dict(args.ms_path_auto)
            ms_dict_non_train = build_ms_dict(args.ms_path_non)
            ms_dict_rockyou_train = build_rockyou_ms_dict(args.ms_path_rockyou, 500)
            model = build_model(
                include_rockyou=True, bins=3, weight=3,
                ms_dict_auto=ms_dict_auto_train, ms_dict_non=ms_dict_non_train,
                ms_dict_rockyou=ms_dict_rockyou_train)
            save_model(args.save_path + "_gt.pkl.gz", model)

            auto_words = grab_words(2000, args.words_path)
            non_words = grab_words(2000, args.words_path)
            ms_dict_rockyou_train = build_rockyou_ms_dict(args.ms_path_rockyou, 500)
            model = build_model_sim(ms_dict_rockyou=ms_dict_rockyou_train,
                                        englishDictionary=englishDictionary,
                                        ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, bins=3, weight=3)
            save_model(args.save_path + "_sim.pkl.gz", model)

            print("models saved")
