from typing import List, Dict, Tuple
from argparse import ArgumentParser

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import json

from smarttvleakage.dictionary import EnglishDictionary, restore_dictionary
from smarttvleakage.suggestions_model.alg_determine_autocomplete import get_score_from_ms
from smarttvleakage.suggestions_model.manual_score_dict import (build_msfd,
                                                            build_ms_dict, build_rockyou_ms_dict)
from smarttvleakage.suggestions_model.simulate_ms import (grab_words, simulate_ms, add_mistakes, 
                                                          add_mistakes_to_ms_dict, build_gt_ms_dict, build_moves)
from smarttvleakage.utils.file_utils import read_pickle_gz, save_pickle_gz

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType, SmartTVType, Direction

from smarttvleakage.suggestions_model.msfd_math import combine_confidences, normalize_msfd_score

from smarttvleakage.dictionary.rainbow import PasswordRainbow
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.audio.sounds import SAMSUNG_DELETE, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT

from smarttvleakage.dictionary.dictionaries import NgramDictionary


from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_suggestions

##################### DATA STRUCTS ##########################

# Make Data Structures
def make_column_titles_dist_new(bins : List[int]) -> List[str]:
    """makes titles for distance bins"""
    if bins == []:
        return []
    column_titles = []
    for i in range(max(bins) + 1):
        column_titles.append("")
    for i, b in enumerate(bins):
        column_titles[b] += str(i) + ","
    return column_titles


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


def make_df_new(bins_transform : List[int], weighted : int,
            ms_dict_auto : Dict[str, List[int]], ms_dict_non : Dict[str, List[int]], bin_max = 10):
    """Makes a datagrame given a bin distribution, weighting, and auto and non dictionaries"""
    
    data = np.empty((0, max(bins_transform) + 1 + 2), dtype=float)
    id = 0
    for key in ms_dict_auto.keys():
        list_dist = make_row_dist(ms_dict_auto[key], list(range(bin_max)), weighted)
        list_dist = transform_hist(list_dist, bins_transform)
        new_row = np.array([[id] + [1] + list_dist], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1
    for key in ms_dict_non.keys():
        list_dist = make_row_dist(ms_dict_non[key], list(range(bin_max)), weighted)
        list_dist = transform_hist(list_dist, bins_transform)
        new_row = np.array([[id] + [0] + list_dist], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1

    column_titles = ["id"] + ["ac"] + make_column_titles_dist_new(bins_transform)
    df = pd.DataFrame(data = data, columns = column_titles)
    print(df)
    return df




def transform_hist(hist : List[int], transform : List[int]) -> List[int]:
    "Transforms a given histogram using a given transform"
    new_hist = []
    for i in range(max(transform)+1):
        new_hist.append(0)
    for i, m in enumerate(hist):
        new_hist[transform[i]] += m

    return new_hist



def get_transforms(size : int):
    """Returns all the transforms for a given bin size"""
    transforms = {}
    bins = 2
    while size * (bins-1) < 10:
        new_transform = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for b in range(1, bins):

            start = size*b
            for i in range(start, 10):
                new_transform[i] += 1
        transforms[bins] = new_transform
        bins += 1

    return transforms



################### BUILD MODELS ################



def build_model_sim_new(ms_dict_rockyou, englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, bin_transform : List[int] = [], weight : int = 3,
                    mistakes : bool = False, max_depth : int = 3):
    """Builds a model on simulated data"""

    ms_dict_auto = {}
    ms_dict_non = {}
    for word in words_auto:
        ms_dict_auto[word] = simulate_ms(englishDictionary, ss_path, word, True)
    for word in words_non:
        ms_dict_non[word] = simulate_ms(englishDictionary, ss_path, word, False)

    if mistakes:
        ms_dict_auto = add_mistakes_to_ms_dict(ms_dict_auto)
        ms_dict_non = add_mistakes_to_ms_dict(ms_dict_non)

    if include_rockyou:
        for key in ms_dict_rockyou:
            ms_dict_non[key] = ms_dict_rockyou[key]

    df = make_df_new(bin_transform, weight, ms_dict_auto, ms_dict_non)
    model = RandomForestClassifier(max_depth=max_depth)

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model



def save_model(path : str, model):
    """Saves a model"""
    save_pickle_gz(model, path)






#################### CLASSIFY #########################



def classify_moves(model, moves : List[Move]):

    ms = [move.num_moves for move in moves]
    bins_transform = get_transforms(1)[7] # (Hardcoded for bins (1/7))
   
    data = np.empty((0, max(bins_transform) + 1), dtype=float)
    list_dist = make_row_dist(ms, range(10), 3)
    list_dist = transform_hist(list_dist, bins_transform)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)

    column_titles = make_column_titles_dist_new(bins_transform)
    df = pd.DataFrame(data = data, columns = column_titles)

    # now predict from the dataframe, and then add manual

    pred_probas = model.predict_proba(df)[0]
    if pred_probas[0] >= .5:
        return 0
    if pred_probas[1] > .5:
        return 1


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
    parser.add_argument("--progress-path", type=str, required=False)
    parser.add_argument("--progress", type=str, required=False)
    parser.add_argument("--test", type=str, required=False)
    parser.add_argument("--json-path", type=str, required=False)
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

    if args.ed_path is None:
        englishDictionary = restore_dictionary(
            "suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "build":
        englishDictionary = EnglishDictionary(50)
        englishDictionary.build(
            "suggestions_model/local/dictionaries/enwiki-20210820-words-frequency.txt", 50, True, False)
        englishDictionary.save("suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "ry":
        rockyouDictionary = NgramDictionary.restore("suggestions_model/local/rockyou_dict_updated.pkl.gz")
    elif args.ed_path != "skip":
        englishDictionary = EnglishDictionary.restore(args.ed_path)

    if args.words_path is None:
        args.words_path = "suggestions_model/local/dictionaries/big.txt"

    if args.ss_path is None:
        args.ss_path = "graphs/autocomplete.json"

 
    # use 7 for main tests
    if args.test is None:
        test = 31
    else:
        test = int(args.test)










################### CLASSIFY TESTS #################


    # Classify Test (Bins)
    if test == 1:
        print("test 1")

        #model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 2000)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 5000)
        print("test dicts built")
        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        #ac = .26
        #nc = .32
        #ac = .5
        #nc = .5
        peak = 30
        bins_nums = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        transforms = {}
        transforms[2] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        transforms[3] = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        transforms[4] = [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
        transforms[5] = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4]
        transforms[6] = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]
        transforms[7] = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6]
        transforms[8] = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7]
        transforms[9] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]
        transforms[10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        steps = ["1step", "2step"]
        step_dict = {}
        step_dict["1step"] = (.5, .5)
        step_dict["2step"] = (.26, .32)
        mts = ["mistakes", "nomistakes"]
    
        mistakes_list = [0, 1, 2, 3]

        mds = [1]

        for md in mds:
            for bins in bins_nums:
                print("bins: " + str(bins))
                for mt in mts:
                    #models/size2/model_size2_

                    # model_path = args.model_path + str(bins)
                    # model_path += "bins_" + mt + "__sim.pkl.gz"

                    model_path = args.model_path + "/md" + str(md) + "/bins"
                    model_path += "/size1/model_size1_"
                    model_path += str(bins) + "bins_" + mt + "_sim.pkl.gz"
                    model = read_pickle_gz(model_path)
                    for step in steps:
                        ac = step_dict[step][0]
                        nc = step_dict[step][1]

                        if args.save_path is not None:
                            #dec/size2

                            # full_path = args.save_path + "/" + str(bins) + "bins/"
                            # full_path += mt + "_" + step

                            full_path = args.save_path + "/md" + str(md) + "/bins/size1"
                            full_path += "/" + str(bins) + "bins/"
                            full_path += mt + "_" + step

                        english_accs = []
                        english_f1s = []
                        password_accs = []
                        password_f1s = []

                        for mistakes in mistakes_list:

                            results = {}
                            print("classifying autos")
                            for key, val in ms_dict_auto_test.items():
                                pred = classify_ms_with_msfd_full_new(model, msfd, db, add_mistakes(val, mistakes), bins_transform=transforms[bins], weight=3,
                                    peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                                results[(key, "auto")] = pred
                            print("classifying nons")
                            for key, val in ms_dict_non_test.items():
                                pred = classify_ms_with_msfd_full_new(model, msfd, db, add_mistakes(val, mistakes), bins_transform=transforms[bins], weight=3,
                                    peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                                results[(key, "non")] = pred
                            print("classifying phpbbs")
                            for key, val in ms_dict_phpbb_test.items():
                                pred = classify_ms_with_msfd_full_new(model, msfd, db, add_mistakes(val, mistakes), bins_transform=transforms[bins], weight=3,
                                    peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
                                results[(key, "phpbb")] = pred
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

                            # accuracy and f1 for english and phpbb sets
                            acc_english = accuracy_score(gt[0], preds[0])
                            f1_english = f1_score(gt[0], preds[0])
                            acc_phpbb = accuracy_score(gt[1], preds[1])
                            f1_phpbb = f1_score(gt[1], preds[1])
                            graph_texts = []
                            graph_texts.append("Mistakes: " + str(mistakes))
                            graph_texts.append("English Accuracy:" + str(acc_english)[:6])
                            graph_texts.append("English F1: " + str(f1_english)[:6])
                            graph_texts.append("Phpbb Accuracy: " + str(acc_phpbb)[:6])
                            graph_texts.append("Phpbb F1: " + str(f1_phpbb)[:6])

                            # CM
                            cm_array = np.array(cm_data)
                            inputs = ["auto", "non", "phpbb"]
                            outputs = ["non", "auto"]
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
                            ax.set_title("suggestions classifications (" + mt +" model, " + step +")", fontsize=20)
                            # print textstr
                            for i, t in enumerate(graph_texts):
                                fig.text(.05, .2 - (.05 * i), t, fontsize=12)
                            #fig.tight_layout()
                            #plt.show()

                            if args.save_path is not None:
                                fig.savefig(full_path + "_" + str(mistakes) + "_mistakes.png")

                            english_accs.append(acc_english)
                            english_f1s.append(f1_english)
                            password_accs.append(acc_phpbb)
                            password_f1s.append(f1_phpbb)

                        lines = []
                        for mistakes in mistakes_list:
                            lines.append("mistakes: " + str(mistakes) + "\n")
                            lines.append("(english) acc: " + str(english_accs[mistakes])
                                + "; f1: " + str(english_f1s[mistakes]) + "\n")
                            lines.append("(phpbb) acc: " + str(password_accs[mistakes])
                                + "; f1: " + str(password_f1s[mistakes]) + "\n\n")

                        with open(full_path + ".txt", "w") as f:
                            f.writelines(lines)
                            f.close()


    



############### BUILD MODELS ####################

    # build and save model
    elif test == 0:
        count = 2000
        max_depth = 1

        #bs = [[0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        #    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]]
        bs = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
            [0, 1, 2, 3, 4, 5, 6, 6, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 7],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        print("test 0")
        if args.save_path is None:
            print("no save path")
        else:
            # ms_dict_auto_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_auto))
            # ms_dict_non_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_non))
            # ms_dict_rockyou_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, 2000))
            # model = build_model(
            #     include_rockyou=True, bins=6, weight=3,
            #     ms_dict_auto=ms_dict_auto_train, ms_dict_non=ms_dict_non_train,
            #     ms_dict_rockyou=ms_dict_rockyou_train)
            # save_model(args.save_path + "_gt.pkl.gz", model)

            # Test out building models with strange bin histograms!!!
            # Then do audio!!

            auto_words = grab_words(2000, args.words_path)
            non_words = grab_words(2000, args.words_path)
            ms_dict_rockyou_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, 2000))

            for b in bs:
                
                model = build_model_sim_new(ms_dict_rockyou=ms_dict_rockyou_train,
                                            englishDictionary=englishDictionary,
                                            ss_path=args.ss_path,
                                            words_auto=auto_words, words_non=non_words,
                                            include_rockyou=True, bin_transform=b, weight=3,
                                            mistakes=True, max_depth = max_depth)
                save_model(args.save_path + str(max(b) + 1) + "bins_mistakes__sim.pkl.gz", model)

                print("model saved: " + str(max(b) + 1))




############### BUILD TEST/TRAIN SET #######


    if test == 20: # build train set
        print("test 20")
        if args.save_path is None:
            print("no save path")

        count = 2000

        english_words = grab_words(count, args.words_path)
        ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, count)
        ms_dict_rockyou_train_mistakes = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, count))

        train_set = {}
        train_set["words"] = english_words
        train_set["rockyou"] = ms_dict_rockyou_train

        save_pickle_gz(train_set, args.save_path)

    if test == 19: # build test set
        print("test 20")
        if args.save_path is None:
            print("no save path")

        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 2000)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 5000)

        test_set = {}
        test_set["auto"] = ms_dict_auto_test
        test_set["non"] = ms_dict_non_test
        test_set["rockyou"] = ms_dict_rockyou_test
        test_set["phpbb"] = ms_dict_phpbb_test

        save_pickle_gz(test_set, args.save_path)

    # build train/test dicts as dicts
    if test == 18:
        print("test 18")

        english_words = grab_words(2000, args.words_path)
        ms_dict_auto = {}
        ms_dict_non = {}
        for word in english_words:
            ms_dict_auto[word] = simulate_ms(englishDictionary, args.ss_path, word, True)
            ms_dict_non[word] = simulate_ms(englishDictionary, args.ss_path, word, False)
        ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, 2000)

        save_pickle_gz(ms_dict_auto, "word_dicts/train_english_auto.pkl.gz")
        save_pickle_gz(ms_dict_non, "word_dicts/train_english_non.pkl.gz")
        save_pickle_gz(ms_dict_auto, "word_dicts/train_rockyou.pkl.gz")

        print("saved nons")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 2000)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 5000)

        save_pickle_gz(ms_dict_auto_test, "word_dicts/test_english_auto.pkl.gz")
        save_pickle_gz(ms_dict_non_test, "word_dicts/test_english_non.pkl.gz")
        save_pickle_gz(ms_dict_rockyou_test, "word_dicts/test_rockyou.pkl.gz")
        save_pickle_gz(ms_dict_phpbb_test, "word_dicts/test_phpbb.pkl.gz")


