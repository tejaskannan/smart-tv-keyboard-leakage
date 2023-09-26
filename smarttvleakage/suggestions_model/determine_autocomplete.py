from typing import List, Dict, Tuple, Any
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
def make_column_titles_dist(bins : List[int]) -> List[str]:
    """makes titles for distance bins"""
    if bins == []:
        return []
    return list(map(lambda n: "d " + n, map(str, bins)))

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



def make_df_list(ms_dict_auto : Dict[str, List[int]], ms_dict_non : Dict[str, List[int]],
                max_len : int = 10):
    """Makes a datagrame given a bin distribution, weighting, and auto and non dictionaries"""

    data = np.empty((0, max_len + 2), dtype=float)
    id = 0
    for _, val in ms_dict_auto.items():
        print(val)
        padded_ms = val
        for i in range(max_len - len(val)):
            padded_ms.append(-1)
        padded_ms = padded_ms[:max_len]
        new_row = np.array([[id] + [1] + padded_ms], dtype=float)
        print(padded_ms)
        print("\n")
        data = np.append(data, new_row, axis=0)
        id = id + 1
    for _, val in ms_dict_non.items():
        padded_ms = val
        for i in range(max_len - len(val)):
            padded_ms.append(-1)
        padded_ms = padded_ms[:max_len]
        new_row = np.array([[id] + [0] + padded_ms], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1

    column_titles = ["id"] + ["ac"] + list(range(max_len))
    df = pd.DataFrame(data = data, columns = column_titles)
    print(df)
    return df


def make_oh_array(ms, max, max_len):
    oh = []
    for move in ms:
        move = min(move, max-1)
        for i in range(max):
            if i == move:
                oh.append(1)
            else:
                oh.append(0)
        
    for i in range(max_len - len(oh)):
        oh.append(0)
    oh = oh[:max_len]

    return oh



def make_df_oh(ms_dict_auto : Dict[str, List[int]], ms_dict_non : Dict[str, List[int]],
                max_len : int = 50, max : int = 5):
    """Makes a datagrame given a bin distribution, weighting, and auto and non dictionaries"""

    data = np.empty((0, max_len + 2), dtype=float)
    id = 0
    for _, val in ms_dict_auto.items():
        
        padded_ms = make_oh_array(val, max, max_len)
        new_row = np.array([[id] + [1] + padded_ms], dtype=float)
        print(padded_ms)
        print("\n")
        data = np.append(data, new_row, axis=0)
        id = id + 1
    for _, val in ms_dict_non.items():
        padded_ms = make_oh_array(val, max, max_len)
        new_row = np.array([[id] + [0] + padded_ms], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1

    column_titles = ["id"] + ["ac"] + list(range(max_len))
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

# sim new - 

# All take:
# ms_dict_rockyou
# englishDictionary
# ss_path
# words_auto
# words_non

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


def build_model_sim_list(ms_dict_rockyou, max_len : int,
                    englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, mistakes : bool = False,
                    max_depth : int = 3):
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

    df = make_df_list(ms_dict_auto, ms_dict_non, max_len)
    model = RandomForestClassifier(max_depth=max_depth)

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model


def build_model_sim_oh(ms_dict_rockyou, max_len : int, max : int,
                    englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, mistakes : bool = False,
                    max_depth : int = 3):
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

    df = make_df_oh(ms_dict_auto, ms_dict_non, max_len, max)
    model = RandomForestClassifier(max_depth=max_depth)

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model


def build_model_sim_list_ada(ms_dict_rockyou, max_len : int,
                    englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, mistakes : bool = False,
                    max_depth : int = 3):
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

    df = make_df_list(ms_dict_auto, ms_dict_non, max_len)
    if max_depth == 1:
        model = AdaBoostClassifier()
    elif max_depth == 0:
        sub_model = DecisionTreeClassifier()
        model = AdaBoostClassifier(base_estimator = sub_model)
    else:
        sub_model = DecisionTreeClassifier(max_depth=max_depth)
        model = AdaBoostClassifier(base_estimator = sub_model)

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model

def build_model_sim_oh_ada(ms_dict_rockyou, max_len : int, max : int,
                    englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, mistakes : bool = False):
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

    df = make_df_oh(ms_dict_auto, ms_dict_non, max_len, max)
    if max_depth == 1:
        model = AdaBoostClassifier()
    elif max_depth == 0:
        sub_model = DecisionTreeClassifier()
        model = AdaBoostClassifier(base_estimator = sub_model)
    else:
        sub_model = DecisionTreeClassifier(max_depth=max_depth)
        model = AdaBoostClassifier(base_estimator = sub_model)

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model


def build_model_sim_new_ada(ms_dict_rockyou,
                    englishDictionary, ss_path : str, words_auto, words_non,
                    include_rockyou : bool = False, bin_transform : List[int] = [], weight : int = 3,
                    mistakes : bool = False, max_depth : int = 1, alg="SAMME.R"):
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

    if max_depth == 1:
        model = AdaBoostClassifier(algorithm=alg)
    elif max_depth == 0:
        sub_model = DecisionTreeClassifier()
        model = AdaBoostClassifier(base_estimator = sub_model, algorithm=alg)
    else:
        sub_model = DecisionTreeClassifier(max_depth=max_depth)
        model = AdaBoostClassifier(base_estimator = sub_model, algorithm=alg)

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model




def save_model(path : str, model):
    """Saves a model"""
    save_pickle_gz(model, path)









#################### CLASSIFY #########################


def classify_ms_with_msfd_full_new(model, msfd, db,
                ms : List[int],
                bins_transform : List[int], weight : int = 3, peak : float = 30,
                auto_cutoff : float = .5, non_cutoff : float = .5) -> Tuple[int, float, float, float]:
    """Classifies a move sequence using ML and algorithmic method,
    returns class, ML confidence, manual score, combined score"""

    data = np.empty((0, max(bins_transform) + 1), dtype=float)
    list_dist = make_row_dist(ms, range(10), weight)
    list_dist = transform_hist(list_dist, bins_transform)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)

    column_titles = make_column_titles_dist_new(bins_transform)
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


def classify_ms_with_msfd_full_list(model, msfd, db,
                ms : List[int], peak : float = 30,
                auto_cutoff : float = .5, non_cutoff : float = .5,
                max_len : int = 10) -> Tuple[int, float, float, float]:
    """Classifies a move sequence using ML and algorithmic method,
    returns class, ML confidence, manual score, combined score"""

    print("ms: ")
    print(ms)

    data = np.empty((0, max_len), dtype=float)
    padded_ms = ms
    print(ms)
    for i in range(max_len - len(ms)):
        padded_ms.append(-1)
    padded_ms = padded_ms[:max_len]
    print(padded_ms)
    new_row = np.array(padded_ms, dtype=float)
    data = np.append(data, [new_row], axis=0)
    print(data)

    column_titles = list(range(max_len))
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


def classify_ms_with_msfd_full_oh(model, msfd, db,
                ms : List[int], peak : float = 30,
                auto_cutoff : float = .5, non_cutoff : float = .5,
                max_len : int = 50, max : int = 5) -> Tuple[int, float, float, float]:
    """Classifies a move sequence using ML and algorithmic method,
    returns class, ML confidence, manual score, combined score"""

    print("ms: ")
    print(ms)

    data = np.empty((0, max_len), dtype=float)

    padded_ms = make_oh_array(ms, max, max_len)
    
    new_row = np.array(padded_ms, dtype=float)
    data = np.append(data, [new_row], axis=0)
    print(data)

    column_titles = list(range(max_len))
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




def classify_moves(model: Any, moves: List[Move], cutoff: float) -> int:
    moves = [m for m in moves if m.end_sound != SAMSUNG_DELETE]
    ms = [move.num_moves for move in moves]
    
    # If we see at least 3 zeros moves after the first move, then the user
    # must have typed the same character 4 times in a row to start (dynamic
    # suggestions do not start until after the first move). Based on the English language,
    # it is exceedingly unlikely that this result is a proper English word.
    num_starting_zeros = 0
    for move_idx in range(1, len(ms)):
        if (ms[move_idx] > 0):
            break

        num_starting_zeros += 1

    if num_starting_zeros >= 3:
        return 0

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
    return int(pred_probas[1] >= cutoff)

    #if pred_probas[0] >= .5:
    #    return 0
    #if pred_probas[1] > .5:
    #    return 1

def classify_moves_prob(model, moves : List[Move]):

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
    return pred_probas[1]


# Train Set : [English_words (2000), Rockyou_dict (2000)]
# Test Set: [Non (106?), Auto (106?), Rockyou (100), Phpbb (5000)]


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

    # Tests
    # 0 - -4 : Building Models
    # 1-4 Classifying
    # 19-20 Building Sets
    # 10-12 Other
    # 30-40 Stats
    # 41 - test Classify Move

    #50-60 Test HT data
    
 
    # use 7 for main tests
    if args.test is None:
        test = 31
    else:
        test = int(args.test)




###################### HT TEST ################


    # 55, but with select cutoffs
    if test == 56:
        print("test 56")

        overrides = (0, 0)

        subjects = ["B", "C", "D", "H"]
        #subjects = ["B"]
        acc = {}
        acc[0] = 0
        acc[1] = 0

        ms = build_gt_ms_dict()

        for subject in subjects:
            for ty in [0, 1]:
                #load_ty = (ty * -1) + 1
                for n in range(10):
                    print(subject + str(ty) + str(n))
                    moves = build_moves(ms[subject][ty][n])

                    # cut off last move
                    moves = moves[:len(moves)-1]

                    # select override
                    selects = 0
                    for move in moves:
                        print(move.end_sound)
                        if move.end_sound == "select":
                            selects += 1

                   

                    conf = classify_moves_prob(read_pickle_gz(args.model_path), moves)
                    print(conf)
                    cutoff = .5 # .4 ...
                    if conf > cutoff:
                        cls = 1
                    else:
                        cls = 0

                    
                    # select override
                    if cls == 1:
                        if selects > 0:
                            print("select override")
                            cls = 0
                            overrides = (overrides[0]+1, overrides[1])
                            if cls == ty:
                                print("override success")
                                overrides = (overrides[0], overrides[1]+1)

                    if cls == ty:
                        acc[ty] += 1
        
        print("password acc: " + str(acc[0]/(10 * len(subjects))))
        print("web searches acc: " + str(acc[1]/(10 * len(subjects))))
        print(acc[0] + acc[1])





    # 54, but with confidence cutoff adjustment
    if test == 55:
        print("test 55")

        subjects = ["B", "C", "D", "H"]
        #subjects = ["B"]
        acc = {}
        acc[0] = 0
        acc[1] = 0

        ms = build_gt_ms_dict()

        for subject in subjects:
            for ty in [0, 1]:
                #load_ty = (ty * -1) + 1
                for n in range(10):
                    print(subject + str(ty) + str(n))
                    moves = build_moves(ms[subject][ty][n])

                    # cut off last move
                    moves = moves[:len(moves)-1]

                   

                    conf = classify_moves_prob(read_pickle_gz(args.model_path), moves)
                    print(conf)
                    cutoff = .45 # .45 improves ws .625 -> .7
                    if conf > cutoff:
                        cls = 1
                    else:
                        cls = 0

                    if cls == ty:
                        acc[ty] += 1
        
        print("password acc: " + str(acc[0]/(10 * len(subjects))))
        print("web searches acc: " + str(acc[1]/(10 * len(subjects))))
        print(acc[0] + acc[1])




    # 51, but observed data
    if test == 54:
        print("test 54")

        subjects = ["B", "C", "D", "H"]
        #subjects = ["B"]
        acc = {}
        acc[0] = 0
        acc[1] = 0

        ms = build_gt_ms_dict()

        for subject in subjects:
            for ty in [0, 1]:
                #load_ty = (ty * -1) + 1
                for n in range(10):
                    print(subject + str(ty) + str(n))
                    moves = build_moves(ms[subject][ty][n])

                    # cut off last move
                    moves = moves[:len(moves)-1]

                   

                    cls = classify_moves(read_pickle_gz(args.model_path), moves)
                    print(cls)
                    if cls == ty:
                        acc[ty] += 1
        
        print("password acc: " + str(acc[0]/(10 * len(subjects))))
        print("web searches acc: " + str(acc[1]/(10 * len(subjects))))
        print(acc[0] + acc[1])



    # 51, but with added step for Select sounds
    if test == 53:
        print("test 53")

        subjects = ["A", "B", "C", "D", "H", "G"]
        #subjects = ["B"]
        fnames = {}
        fnames[0] = "samsung_passwords.json"
        fnames[1] = "web_searches.json"

        overrides = (0, 0)


        acc = {}
        acc[0] = 0
        acc[1] = 0

        for subject in subjects:
            for ty, fname in fnames.items():
                path = args.json_path + "/subject" + subject + "/" + fname
                with open(path) as f:
                    ht_dict = json.load(f)
                    print("len: " + str(len(ht_dict["move_sequences"])))
                    for ms in ht_dict["move_sequences"]:
                        moves = []

                        # cut off last move if its a select
                        if ms[len(ms)-1]["end_sound"] == "select": 
                            ms = ms[:len(ms)-1]

                        selects = 0

                        for move in ms:
                            if move["end_sound"] == "key_select" or move["end_sound"] == "select":
                                moves.append(move["num_moves"])
                                if move["end_sound"] == "select":
                                    selects += 1


                        move_sequence = [Move(num_moves=num_moves,
                                    end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in moves]
                        cls = classify_moves(read_pickle_gz(args.model_path), move_sequence)
                        print(cls)

                        if cls == 1:
                            if selects > 0:
                                print("select override")
                                cls = 0
                                overrides = (overrides[0]+1, overrides[1])
                                if cls == ty:
                                    print("override success")
                                    overrides = (overrides[0], overrides[1]+1)

                        if cls == ty:
                            acc[ty] += 1
                    f.close()
        
        print("password acc: " + str(acc[0]/(10 * len(subjects))))
        print("web searches acc: " + str(acc[1]/(10 * len(subjects))))
        print(overrides)
        print(acc[0] + acc[1])





    if test == 52:
        print("test 52")

        with open(args.json_path) as f:
            ht_dict = json.load(f)
            ms = ht_dict["move_sequences"][int(args.target)]
            for move in ms:
                print(str(move["num_moves"]) + ": " + move["end_sound"])
                
            f.close()


    if test == 51:
        print("test 51")

        #subjects = ["A", "B", "C", "D", "H", "G"]
        #subjects = ["B"]
        subjects = ["B", "C", "D", "H"]
        fnames = {}
        fnames[0] = "samsung_passwords.json"
        fnames[1] = "web_searches.json"


        acc = {}
        acc[0] = 0
        acc[1] = 0

        for subject in subjects:
            for ty, fname in fnames.items():
                path = args.json_path + "/subject" + subject + "/" + fname
                with open(path) as f:
                    ht_dict = json.load(f)
                    for ms in ht_dict["move_sequences"]:
                        moves = []

                        # cut off last move if its a select
                        if ms[len(ms)-1]["end_sound"] == "select": 
                            ms = ms[:len(ms)-1]

                        for move in ms:
                            if move["end_sound"] == "key_select" or move["end_sound"] == "select":
                                moves.append(move["num_moves"])
                        move_sequence = [Move(num_moves=num_moves,
                                    end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in moves]
                        cls = classify_moves(read_pickle_gz(args.model_path), move_sequence)
                        print(cls)
                        if cls == ty:
                            acc[ty] += 1
                    f.close()
        
        print("password acc: " + str(acc[0]/(10 * len(subjects))))
        print("web searches acc: " + str(acc[1]/(10 * len(subjects))))
        print(acc[0] + acc[1])


    if test == 50:
        print("test 50")
        with open(args.json_path) as f:
            ht_dict = json.load(f)
            for ms in ht_dict["move_sequences"]:
                moves = []
                for move in ms:
                    if move["end_sound"] == "key_select":
                        moves.append(move["num_moves"])
                move_sequence = [Move(num_moves=num_moves,
                            end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in moves]
                print(classify_moves(read_pickle_gz(args.model_path), move_sequence))
            f.close()









 
    if test == 41:
        print("test 41")
        val = [4, 6, 2, 2, 4]
        move_sequence = [Move(num_moves=num_moves,
                            end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in val]
        print(classify_moves(read_pickle_gz(args.model_path), move_sequence))



############### Statistical TESTs ###################

    # rewrite 30, but generalized
    if test == 31:
        print("test 31")
        mistakes = True
        dicts = "train"

        if dicts == "train":
            ms_dict_auto = build_ms_dict("word_dicts/train_english_auto.pkl.gz")
            ms_dict_non = build_ms_dict("word_dicts/train_english_non.pkl.gz")
            ms_dict_rockyou = build_ms_dict("word_dicts/train_rockyou.pkl.gz")
        else:
            ms_dict_auto = build_ms_dict("word_dicts/test_english_auto.pkl.gz")
            ms_dict_non = build_ms_dict("word_dicts/test_english_non.pkl.gz")
            ms_dict_rockyou = build_ms_dict("word_dicts/test_phpbb.pkl.gz")

        if mistakes:
            ms_dict_auto = add_mistakes_to_ms_dict(ms_dict_auto)
            ms_dict_non = add_mistakes_to_ms_dict(ms_dict_non)
            ms_dict_rockyou = add_mistakes_to_ms_dict(ms_dict_rockyou)

        weight = 3
        bins_transform = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6]
        
        
        hist_non = {}
        for word, moves in ms_dict_non.items():
            list_dist = make_row_dist(moves, range(10), weight)
            list_dist = transform_hist(list_dist, bins_transform)
            for b, w in enumerate(list_dist):
                if b in hist_non:
                    hist_non[b] += w
                else:
                    hist_non[b] = w

        for word, moves in ms_dict_rockyou.items():
            list_dist = make_row_dist(moves, range(10), weight)
            list_dist = transform_hist(list_dist, bins_transform)
            for b, w in enumerate(list_dist):
                if b in hist_non:
                    hist_non[b] += w
                else:
                    hist_non[b] = w

        hist_auto = {}
        for word, moves in ms_dict_auto.items():
            list_dist = make_row_dist(moves, range(10), weight)
            list_dist = transform_hist(list_dist, bins_transform)
            for b, w in enumerate(list_dist):
                if b in hist_auto:
                    hist_auto[b] += w
                else:
                    hist_auto[b] = w


        print("non data:")
        hist_non_sorted = sorted(hist_non.items(), reverse=True, key=lambda x:x[1])
        for m, times in hist_non_sorted:
            print("m: " + str(m) + ", uses: " + str(times))
        
        print("\nauto data:")
        hist_auto_sorted = sorted(hist_auto.items(), reverse=True, key=lambda x:x[1])
        for m, times in hist_auto_sorted:
            print("m: " + str(m) + ", uses: " + str(times))

        print("\nsubtracted data:")
        hist_subtracted = {}
        for m, times in hist_non.items():
            hist_subtracted[m] = times * len(ms_dict_auto)
        for m, times in hist_auto.items():
            if m not in hist_subtracted:
                hist_subtracted[m] = 0
            
            hist_subtracted[m] -= times * (len(ms_dict_non) + len(ms_dict_rockyou))
            hist_subtracted[m] = abs(hist_subtracted[m])

        hist_subtracted_sorted = sorted(hist_subtracted.items(), reverse=True, key=lambda x:x[1])
        for m, times in hist_subtracted_sorted:
            print("m: " + str(m) + ", uses: " + str(times))
        


    # This shows total uses of dif move lengths accross all words
    # but, I wonder what the odds of such move lengths in a given word are?
    if test == 30:
        print("test 30")

        ms_dict_auto = build_ms_dict("word_dicts/train_english_auto.pkl.gz")
        ms_dict_non = build_ms_dict("word_dicts/train_english_non.pkl.gz")
        ms_dict_rockyou = build_ms_dict("word_dicts/train_rockyou.pkl.gz")


        hist_non = {}
        for word, moves in ms_dict_non.items():
            for i, m in enumerate(moves):
                if m in hist_non:
                    hist_non[m] += move_weight(i, 3)
                else:
                    hist_non[m] = move_weight(i, 3)

        for word, moves in ms_dict_rockyou.items():
            for i, m in enumerate(moves):
                if m in hist_non:
                    hist_non[m] += move_weight(i, 3)
                else:
                    hist_non[m] = move_weight(i, 3)

        hist_auto = {}
        for word, moves in ms_dict_auto.items():
            for i, m in enumerate(moves):
                if m in hist_auto:
                    hist_auto[m] += move_weight(i, 3) * 2 #bc half the words
                else:
                    hist_auto[m] = move_weight(i, 3) * 2 #bc half the words


        print("non data:")
        hist_non_sorted = sorted(hist_non.items(), reverse=True, key=lambda x:x[1])
        for m, times in hist_non_sorted:
            print("m: " + str(m) + ", uses: " + str(times))
        
        print("\nauto data:")
        hist_auto_sorted = sorted(hist_auto.items(), reverse=True, key=lambda x:x[1])
        for m, times in hist_auto_sorted:
            print("m: " + str(m) + ", uses: " + str(times))

        print("\nsubtracted data:")
        hist_subtracted = hist_non
        for m, times in hist_auto.items():
            if m not in hist_subtracted:
                hist_subtracted[m] = 0
            
            hist_subtracted[m] -= times
            hist_subtracted[m] = abs(hist_subtracted[m])

        hist_subtracted_sorted = sorted(hist_subtracted.items(), reverse=True, key=lambda x:x[1])
        for m, times in hist_subtracted_sorted:
            print("m: " + str(m) + ", uses: " + str(times))
        









################### CLASSIFY TESTS #################

    # Classify Test (Bins) OVERFITTING
    if test == 4:
        print("test 4")

        model = read_pickle_gz(args.model_path)
        print("building test dicts")
        auto_words = grab_words(2000, args.words_path)
        non_words = grab_words(2000, args.words_path)
        ms_dict_auto_test = {}
        ms_dict_non_test = {}
        for word in auto_words:
            ms_dict_auto_test[word] = simulate_ms(englishDictionary, args.ss_path, word, True)
        for word in non_words:
            ms_dict_non_test[word] = simulate_ms(englishDictionary, args.ss_path, word, False)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 2000)

        print("test dicts built")
        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        ac = .5
        nc = .5
        peak = 30
        size = 1
        mistakes = 0
        
        transforms = get_transforms(1)
        bins = 10



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
        for key, val in ms_dict_rockyou_test.items():
            pred = classify_ms_with_msfd_full_new(model, msfd, db, add_mistakes(val, mistakes), bins_transform=transforms[bins], weight=3,
                peak=peak, auto_cutoff=ac, non_cutoff=nc)[0]
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

        # accuracy and f1 for english and phpbb sets
        acc_english = accuracy_score(gt[0], preds[0])
        f1_english = f1_score(gt[0], preds[0])
        acc_phpbb = accuracy_score(gt[1], preds[1])
        f1_phpbb = f1_score(gt[1], preds[1])
        

        print("(english) acc: " + str(acc_english)
            + "; f1: " + str(f1_english) + "\n")
        print("(rockyou) acc: " + str(acc_phpbb)
            + "; f1: " + str(f1_phpbb) + "\n\n")

 



    # Classify Test (OH)
    if test == 3:
        print("test 15")

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

        steps = ["1step", "2step"]
        step_dict = {}
        step_dict["1step"] = (.5, .5)
        step_dict["2step"] = (.26, .32)
        mts = ["mistakes", "nomistakes"]
        #mt = "nomistakes"
    
        mistakes_list = [0, 1, 2, 3]

        maxs = [2, 3, 4, 5]

        for max in maxs:

            for mt in mts:
                model = read_pickle_gz(args.model_path + mt + "_oh_" + str(max) + "_sim.pkl.gz")
                for step in steps:
                    ac = step_dict[step][0]
                    nc = step_dict[step][1]

                    if args.save_path is not None:
                        #dec/size2
                        full_path = args.save_path + "/" + "oh" + str(max) + "/"
                        full_path += mt + "_" + step

                    english_accs = []
                    english_f1s = []
                    password_accs = []
                    password_f1s = []

                    for mistakes in mistakes_list:

                        results = {}
                        print("classifying autos")
                        for key, val in ms_dict_auto_test.items():
                            pred = classify_ms_with_msfd_full_oh(model, msfd, db, add_mistakes(val, mistakes),
                                peak=peak, auto_cutoff=ac, non_cutoff=nc, max_len=max*10, max=max)[0]
                            results[(key, "auto")] = pred
                        print("classifying nons")
                        for key, val in ms_dict_non_test.items():
                            pred = classify_ms_with_msfd_full_oh(model, msfd, db, add_mistakes(val, mistakes),
                                peak=peak, auto_cutoff=ac, non_cutoff=nc, max_len=max*10, max=max)[0]
                            results[(key, "non")] = pred
                        print("classifying phpbbs")
                        for key, val in ms_dict_phpbb_test.items():
                            pred = classify_ms_with_msfd_full_oh(model, msfd, db, add_mistakes(val, mistakes),
                                peak=peak, auto_cutoff=ac, non_cutoff=nc, max_len=max*10, max=max)[0]
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

    # Classify Test (List)
    if test == 2:
        print("test 10")

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

        steps = ["1step", "2step"]
        step_dict = {}
        step_dict["1step"] = (.5, .5)
        step_dict["2step"] = (.26, .32)
        mts = ["mistakes", "nomistakes"]
        #mt = "nomistakes"
    
        mistakes_list = [0, 1, 2, 3]

        max_len = 10

        for mt in mts:
            model = read_pickle_gz(args.model_path + mt + "_list_10_sim.pkl.gz")
            for step in steps:
                ac = step_dict[step][0]
                nc = step_dict[step][1]

                if args.save_path is not None:
                    #dec/size2
                    full_path = args.save_path + "/"
                    full_path += mt + "_" + step

                english_accs = []
                english_f1s = []
                password_accs = []
                password_f1s = []

                for mistakes in mistakes_list:

                    results = {}
                    print("classifying autos")
                    for key, val in ms_dict_auto_test.items():
                        pred = classify_ms_with_msfd_full_list(model, msfd, db, add_mistakes(val, mistakes),
                            peak=peak, auto_cutoff=ac, non_cutoff=nc, max_len=max_len)[0]
                        results[(key, "auto")] = pred
                    print("classifying nons")
                    for key, val in ms_dict_non_test.items():
                        pred = classify_ms_with_msfd_full_list(model, msfd, db, add_mistakes(val, mistakes),
                            peak=peak, auto_cutoff=ac, non_cutoff=nc, max_len=max_len)[0]
                        results[(key, "non")] = pred
                    print("classifying phpbbs")
                    for key, val in ms_dict_phpbb_test.items():
                        pred = classify_ms_with_msfd_full_list(model, msfd, db, add_mistakes(val, mistakes),
                            peak=peak, auto_cutoff=ac, non_cutoff=nc, max_len=max_len)[0]
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

    # Classify Test (Bins)
    if test == 1:
        print("test 1")

        #model = read_pickle_gz(args.model_path)
        print("building test dicts")
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_ms_dict(args.ms_path_rockyou, 100, 2000)
        ms_dict_phpbb_test = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", 500)
        print("test dicts built")
        db_path = "rockyou-samsung-updated.db"
        db = PasswordRainbow(db_path)

        #ac = .26
        #nc = .32
        #ac = .5
        #nc = .5
        peak = 30
        bins_nums = [7]
        transforms = {}
        transforms[7] = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6]

        steps = ["1step"]
        step_dict = {}
        step_dict["1step"] = (.5, .5)
        mts = ["mistakes"]
    
        mistakes_list = [0]

        mds = [0]

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

                        #if args.save_path is not None:
                            #dec/size2

                            # full_path = args.save_path + "/" + str(bins) + "bins/"
                            # full_path += mt + "_" + step

                            # full_path = args.save_path + "/md" + str(md) + "/bins/size1"
                            # full_path += "/" + str(bins) + "bins/"
                            # full_path += mt + "_" + step
                        full_path = "suggestions_model/nowtest.txt"

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

                        with open(full_path, "w") as f:
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


    # build and save model
    elif test == -1:
        count = 2000
        max_depth = 1


        print("test -1")
        if args.save_path is None:
            print("no save path")
        else:
            
            auto_words = grab_words(count, args.words_path)
            non_words = grab_words(count, args.words_path)
            ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, count)
            model = build_model_sim_list(ms_dict_rockyou=ms_dict_rockyou_train, max_len=10,
                                        englishDictionary=englishDictionary,
                                        ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, mistakes=False,
                                        max_depth = max_depth)
            save_model(args.save_path + "_list_10_sim.pkl.gz", model)

            print("model saved: list 10")

    # build and save model OH
    elif test == -2:
        print("test -2")
        if args.save_path is None:
            print("no save path")

        count = 2000
        max_depths = [2, 3, 4]

        maxs = [2, 3, 4, 5]

        auto_words = grab_words(count, args.words_path)
        non_words = grab_words(count, args.words_path)
        ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, count)
        ms_dict_rockyou_train_mistakes = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, count))

        for max_depth in max_depths:
            for mx in maxs:
                save_p = args.save_path + "/md" + str(max_depth)
                save_p += "/oh"


                
                model = build_model_sim_oh(ms_dict_rockyou=ms_dict_rockyou_train_mistakes, max_len=mx*10, max=mx,
                                            englishDictionary=englishDictionary,
                                            ss_path=args.ss_path,
                                            words_auto=auto_words, words_non=non_words,
                                            include_rockyou=True, mistakes=True,
                                            max_depth = max_depth)
                save_model(save_p + "/model_mistakes_oh_" + str(mx) + "_sim.pkl.gz", model)

                print("model saved: oh")

                model = build_model_sim_oh(ms_dict_rockyou=ms_dict_rockyou_train, max_len=mx*10, max=mx,
                                            englishDictionary=englishDictionary,
                                            ss_path=args.ss_path,
                                            words_auto=auto_words, words_non=non_words,
                                            include_rockyou=True, mistakes=False,
                                            max_depth = max_depth)
                save_model(save_p + "/model_nomistakes_oh_" + str(mx) + "_sim.pkl.gz", model)

                print("model saved: oh")

    # build and save model ADA
    elif test == -3:
        count = 2000
        max_depth = 3




        print("test -3")
        if args.save_path is None:
            print("no save path")
        else:
            
            auto_words = grab_words(count, args.words_path)
            non_words = grab_words(count, args.words_path)
            # ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, count)
            ms_dict_rockyou_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, count))
            model = build_model_sim_list_ada(ms_dict_rockyou=ms_dict_rockyou_train, max_len=10,
                                        englishDictionary=englishDictionary,
                                        ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, mistakes=True,
                                        max_depth = max_depth)
            save_model(args.save_path + "_mistakes_list_10_sim.pkl.gz", model)

            print("model saved: ada (mistakes)")

            auto_words = grab_words(count, args.words_path)
            non_words = grab_words(count, args.words_path)
            ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, count)
            # ms_dict_rockyou_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, count))
            model = build_model_sim_list_ada(ms_dict_rockyou=ms_dict_rockyou_train, max_len=10,
                                        englishDictionary=englishDictionary,
                                        ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, mistakes=False,
                                        max_depth = max_depth)
            save_model(args.save_path + "_nomistakes_list_10_sim.pkl.gz", model)

            print("model saved: ada")

   # build and save model ADA Bins
    elif test == -4:
        print("test -4")

        count = 2000
        # bs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        # bs = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        #     [0, 1, 2, 3, 3, 3, 3, 3, 3, 3],
        #     [0, 1, 2, 3, 4, 4, 4, 4, 4, 4],
        #     [0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
        #     [0, 1, 2, 3, 4, 5, 6, 6, 6, 6],
        #     [0, 1, 2, 3, 4, 5, 6, 7, 7, 7],
        #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]]
        # ***
        bs = [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1,],
            [0, 0, 1, 1, 2, 2, 2, 2, 2, 2],
            [0, 0, 1, 1, 2, 2, 3, 3, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]]
        max_depth = 1
        alg="SAMME"


        if args.save_path is None:
            print("no save path")
        else:
   

            auto_words = grab_words(2000, args.words_path)
            non_words = grab_words(2000, args.words_path)
            ms_dict_rockyou_train = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, 2000))

            for b in bs:
                
                model = build_model_sim_new_ada(ms_dict_rockyou=ms_dict_rockyou_train,
                                            englishDictionary=englishDictionary,
                                            ss_path=args.ss_path,
                                            words_auto=auto_words, words_non=non_words,
                                            include_rockyou=True, bin_transform=b, weight=3,
                                            mistakes=True, max_depth=max_depth, alg=alg)
                save_model(args.save_path + str(max(b) + 1) + "bins_mistakes_sim.pkl.gz", model)

                print("model saved: " + str(max(b) + 1))

            
            
            # nomistakes

            ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, 2000)

            for b in bs:
                
                model = build_model_sim_new_ada(ms_dict_rockyou=ms_dict_rockyou_train,
                                            englishDictionary=englishDictionary,
                                            ss_path=args.ss_path,
                                            words_auto=auto_words, words_non=non_words,
                                            include_rockyou=True, bin_transform=b, weight=3,
                                            mistakes=False, max_depth=max_depth, alg=alg)
                save_model(args.save_path + str(max(b) + 1) + "bins_nomistakes_sim.pkl.gz", model)

                print("model saved: " + str(max(b) + 1))

    # build and save model OH ADA
    elif test == -5:
        print("test -5")
        if args.save_path is None:
            print("no save path")

        count = 2000

        maxs = [2, 3, 4, 5]

        auto_words = grab_words(count, args.words_path)
        non_words = grab_words(count, args.words_path)
        ms_dict_rockyou_train = build_ms_dict(args.ms_path_rockyou, count)
        ms_dict_rockyou_train_mistakes = add_mistakes_to_ms_dict(build_ms_dict(args.ms_path_rockyou, count))

        for mx in maxs:
            save_p = args.save_path + "/oh"


            
            model = build_model_sim_oh_ada(ms_dict_rockyou=ms_dict_rockyou_train_mistakes, max_len=mx*10, max=mx,
                                        englishDictionary=englishDictionary,
                                        ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, mistakes=True)
            save_model(save_p + "/model_mistakes_oh_" + str(mx) + "_sim.pkl.gz", model)

            print("model saved: oh")

            model = build_model_sim_oh_ada(ms_dict_rockyou=ms_dict_rockyou_train, max_len=mx*10, max=mx,
                                        englishDictionary=englishDictionary,
                                        ss_path=args.ss_path,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, mistakes=False)
            save_model(save_p + "/model_nomistakes_oh_" + str(mx) + "_sim.pkl.gz", model)

            print("model saved: oh")



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



################### WALK THROUGH ################

    # write to count the times each feature is used
    if test == 12:
        model = read_pickle_gz(args.model_path)
        print("estimators: " + str(len(model.estimators_)))
        feature_used = {}
        depths = []
        for i in range(10):
            feature_used[i] = 0
        
        
        for i, clf in enumerate(model.estimators_):
            if i > 1000:
                break

            # print("estimator: " + str(i))

            depths.append(clf.get_depth())

            n_nodes = clf.tree_.node_count
            children_left = clf.tree_.children_left
            children_right = clf.tree_.children_right
            feature = clf.tree_.feature
            threshold = clf.tree_.threshold

            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, depth = stack.pop()
                node_depth[node_id] = depth

                # If the left and right child of a node is not the same we have a split
                # node
                is_split_node = children_left[node_id] != children_right[node_id]
                # If a split node, append left and right children and depth to `stack`
                # so we can loop through them
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True

            # print(
            #     "The binary tree structure has {n} nodes and has "
            #     "the following tree structure:\n".format(n=n_nodes)
            # )
            for i in range(n_nodes):
                if is_leaves[i]:
                    continue
                    # print(
                    #     "{space}node={node} is a leaf node.".format(
                    #         space=node_depth[i] * "\t", node=i
                    #     )
                    # )
                else:
                    # print(
                    #     "{space}node={node} is a split node: "
                    #     "go to node {left} if X[:, {feature}] <= {threshold} "
                    #     "else to node {right}.".format(
                    #         space=node_depth[i] * "\t",
                    #         node=i,
                    #         left=children_left[i],
                    #         feature=feature[i],
                    #         threshold=threshold[i],
                    #         right=children_right[i],
                    #     )
                    # )

                    feature_used[feature[i]] += 1

        av = sum(depths) / len(depths)
        print("average depth: " + str(av))
        stdev = 0
        for depth in depths:
            stdev += (depth - av) * (depth - av)
        stdev = math.sqrt(stdev / len(depths))
        print("stdev depth: " + str(stdev))
        for feature, times in feature_used.items():
            print("used " + str(feature) + " " + str(times) + " times.")

    # Walk Through RF Model
    if test == 11:
        model = read_pickle_gz(args.model_path)
        print("estimators: " + str(len(model.estimators_)))

        
        for i, clf in enumerate(model.estimators_):
            if i > 0:
                break

            print("estimator: " + str(i))


            n_nodes = clf.tree_.node_count
            children_left = clf.tree_.children_left
            children_right = clf.tree_.children_right
            feature = clf.tree_.feature
            threshold = clf.tree_.threshold

            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, depth = stack.pop()
                node_depth[node_id] = depth

                # If the left and right child of a node is not the same we have a split
                # node
                is_split_node = children_left[node_id] != children_right[node_id]
                # If a split node, append left and right children and depth to `stack`
                # so we can loop through them
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True

            print(
                "The binary tree structure has {n} nodes and has "
                "the following tree structure:\n".format(n=n_nodes)
            )
            for i in range(n_nodes):
                if is_leaves[i]:
                    print(
                        "{space}node={node} is a leaf node.".format(
                            space=node_depth[i] * "\t", node=i
                        )
                    )
                else:
                    print(
                        "{space}node={node} is a split node: "
                        "go to node {left} if X[:, {feature}] <= {threshold} "
                        "else to node {right}.".format(
                            space=node_depth[i] * "\t",
                            node=i,
                            left=children_left[i],
                            feature=feature[i],
                            threshold=threshold[i],
                            right=children_right[i],
                        )
                    )



############################ ??? ###################


    ## ?? Regarding testing with recovery
    if test == 10:

        
        keyboard_type = KeyboardType.SAMSUNG
        graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
        count = 1000
        pcount = 0
        dictionary = englishDictionary
        tv_type = SmartTVType.SAMSUNG

        dictionary.set_characters(graph.get_characters())

        
            
        ms_dict_phpbb = build_ms_dict("suggestions_model/local/ms_dict_phpbb.pkl.gz", count)
        phpbb_rec_dict = {}
        # key: word, val: (ms, rank_sug, rank_standard)

        if args.progress == "load":
            meta = read_pickle_gz(args.progress_path)
            count = meta[0]
            ms_dict_phpbb = meta[1]
            pcount = meta[2]
            phpbb_rec_dict = meta[3]

        done = 0
        for key, val in ms_dict_phpbb.items():
            print("key: " + key)

            if args.progress == "load":
                if (done < pcount):
                    done += 1
                    continue

            move_sequence = [Move(num_moves=num_moves,
                            end_sound=SAMSUNG_KEY_SELECT, directions=Direction.ANY) for num_moves in val]


            ranked_candidates = get_words_from_moves_suggestions(
                move_sequence=move_sequence, graph=graph, dictionary=dictionary,
                did_use_autocomplete=False, max_num_results=50)
            sug_res = (-1, -1)
            for rank, (guess, score, num_candidates) in enumerate(ranked_candidates):
                if guess == key:
                    sug_res = (rank, score)
                    break

            ranked_candidates = get_words_from_moves(
            move_sequence=move_sequence, graph=graph, dictionary=dictionary,
            tv_type=tv_type,  max_num_results=50, precomputed=None,
            includes_done=False, start_key="q", is_searching_reverse=False)
            non_res = (-1, -1)
            for rank, (guess, score, num_candidates) in enumerate(ranked_candidates):
                if guess == key:
                    non_res = (rank, score)
                    break

            phpbb_rec_dict[key] = (val, sug_res, non_res)
            done += 1

            if args.progress is not None:
                if done % 4 == 0:
                    print("done with " + str(done))
                    meta = (count, ms_dict_phpbb, done, phpbb_rec_dict)
                    save_pickle_gz(meta, args.progress_path)

        if args.save_path is not None:
            save_pickle_gz(phpbb_rec_dict, args.save_path)

