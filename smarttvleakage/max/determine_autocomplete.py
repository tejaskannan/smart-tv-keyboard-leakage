from enum import auto
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from argparse import ArgumentParser

from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV

from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION

from smarttvleakage.max.alg_determine_autocomplete import get_score_from_ms, get_score_from_ms_improved, adjust_for_len
from smarttvleakage.max.manual_score_dict import build_msfd, build_ms_dict, buildDict
from smarttvleakage.max.simulate_ms import grab_words, simulate_ms
import math

from smarttvleakage.utils.file_utils import save_pickle_gz





# Make Data Structures

# Distance Histogram
# [Int] -> [String]
def make_column_titles_dist(bins : List[int]) -> List[str]:
    if bins == []:
        return []
    return list(map(lambda n: "d " + n, map(str, bins)))

# [Int], [Int], Int -> (Dict: Int -> Int)
def moves_to_hist_dist(moves : List[int], bins : List[int], weighted : int) -> dict[int, int]:
    # make empty hist
    
    hist = {}
    for bin in bins:
        hist[bin] = 0
    
    for i in range(len(moves)):
        if (moves[i] in bins):
            if weighted == 0: #unw/nn
                hist[moves[i]] = hist[moves[i]] + 1
            elif weighted == 1: #w/nn
                hist[moves[i]] = hist[moves[i]] + i
            elif weighted == 2: #unw
                hist[moves[i]] = hist[moves[i]] + 1
            elif weighted == 3: #w
                hist[moves[i]] = hist[moves[i]] + i
            elif weighted == 4: #dw/nn
                hist[moves[i]] = hist[moves[i]] + i*i
            elif weighted == 5: #dw
                hist[moves[i]] = hist[moves[i]] + i*i
        else:
            if weighted == 0:
                hist[bins[len(bins) - 1]] = hist[bins[len(bins) - 1]] + 1
            elif weighted == 1:
                hist[bins[len(bins) - 1]] = hist[bins[len(bins) - 1]] + i
            elif weighted == 2:
                hist[bins[len(bins) - 1]] = hist[bins[len(bins) - 1]] + 1
            elif weighted == 3:
                hist[bins[len(bins) - 1]] = hist[bins[len(bins) - 1]] + i
            elif weighted == 4:
                hist[bins[len(bins) - 1]] = hist[bins[len(bins) - 1]] + i*i
            elif weighted == 5:
                hist[bins[len(bins) - 1]] = hist[bins[len(bins) - 1]] + i*i
    for bin in bins:
        if weighted == 0:
            hist[bin] = hist[bin] / len(moves)
        elif weighted == 1:
            total_points = (len(moves) * (len(moves) + 1)) / 2
            hist[bin] = hist[bin] / max(1, total_points)
        elif weighted == 2:
            hist[bin] = hist[bin]
        elif weighted == 3:
            hist[bin] = hist[bin]
        elif weighted == 4:
            total_points = 0
            for i in range(len(moves)):
                total_points = total_points + i*i
            hist[bin] = hist[bin] / max(1, total_points)
        elif weighted == 5:
            hist[bin] = hist[bin]
    return hist




# Turns a histogram dictionary into a list for features
# (Dict: Int -> Int), [Int] -> [Int]
def hist_to_list(hist : dict[int, int], bins : int) -> List[int]:
    lst = []
    for bin in bins:
        lst.append(hist[bin])
    return lst


# [Int], _, [Int], Int -> [Int]
def make_row_dist(moves : List[int], bins : List[int], weighted : int) -> List[int]:
    if bins == []:
        return []
    return hist_to_list(moves_to_hist_dist(moves, bins, weighted), bins)


# [Int], Int -> DF
def make_df(bins_dist : List[int], weighted : int, ms_dict_auto : dict[str, list[int]], ms_dict_non : dict[str, list[int]]):


    data = np.empty((0, len(bins_dist) + 2), dtype=float)

    auto_list = []
    for key in ms_dict_auto:
        auto_list.append(ms_dict_auto[key])
    non_list = []
    for key in ms_dict_non:
        non_list.append(ms_dict_non[key])
    
    id = 0
    for moves in auto_list:
        list_dist = make_row_dist(moves, bins_dist, weighted)
        new_row = np.array([[id] + [1] + list_dist], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1

    for moves in non_list:
        list_dist = make_row_dist(moves, bins_dist, weighted)
        new_row = np.array([[id] + [0] + list_dist], dtype=float)
        data = np.append(data, new_row, axis=0)
        id = id + 1

    column_titles = ["id"] + ["ac"] + make_column_titles_dist(bins_dist)
    df = pd.DataFrame(data = data, columns = column_titles)
    print(df)
    return df




# Takes an ID and returns the corresponding (word, ty)
# Int -> (String, String)
def id_to_real(id : int, ms_dict_auto : dict[str, list[int]], ms_dict_non : dict[str, list[int]]) -> tuple[str, str]:

    full_list = []
    for key in ms_dict_auto:
        full_list.append((key, "auto"))
    for key in ms_dict_non:
        full_list.append((key, "non"))

    return full_list[id]



## GET SCORES
def get_scores(df, model, seed : int):
    #df.drop(["id"], axis=1, inplace=True)
    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    X = X.drop(["id"], axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=seed)
    model.fit(X_train, y_train)

    pred=model.predict(X_test)
    return (accuracy_score(y_test, pred), f1_score(y_test, pred))


def get_pred(df, model, seed : int):
    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=seed)
    # Trains without ID
    model.fit(X_train.drop(["id"], axis=1, inplace=False), y_train)

    pred=model.predict(X_test.drop(["id"], axis=1, inplace=False))
    #print(zip(X_test["id"], y_test, pred))
    return zip(X_test["id"], y_test, pred)

# DF, Model, Int -> [(Int, Int, (Float, Float))]
# The returned list has structure [(ID, Ty, (chance of "non", chance of "auto"))]
def get_pred_probas(df, model, seed : int) -> List[tuple[int, str, tuple[float, float]]]:
    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=seed)
    # Trains without ID
    model.fit(X_train.drop(["id"], axis=1, inplace=False), y_train)

    pred_probas=model.predict_proba(X_test.drop(["id"], axis=1, inplace=False))
    return zip(X_test["id"], y_test, pred_probas)



def get_word_results(df, model, iters):

    results_dict = {}
    for i in range(iters):
        pred = get_pred(df, model, i)
        
        for (id, gt, res) in pred:
            (word, x) = id_to_real(int(id), build_ms_dict("auto"), build_ms_dict("non"))
            if (word, x) in results_dict:
                (correct, total) = results_dict[(word, x)]
                if gt == res:
                    results_dict[(word, x)] = (correct + 1, total + 1)
                else:
                    results_dict[(word, x)] = (correct, total + 1)
            else:
                if gt == res:
                    results_dict[(word, x)] = (1, 1)
                else:
                    results_dict[(word, x)] = (0, 1)
        
    return results_dict


# Returns a dictionary mapping (word, ty) to a sum of prediction chances and a total number of predictions
# DF, Model, Int -> (Dict: (String, String) -> (Float, Float))
def get_word_results_proba(df, model, iters : int, ms_dict_auto : dict[str, list[int]], ms_dict_non : dict[str, list[int]]) -> dict[tuple[str, str], tuple[float, float]]:

    results_dict = {}
    for i in range(iters):
        pred_probas = get_pred_probas(df, model, i)
        
        for (id, gt, res) in pred_probas:
            
            (word, x) = id_to_real(int(id), ms_dict_auto, ms_dict_non)
            if (word, x) in results_dict:
                (correct, total) = results_dict[(word, x)]
                if gt == 0:
                    results_dict[(word, x)] = (correct + res[0], total + 1)
                else:
                    results_dict[(word, x)] = (correct + res[1], total + 1)
            else:
                if gt == 0:
                    results_dict[(word, x)] = (res[0], 1)
                else:
                    results_dict[(word, x)] = (res[1], 1)
        
    return results_dict







def test_config(df, model, iters : int):
    accs = []
    f1s = []
    for i in range(iters):
        acc, f1 = get_scores(df, model, i)
        accs.append(acc)
        f1s.append(f1)
    acc = sum(accs) / len(accs)
    f1 = sum(f1s) / len(f1s)
    print("acc: " + str(sum(accs) / len(accs)))
    print("f1s: " + str(sum(f1s) / len(f1s)))
    return (acc, f1)

def id_to_weight(id : int) -> str:
    if id == 0:
        return "unweighted, normalized"
    elif id == 1:
        return "weighted, normalized"
    elif id == 2:
        return "unweighted, not normalized"
    elif id == 3:
        return "weighted, not normalized"
    elif id == 4:
        return "double weighted, normalized"
    elif id == 5:
        return "double weighted, not normalized"

# 0: unw, 1: w, 2: unw/nn, 3: w/nn, 4: dw, 5: dw/nn
def eval_weighting(max_bins_dist : int, iters : int):

    ms_dict_auto = build_ms_dict("auto")
    ms_dict_non = build_ms_dict("non")

    data = np.empty((0, 6), dtype=float)
    for i in range(max_bins_dist):
        bins_dist = range(i + 1)
        row = []
       
        df = make_df(bins_dist, 0, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("Testing dist: " + str(i+1) + ", unweighted")
        accRFC, _ = test_config(df, model, iters)
        row.append(accRFC)

        df = make_df(bins_dist, 1, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("Testing dist: " + str(i+1) + ", weighted")
        accRFC, _ = test_config(df, model, iters)
        row.append(accRFC)

        df = make_df(bins_dist, 2, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("Testing dist: " + str(i+1) + ", unweighted, no norm")
        accRFC, _ = test_config(df, model, iters)
        row.append(accRFC)

        df = make_df(bins_dist, 3, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("Testing dist: " + str(i+1) + ", weighted, no norm")
        accRFC, _ = test_config(df, model, iters)
        row.append(accRFC)

        df = make_df(bins_dist, 4, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("Testing dist: " + str(i+1) + ", dub weighted")
        accRFC, _ = test_config(df, model, iters)
        row.append(accRFC)

        df = make_df(bins_dist, 5, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("Testing dist: " + str(i+1) + ", dub weighted, no norm")
        accRFC, _ = test_config(df, model, iters)
        row.append(accRFC)
        
        new_row = np.array([row], dtype=float)
        data = np.append(data, new_row, axis=0)

    df = pd.DataFrame(data = data, columns = ["unw", "w", "unw/nn", "w/nn", "dw", "dw/nn"])
    df.index = list(map(lambda n: n + 1, range(max_bins_dist)))
    print(df)
    return df



# Takes a result dict from the ML method (get_word_results_proba), 
# and returns a result dictionary with a method selection and results from each method
# dict (word, ty) -> (accuracy sum, total), float ->
# dict (word, ty) -> (method, ml_certainty, manual_score)
def combine_algorithmic(results_dict : dict[tuple[str, str], 
                        tuple[float, int]], 
                        certainty_cutoff : float,
                        ms_dict_auto : dict[str, list[int]],
                        ms_dict_non : dict[str, list[int]]) -> dict[tuple[str, str], tuple[str, float, float]]:
    print("combine_algorithmic")

    msfd = build_msfd()

    
    final_dict = {} # (method, certainty, manual_score)
    for key in results_dict:
        (correct, total) = results_dict[key]

        accuracy = correct / total

        if key[1] == "non":
            ms = ms_dict_non[key[0]]
        else:
            ms = ms_dict_auto[key[0]]

        ms_string = ""
        for m in ms:
            ms_string += str(m) + ","
        
        print(ms_string)
        if ms_string in msfd:
            manual_score = adjust_for_len(msfd[ms_string][1], key[0], 1) # [(guess, score)]
        else:
            manual_score = 0

        if accuracy > (1-certainty_cutoff) or accuracy < certainty_cutoff:
            final_dict[key] = ("ml", accuracy, manual_score)
        else:
            final_dict[key] = ("alg", accuracy, manual_score)
    return final_dict

# Takes in a list of bins, weighting types, certainty cutoffs, and a length limiter (or -1)
# returns a dict from paramter config to results
# (bins, weight, certainty_cutoff) -> 
# (dict: (word, ty) -> (method, ml_score, manual_score))
def eval_params(bins_list : List[int],
                 weights : List[int], 
                 certainty_cutoffs : List[float], 
                 max_length : int, 
                 iters : int) -> dict[tuple[int, int, float], dict[tuple[str, str], tuple[str, float, float]]]:

    ms_dict_auto = build_ms_dict("auto")
    ms_dict_non = build_ms_dict("non")

    # Limit word length for runtime
    if max_length > 0:
        to_remove = []
        for key in ms_dict_auto:
            if len(key) > max_length:
                to_remove.append(key)
        for key in to_remove:
            del ms_dict_auto[key]
        to_remove = []
        for key in ms_dict_non:
            if len(key) > max_length:
                to_remove.append(key)
        for key in to_remove:
            del ms_dict_non[key]

    # print word count
    nons = 0
    autos = 0
    for key in ms_dict_auto:
        autos = autos + 1
    for key in ms_dict_non:
        nons = nons + 1
    print("autos: " + str(autos) + ", nons: " + str(nons))


    param_dict = {}
    for bins in bins_list:
        print("eval new bins")
        for weight in weights:
            print("eval new weight: " + id_to_weight(weight))

            df = make_df(range(bins), weight, ms_dict_auto, ms_dict_non)
            model = RandomForestClassifier()
            results_dict = get_word_results_proba(df, model, iters, ms_dict_auto, ms_dict_non) # dict (word, ty) -> (accuracy sum, total)

            for certainty_cutoff in certainty_cutoffs:
                param_dict[(bins, weight, certainty_cutoff)] = {}

                final_dict = combine_algorithmic(results_dict, certainty_cutoff, ms_dict_auto, ms_dict_non)
                for key in final_dict:
                    param_dict[(bins, weight, certainty_cutoff)][key] = final_dict[key]

    return param_dict


# Same as above, but uses simulates move sequences
# (bins, weight, certainty_cutoff) -> 
# (dict: (word, ty) -> (method, ml_score, manual_score))
def eval_params_sim(bins_list : List[int],
                 weights : List[int], 
                 certainty_cutoffs : List[float], 
                 max_length : int, 
                 iters : int) -> dict[tuple[int, int, float], dict[tuple[str, str], tuple[str, float, float]]]:


    englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")
    wcs = buildDict(100)
    words = grab_words(5000)
    print("words grabbed")
    ms_dict_auto = {}
    ms_dict_non = {}
    for word in words:
        ms_dict_auto[word] = simulate_ms(englishDictionary, wcs, word, True, 1)
        ms_dict_non[word] = simulate_ms(englishDictionary, wcs, word, False, 1)

    print("ms_dicts formed")

    # Limit word length for runtime
    if max_length > 0:
        to_remove = []
        for key in ms_dict_auto:
            if len(key) > max_length:
                to_remove.append(key)
        for key in to_remove:
            del ms_dict_auto[key]
        to_remove = []
        for key in ms_dict_non:
            if len(key) > max_length:
                to_remove.append(key)
        for key in to_remove:
            del ms_dict_non[key]

    # print word count
    nons = 0
    autos = 0
    for key in ms_dict_auto:
        autos = autos + 1
    for key in ms_dict_non:
        nons = nons + 1
    print("autos: " + str(autos) + ", nons: " + str(nons))


    param_dict = {}
    for bins in bins_list:
        print("eval new bins")
        for weight in weights:
            print("eval new weight: " + id_to_weight(weight))

            df = make_df(range(bins), weight, ms_dict_auto, ms_dict_non)
            model = RandomForestClassifier()
            results_dict = get_word_results_proba(df, model, iters, ms_dict_auto, ms_dict_non) # dict (word, ty) -> (accuracy sum, total)

            for certainty_cutoff in certainty_cutoffs:
                param_dict[(bins, weight, certainty_cutoff)] = {}

                final_dict = combine_algorithmic(results_dict, certainty_cutoff, ms_dict_auto, ms_dict_non)
                for key in final_dict:
                    param_dict[(bins, weight, certainty_cutoff)][key] = final_dict[key]

    return param_dict



# Takes a combined result dictionary and prints relevant analysis
# (dict: (word, ty) -> (method, ml_score, manual_score)) ->
def analyze_params(final_dict : dict[tuple[str, str], tuple[str, float, float]], 
                    manual_score_cutoff : float) -> float:
    ml_right = (0, 0) # used, unused
    alg_right = 0
    right = 0
    total = (0, 0)

    alg_fixes = 0
    alg_losses = 0

    for key in final_dict:
        word = key[0]
        ty = key[1]
        method, ml_score, manual_score = final_dict[key]

        if method == "ml":
            total = (total[0]+1, total[1])
            if ml_score > .5:
                ml_right = (ml_right[0]+1, ml_right[1])            
                right += 1

        elif method == "alg":
            total = (total[0], total[1]+1)
            if ty == "non":
                if manual_score >= manual_score_cutoff:
                    alg_right += 1
                    right += 1
                    if ml_score < .5:
                        alg_fixes += 1
                    else:
                        ml_right = (ml_right[0], ml_right[1]+1) 
                else:
                    if ml_score > .5:
                        alg_losses += 1
                        ml_right = (ml_right[0], ml_right[1]+1) 
            elif ty == "auto":
                if manual_score < manual_score_cutoff:
                    alg_right += 1
                    right += 1
                    if ml_score < .5:
                        alg_fixes += 1
                    else:
                        ml_right = (ml_right[0], ml_right[1]+1) 
                else:
                    if ml_score > .5:
                        alg_losses += 1
                        ml_right = (ml_right[0], ml_right[1]+1) 


    print("total words: " + str((total[0]+total[1])))
    print("ml_right: " + str(ml_right[0]))
    print("alg_right: " + str(alg_right))
    print("losses: " + str(alg_losses), end="; ")
    print("fixes: " + str(alg_fixes))
    print("ml_accuracy: " + str((ml_right[0]+ml_right[1]) / (total[0]+total[1])), end="; ")
    print("final accuracy: " + str(right / (total[0]+total[1])))
    return (right / (total[0]+total[1]))

# something is off w/ this
# works mostly but not w .0001, e.g.
# well that's the only one I've found so far
# but prob bc the split function
# fix this tomorrow
def normalize_manual_score(manual_score : float) -> float:
    if manual_score <= 0:
        return 0

    midpoint = .00067
    c1 = 1 - math.log(midpoint, 10)
    c2 = math.log(midpoint, 10) + 1
    if manual_score == .00067:
        return .5
    elif manual_score > .00067:
        x = math.log(manual_score, 10) + c1
        return 1-(pow(.5, x))
    else:
        x = c2 - math.log(manual_score, 10)
        return pow(.5, x)

def combine_confidences(ml_score : float, manual_score : float) -> int:
    return (ml_score + manual_score) / 2

# uses combined confidences
# Takes a combined result dictionary and prints relevant analysis
# (dict: (word, ty) -> (method, ml_score, manual_score)) ->
def analyze_params_new(final_dict : dict[tuple[str, str], tuple[str, float, float]], 
                    manual_score_cutoff : float,
                    certainty_cutoff : float) -> float:
    
    ml_right = 0
    manual_right = 0
    combine_right = 0
    cutoff_right = 0

    for key in final_dict:
        method, ml_score, manual_score = final_dict[key]
        ty = key[1]

        if ml_score > .5:
            ml_right += 1


        if ty == "auto":
            if manual_score < manual_score_cutoff:
                manual_right += 1
        else:
            if manual_score > manual_score_cutoff:
                manual_right += 1

                
        if ml_score > 1-certainty_cutoff:
            cutoff_right += 1
        elif ml_score > certainty_cutoff:
            if ty == "auto":
                if manual_score < manual_score_cutoff:
                    cutoff_right += 1
            else:
                if manual_score > manual_score_cutoff:
                    cutoff_right += 1

        
        if ty == "auto":
            combined_score = combine_confidences(1-ml_score, normalize_manual_score(manual_score))
            if combined_score < .5:
                combine_right += 1
        else:
            combined_score = combine_confidences(ml_score, normalize_manual_score(manual_score))
            if combined_score > .5:
                combine_right += 1


    print("ml_right: " + str(ml_right))
    print("manual_right: " + str(manual_right))
    print("cutoff_right: " + str(cutoff_right))
    print("combine_right: " + str(combine_right))

    return 1.0



        


                    





def build_model():
    bins = 5
    weight = 4

    ms_dict_auto = build_ms_dict("auto")
    ms_dict_non = build_ms_dict("non")

    df = make_df(range(bins), weight, ms_dict_auto, ms_dict_non)
    model = RandomForestClassifier()

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model



def build_model_sim():
    bins = 3
    weight = 3

    englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")
    wcs = buildDict(100)
    words = grab_words(5000)
    ms_dict_auto = {}
    ms_dict_non = {}
    for word in words:
        ms_dict_auto[word] = simulate_ms(englishDictionary, wcs, word, True, 1)
        ms_dict_non[word] = simulate_ms(englishDictionary, wcs, word, False, 1)

    df = make_df(range(bins), weight, ms_dict_auto, ms_dict_non)
    model = RandomForestClassifier()

    y = df.ac
    X = df.drop(["ac"], axis=1, inplace=False)
    # Trains without ID
    model.fit(X.drop(["id"], axis=1, inplace=False), y)

    return model

def save_model(path : str, model):
    save_pickle_gz(model, path)
    return




# takes in a model and a move sequence, returns an int; 1 for auto, 0 for non
def classify_ms(model, ms : list[int]) -> int:
    bins = 5
    weight = 4
    certainty_cutoff = .26
    manual_cutoff = 0.00067


    data = np.empty((0, bins), dtype=float)
    list_dist = make_row_dist(ms, range(bins), weight)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)
    

    column_titles = make_column_titles_dist(range(bins))
    df = pd.DataFrame(data = data, columns = column_titles)

    print(df)
    
    # now predict from the dataframe, and then add manual

    pred_probas = model.predict_proba(df)[0]

    if pred_probas[0] > 1-certainty_cutoff:
        return 0
    elif pred_probas[1] > 1-certainty_cutoff:
        return 1
    # rn cc is .5, fixe the key[0] issue
    else: # go into manual scoring
        manual_score = get_score_from_ms_improved(ms, 1)[0][1]
        if manual_score > manual_cutoff:
            return 0
        else:
            return 1

# takes in a model and a move sequence, returns an int; 1 for auto, 0 for non
def classify_ms_sim(model, ms : list[int]) -> int:
    bins = 3
    weight = 3
    certainty_cutoff = .24
    manual_cutoff = 0.00067


    data = np.empty((0, bins), dtype=float)
    list_dist = make_row_dist(ms, range(bins), weight)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)
    

    column_titles = make_column_titles_dist(range(bins))
    df = pd.DataFrame(data = data, columns = column_titles)

    print(df)
    
    # now predict from the dataframe, and then add manual

    pred_probas = model.predict_proba(df)[0]

    if pred_probas[0] > 1-certainty_cutoff:
        return 0
    elif pred_probas[1] > 1-certainty_cutoff:
        return 1
    # rn cc is .5, fixe the key[0] issue
    else: # go into manual scoring
        manual_score = get_score_from_ms_improved(ms, 1)[0][1]
        if manual_score > manual_cutoff:
            return 0
        else:
            return 1



#bins: 5; weight: double weighted, normalized; certainty_cutoff: 0.26
#accuracy: 0.9809523809523809
# best fs 

if __name__ == '__main__':

    ms_dict_auto = build_ms_dict("auto")
    ms_dict_non = build_ms_dict("non")
    
    # Tests
    # 0 - weighting methods
    # 1 - unused
    # 2 - for examination of specific words
    # 3 - for testing combination parameters
    # 4 - for predicting single ms
    # 5 - for testing score combinations
    # 6 - (analyze_params_new) - tests combining confidences
    # 7 - for testing combination parameters, simulated word model
    # 8 - for testing simulated word model on gt words
    # 9 - save model
    test = 9


    
    if test == 5:
        parser = ArgumentParser()
        parser.add_argument('--score', type=str, required=True)
        args = parser.parse_args()
        print(normalize_manual_score(float(args.score)))


    if test == 4:
        model = build_model()
        print(str(classify_ms(model, [3, 6, 6])))

    
    # Test weighting methods
    if test == 0:
        max_bins_dist = 5
        df = eval_weighting(max_bins_dist, 20)

        plt.plot(df["unw"], label="unw")
        plt.plot(df["w"], label="w")
        plt.plot(df["unw/nn"], label="unw/nn")
        plt.plot(df["w/nn"], label="w/nn")
        plt.plot(df["dw"], label="dw")
        plt.plot(df["dw/nn"], label="dw/nn")
        leg = plt.legend(loc='lower center')
        plt.show()



    # For testing which words cause issues  
    if test == 2:
        df = make_df(range(6), 3, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        print("test 2, 6 bins, weighted/nn")

        results_dict = get_word_results(df, model, 100)

        acc_list = []
        for key in results_dict:
            (correct, total) = results_dict[key]
            #print(key)
            #print(str("accuracy: " + str(correct/total)))
            #print("correct: " + str(correct))
            #print("total: " + str(total))
    
            acc_list.append((key, (correct / total)))
        acc_list.sort(key=(lambda x: x[1]), reverse=False)
        for item in acc_list:
            print(item[0])
            print("accuracy: " + str(item[1]))

        for ((word, lst), acc) in acc_list:
            if acc < .9:
                print(lst + ", " + word + ": " + str(acc))
                print("non", end='')
                print(ms_dict_non[word])
                print("auto", end='')
                print(ms_dict_auto[word])

                print("\n")




    if test == 3:
        print("test 3")
        bins_list = [3, 4, 5]
        weights = [4, 3, 5]
        certainty_cutoffs = [.22, .24, .26, .28, .30, .32, .34, .36]
        max_length = -1
        iters = 60

        param_dict = eval_params(bins_list, weights, certainty_cutoffs, max_length, iters)
        results = []
        for key in param_dict:
            bins, weight, certainty_cutoff = key
            print("bins: " + str(bins), end="; ")
            print("weight: " + str(weight) + " (" + id_to_weight(weight) + ")", end="; ")
            print("certainty_cutoff: " + str(certainty_cutoff))
            acc = analyze_params(param_dict[key], 0.00067)
            results.append((key, acc))
            print("\n")
        results.sort(key=(lambda x: x[1]))
        for item in results:
            bins, weight, certainty_cutoff = item[0]
            print("bins: " + str(bins), end="; ")
            print("weight: " + id_to_weight(weight), end="; ")
            print("certainty_cutoff: " + str(certainty_cutoff))
            print("accuracy: " + str(item[1]))


    if test == 6:
        print("test 6")
        certainty_cutoff = .3
        max_length = 9
        iters = 20

        df = make_df(range(3), 4, ms_dict_auto, ms_dict_non)
        model = RandomForestClassifier()
        results_dict = get_word_results_proba(df, model, iters, ms_dict_auto, ms_dict_non) # dict (word, ty) -> (accuracy sum, total)
        final_dict = combine_algorithmic(results_dict, certainty_cutoff, ms_dict_auto, ms_dict_non)

        
        analyze_params_new(final_dict, .000067, certainty_cutoff)


    if test == 7:
        print("test 7")
        bins_list = [3, 4, 5]
        weights = [4, 3, 5]
        certainty_cutoffs = [.22, .24, .26, .28, .30, .32, .34, .36]
        max_length = -1
        iters = 60

        param_dict = eval_params_sim(bins_list, weights, certainty_cutoffs, max_length, iters)
        results = []
        for key in param_dict:
            bins, weight, certainty_cutoff = key
            print("bins: " + str(bins), end="; ")
            print("weight: " + str(weight) + " (" + id_to_weight(weight) + ")", end="; ")
            print("certainty_cutoff: " + str(certainty_cutoff))
            acc = analyze_params(param_dict[key], 0.00067)
            results.append((key, acc))
            print("\n")
        results.sort(key=(lambda x: x[1]))
        for item in results:
            bins, weight, certainty_cutoff = item[0]
            print("bins: " + str(bins), end="; ")
            print("weight: " + id_to_weight(weight), end="; ")
            print("certainty_cutoff: " + str(certainty_cutoff))
            print("accuracy: " + str(item[1]))
    

    if test == 8:
        model = build_model_sim()

        correct = (0, 0)
        total = (0, 0)
        fails = []

        for word in ms_dict_auto:
            print(word)
            print(ms_dict_auto[word])
            cls = classify_ms_sim(model, ms_dict_auto[word])
            if cls == 1:
                correct = (correct[0], correct[1] + 1)
            else:
                fails.append((word, "auto"))
            total = (total[0], total[1] + 1)
        for word in ms_dict_non:
            print(word)
            print(ms_dict_non[word])
            cls = classify_ms_sim(model, ms_dict_non[word])
            if cls == 0:
                correct = (correct[0] + 1, correct[1])
            else:
                fails.append((word, "non"))
            total = (total[0] + 1, total[1])
        
        print("non correct: " + str(correct[0]))
        print("auto correct: " + str(correct[1]))
        print("non total: " + str(total[0]))
        print("auto total: " + str(total[1]))
        print("non accuracy: " + str(correct[0]/total[0]))
        print("auto accuracy: " + str(correct[1]/total[1]))

        for fail, ty in fails:
            print(fail + ", " + ty + ": ", end="")
            if ty == "non":
                print(ms_dict_non[fail])
            else:
                print(ms_dict_auto[fail])

    elif test == 9:
        model = build_model()
        save_model("model.pkl.gz", model)
        print("model saved")