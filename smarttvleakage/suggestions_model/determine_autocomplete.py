from typing import List, Dict
from argparse import ArgumentParser

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

from smarttvleakage.dictionary import EnglishDictionary
from smarttvleakage.suggestions_model.alg_determine_autocomplete import get_score_from_ms
from smarttvleakage.suggestions_model.manual_score_dict import (build_msfd,
                                                                build_ms_dict, build_rockyou_ms_dict)
from smarttvleakage.suggestions_model.simulate_ms import grab_words, simulate_ms
from smarttvleakage.utils.file_utils import save_pickle_gz




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

    if weighted == 0:
        for bin in bins:
            hist[bin] = hist[bin] / len(moves)
    elif weighted == 1:
        for bin in bins:
            hist[bin] = hist[bin] / max(1, sum(bins))
    elif weighted == 4:
        for bin in bins:
            hist[bin] = hist[bin] / max(1, sum(bins))
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
                    englishDictionary, words_auto, words_non,
                    include_rockyou : bool = False, bins : int = 4, weight : int = 3):
    """Builds a model on simulated data"""

    ms_dict_auto = {}
    ms_dict_non = {}
    for word in words_auto:
        ms_dict_auto[word] = simulate_ms(englishDictionary, word, True)
    for word in words_non:
        ms_dict_non[word] = simulate_ms(englishDictionary, word, False)

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


# takes in a model and a move sequence, returns an int; 1 for auto, 0 for non
def classify_ms(model, msfd,
                ms : List[int],
                bins : int = 4, weight : int = 3, certainty_cutoff : float = .3) -> int:
    """Classifies a move sequence"""

    manual_cutoff = 0.00067

    data = np.empty((0, bins), dtype=float)
    list_dist = make_row_dist(ms, range(bins), weight)
    new_row = np.array(list_dist, dtype=float)
    data = np.append(data, [new_row], axis=0)

    column_titles = make_column_titles_dist(range(bins))
    df = pd.DataFrame(data = data, columns = column_titles)

    # now predict from the dataframe, and then add manual

    pred_probas = model.predict_proba(df)[0]

    if pred_probas[0] > 1-certainty_cutoff:
        return 0
    if pred_probas[1] > 1-certainty_cutoff:
        return 1

    # go into manual scoring, use strategy = 2!
    manual_score = get_score_from_ms(ms, 2, msfd)[0][1]
    if manual_score > manual_cutoff:
        return 0
    return 1




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False)
    parser.add_argument("--ms-path-non", type=str, required=False)
    parser.add_argument("--ms-path-rockyou", type=str, required=False)
    parser.add_argument("--ed-path", type=str, required=False)
    parser.add_argument("--words-path", type=str, required=False)
    parser.add_argument("--msfd-path", type=str, required=False)
    parser.add_argument("--results-path", type=str, required=False)
    args = parser.parse_args()

    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
        ms_dict_auto = build_ms_dict(args.ms_path_auto)

    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_auto)

    if args.ms_path_rockyou is None:
        args.ms_path_rockyou = "suggestions_model/local/rockyou.txt"
        ms_dict_rockyou = build_rockyou_ms_dict(args.ms_path_rockyou, 500)

    if args.msfd_path is None:
        args.msfd_path = "suggestions_model/local/msfd.pkl.gz"
        msfd = build_msfd(args.msfd_path)

    if args.ed_path is None:
        englishDictionary = EnglishDictionary.restore(
            "suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "build":
        englishDictionary = EnglishDictionary(50)
        englishDictionary.build(
            "suggestions_model/local/dictionaries/enwiki-20210820-words-frequency.txt", 50, True)
        englishDictionary.save("suggestions_model/local/dictionaries/ed.pkl.gz")
    else:
        englishDictionary = EnglishDictionary.restore(args.ed_path)

    if args.words_path is None:
        args.words_path = "suggestions_model/local/dictionaries/big.txt"

    # Tests
    # 0 - save model
    # 1 - test all dicts
    # 2 - test all dicts configs
    # 3 - (1) with sim model
    # 4 - (2) with sim model
    test = 2


    if test == 4:

        auto_words = grab_words(2000, args.words_path)
        non_words = grab_words(2000, args.words_path)
        ms_dict_rockyou_train = build_rockyou_ms_dict(args.ms_path_rockyou, 500)
        ms_dict_auto_test = build_ms_dict(args.ms_path_auto)
        ms_dict_non_test = build_ms_dict(args.ms_path_non)
        ms_dict_rockyou_test = build_rockyou_ms_dict(args.ms_path_rockyou, 100, 500)

        bss = [3, 4, 5]
        weights = [3, 4, 5]
        certainty_cutoffs = [.25, .3, .35, .4, .45, .5]
        print("test 2")

        print("building rockyou dict")
        print("build rockyou dict")

        results = {}
        for bs in bss:
            for w in weights:
                model = build_model_sim(ms_dict_rockyou=ms_dict_rockyou_train,
                                        englishDictionary=englishDictionary,
                                        words_auto=auto_words, words_non=non_words,
                                        include_rockyou=True, bins=bs, weight=w)
                    
                for cc in certainty_cutoffs:
                    print("testing bins: " + str(bs) + "; cc: " + str(cc) + "; " + id_to_weight(w))

                    gt_array = []
                    label_array = []

                    for _, val in ms_dict_auto_test.items():
                        pred = classify_ms(model, msfd, val, bins=bs, weight=w, certainty_cutoff=.3)
                        gt_array.append(1)
                        label_array.append(pred)

                    for _, val in ms_dict_non_test.items():
                        pred = classify_ms(model, msfd, val, bins=bs, weight=w, certainty_cutoff=.3)
                        gt_array.append(0)
                        label_array.append(pred)

                    for _, val in ms_dict_rockyou_test.items():
                        pred = classify_ms(model, msfd, val, bins=bs, weight=w, certainty_cutoff=.3)
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
        print("test 14")
        #model = read_pickle_gz("max/model_rockyou.pkl.gz")
        model = build_model_sim(build_rockyou_ms_dict(args.ms_path_rockyou, 200), englishDictionary, grab_words(2000, args.words_path), grab_words(2000, args.words_path), include_rockyou=True)

        ms_dict_rockyou = build_rockyou_ms_dict(args.ms_path_rockyou, 105, 200)

        correct_auto = 0
        correct_non = 0
        correct_rockyou = 0
        total_auto = 0
        total_non = 0
        total_rockyou = 0

        for key, val in ms_dict_auto.items():
            print(key)
            if classify_ms(model, msfd, val, certainty_cutoff=.5) == 1:
                correct_auto += 1
            total_auto += 1
        for key, val in ms_dict_non.items():
            print(key)
            if classify_ms(model, msfd, val, certainty_cutoff=.5) == 0:
                correct_non += 1
            total_non += 1
        for key, val in ms_dict_rockyou.items():
            print(key)
            if classify_ms(model, msfd, val, certainty_cutoff=.5) == 0:
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

        print("building rockyou dict")
        print("build rockyou dict")

        results = {}
        for bs in bss:
            for w in weights:
                model = build_model(include_rockyou=True, bins=bs, weight=w, ms_dict_auto=ms_dict_auto_train, ms_dict_non=ms_dict_non_train, ms_dict_rockyou=ms_dict_rockyou_train)
                    
                for cc in certainty_cutoffs:
                    print("testing bins: " + str(bs) + "; cc: " + str(cc) + "; " + id_to_weight(w))

                    gt_array = []
                    label_array = []

                    for _, val in ms_dict_auto_test.items():
                        pred = classify_ms(model, msfd, val, bins=bs, weight=w, certainty_cutoff=.3)
                        gt_array.append(1)
                        label_array.append(pred)

                    for _, val in ms_dict_non_test.items():
                        pred = classify_ms(model, msfd, val, bins=bs, weight=w, certainty_cutoff=.3)
                        gt_array.append(0)
                        label_array.append(pred)

                    for _, val in ms_dict_rockyou_test.items():
                        pred = classify_ms(model, msfd, val, bins=bs, weight=w, certainty_cutoff=.3)
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
        model = build_model(include_rockyou=True, ms_dict_auto=build_ms_dict(args.ms_path_auto, take=53), ms_dict_non=build_ms_dict(args.ms_path_non, take=70), ms_dict_rockyou=ms_dict_rockyou)
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
            pred = classify_ms(model, msfd, val, certainty_cutoff=.3)
            if pred == 1:
                correct_auto += 1
            total_auto += 1

            gt_array.append(1)
            label_array.append(pred)
        for key, val in ms_dict_non.items():
            print(key)
            pred = classify_ms(model, msfd, val, certainty_cutoff=.3)
            if pred == 0:
                correct_non += 1
            total_non += 1

            gt_array.append(0)
            label_array.append(pred)
        for key, val in ms_dict_rockyou.items():
            print(key)
            pred = classify_ms(model, msfd, val, certainty_cutoff=.3)
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
        model = build_model_sim(build_rockyou_ms_dict(args.ms_path_rockyou, 200), englishDictionary, grab_words(1000, args.words_path), grab_words(1000, args.words_path), include_rockyou=True)
        save_model("max/model_rockyou_sim.pkl.gz", model)
        print("model saved")
