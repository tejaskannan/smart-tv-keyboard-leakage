from typing import List, Dict, Tuple
from argparse import ArgumentParser
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

from smarttvleakage.dictionary import EnglishDictionary, restore_dictionary
from smarttvleakage.suggestions_model.alg_determine_autocomplete import get_score_from_ms
from smarttvleakage.suggestions_model.manual_score_dict import (build_msfd,
                                                            build_ms_dict, build_rockyou_ms_dict)
from smarttvleakage.suggestions_model.simulate_ms import grab_words, simulate_ms, add_mistakes, add_mistakes_to_ms_dict
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


# dict looks like
# tree
#  param
#  cutoff
#  l node
#  r node

def build_node(clf, node):
    dict = {}
    dict["feature"] = int(clf.tree_.feature[node])
    dict["threshold"] = int(clf.tree_.threshold[node])

    dict["class"] = -1

    if clf.tree_.children_left[node] == -1:
        dict["left"] = -1
    else:
        dict["left"] = build_node(clf, clf.tree_.children_left[node])

    if clf.tree_.children_right[node] == -1:
        dict["right"] = -1
    else:
        dict["right"] = build_node(clf, clf.tree_.children_right[node])

    return dict
        




def extract_choices():

    model = read_pickle_gz(args.model_path)
    print("estimators: " + str(len(model.estimators_)))

    
    for i, clf in enumerate(model.estimators_):

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







if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=str, required=False) #where to save model
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    args = parser.parse_args()

    mode = 0

    model = read_pickle_gz(args.model_path)
    print("estimators: " + str(len(model.estimators_)))

    
    if mode == 0:

        forest = {}

        for i, clf in enumerate(model.estimators_):
            forest[i] = build_node(clf, 0)

        if args.save_path is None:

            print(forest)




            print("no save path")
        else:
            with open(args.save_path, "w") as f:
                json.dump(forest, f)
                f.close()
    
    
    
    if mode == 1:

        forest = {}
        for i, clf in enumerate(model.estimators_):
            forest[i] = clf.get_params()
        
        if args.save_path is None:
            print("no save path")
        else:
            with open(args.save_path, "w") as f:
                json.dump(forest, f)
                f.close()

