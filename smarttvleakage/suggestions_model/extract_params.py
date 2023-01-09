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









if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save-path", type=str, required=False) #where to save model
    parser.add_argument("--model-path", type=str, required=False) #where to load model
    args = parser.parse_args()

    model = read_pickle_gz(args.model_path)
    print("estimators: " + str(len(model.estimators_)))

    forest = {}
    for i, clf in enumerate(model.estimators_):
        forest[i] = clf.get_params()
    
    if args.save_path is None:
        print("no save path")
    else:
        with open(args.save_path, "w") as f:
            json.dump(forest, f)
            f.close()

