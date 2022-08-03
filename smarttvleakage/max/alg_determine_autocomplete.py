import time
import io
import os.path
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import defaultdict, namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.max.manual_score_dict import get_word_from_ms, build_ms_dict




# returns a score adjusted for length, provided a strategy number 
def adjust_for_len(raw : float, word : str, strategy : int) -> float:
    if strategy == 0: # linear
        return raw * len(word)
    elif strategy == 1: # quadratic
        return raw * pow(len(word), 2)
    elif strategy == 2: # exponential
        return raw * pow(2, len(word))





# gets a dictionary score from a move sequence by taking 500 samples, returns all samples with their scores
# This function is the important one used after the ML model, and could be improved
def get_score_from_ms(ms : list[int], strategy : int) -> list[tuple[str, float]]:
    print(ms)
    # when to cut off
    #SAMPLES = max(500, 20 * pow(len(ms), 2))
    SAMPLES = 500

    graph = MultiKeyboardGraph()
    dictionary = UniformDictionary()
    englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")
    # englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")


    word_list = []
    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves(move_sequence=ms, graph=graph, dictionary=dictionary, max_num_results=SAMPLES)):    
        raw_score = englishDictionary.get_score_for_string(guess, False) 
        score = adjust_for_len(raw_score, guess, strategy)
        word_list.append((guess, score))
    
    
    word_list.sort(key=(lambda x: x[1]), reverse=True)
    #word_list = list(filter(lambda x: x[1] > 0, word_list))

    return word_list

# This improved version uses a pre-built dictionary from ms -> highest scoring word
def get_score_from_ms_improved(ms : list[int], strategy : int) -> list[tuple[str, float]]:
    print(ms)
  
    word, raw_score = get_word_from_ms(ms)
    score = adjust_for_len(raw_score, word, strategy)
    return [(word, score)]


# used for finding best parameters
# return the best partitioning for a list of scores
def get_best_cutoff(scores : list[tuple[float, str]]) -> tuple[int, float, int]:

    scores.sort(key=lambda x: x[0])

    best = (0, 0)
    for i in range(len(scores)):
        sucs = 0
        for (score, ty) in scores[:i]:
            if ty == "auto":
                sucs = sucs+1
        for (score, ty) in scores[i:]:
            if ty == "non":
                sucs = sucs+1
        if sucs > best[1]:
            best = (i, sucs)

    cutoff_point = (scores[best[0]][0]+scores[best[0]+1][0]) / 2
    return (best[0], cutoff_point, best[1])





def test_strategy(strategy : int):
    max_length = 8
    min_length = 1
    scores = []
    print("testing strategy: " + str(strategy))

    print("testing auto")
    ms_dict_auto = build_ms_dict("auto")
    ms_dict_non = build_ms_dict("non")
    for key in ms_dict_auto:
        
        # For Speed
        if len(key) > max_length or len(key) < min_length:
            continue

        word, score = get_score_from_ms(ms_dict_auto[key], strategy)[0]
        print(word)
        print(score)
        
        scores.append((score, "auto"))

    print("testing non")
    for key in ms_dict_non:
        
        # For Speed:
        if len(key) > max_length or len(key) < min_length:
            continue

        word, score = get_score_from_ms(ms_dict_non[key], strategy)[0]
        print(word)
        print(score)
        
        scores.append((score, "non"))

    cut, cut_point, result = get_best_cutoff(scores)
    print("cut: " + str(cut))
    print("cut point: " + str(cut_point))
    print("result: " + str(result))
    print("total: " + str(len(scores)))
    return (cut, cut_point, result, str(len(scores)))




if __name__ == '__main__':
    ms_dict_auto = build_ms_dict("auto")
    ms_dict_non = build_ms_dict("non")


    test = 3


    if test == 3:

        strategies = [0, 1, 2]
        strategy_scores = {}
        for strategy in strategies:
            strategy_scores[strategy] = test_strategy(strategy)


        for key in strategy_scores:
            cut, cut_point, result, total = strategy_scores[key]
            print("strategy: " + str(key))
            print("cut: " + str(cut))
            print("cut point: " + str(cut_point))
            print("result: " + str(result))
            print("total: " + str(total))








    # init score dict
    score_dict = {}
    for key in ms_dict_auto:
        score_dict[key] = []
    for key in ms_dict_non:
        if key not in ms_dict_auto:
            score_dict[key] = []

    if test == 2:
        
        print("testing auto")
        for key in ms_dict_auto:
            
            # For Speed:
            if len(key) > 7:
                continue

            word, score = get_score_from_ms(ms_dict_auto[key], 2)[0]
            print(word)
            print(score)
            
            score_dict[key].append(("auto", score))
       

        print("testing non")
        non_scores = []
        for key in ms_dict_non:

            # For Speed:
            if len(key) > 7:
                continue

            word, score = get_score_from_ms(ms_dict_non[key], 2)[0]
            print(word)
            print(score)
            
            score_dict[key].append(("non", score))
        
        
        non_wins = 0
        auto_wins = 0
        non_total = 0
        auto_total = 0
        for key in score_dict:
            score_list = score_dict[key]
            print(key)
            if len(score_list) == 0:
                continue
            elif len(score_list) == 1:
                print(score_list[0][0] + ": ", end="")
                print(score_list[0][1])
            elif len(score_list) == 2:
                print(score_list[0][0] + ": ", end="")
                print(score_list[0][1])
                print(score_list[1][0] + ": ", end="")
                print(score_list[1][1])

                if score_list[0][1] > score_list[1][1]:
                    auto_wins = auto_wins+1
                elif score_list[0][1] < score_list[1][1]:
                    non_wins = non_wins+1
                auto_total = auto_total + score_list[0][1]
                non_total = non_total + score_list[1][1]
            
            
            print("auto_wins: " + str(auto_wins))
            print("non_wins: " + str(non_wins))

            print("auto total: " + str(auto_total))
            print("non total: " + str(non_total))

            #print("\n")



    if test == 0:
        parser = ArgumentParser()
        parser.add_argument('--moves-list', type=int, required=True, nargs='+', help='A space-separated sequence of the number of moves.')
        args = parser.parse_args()

        graph = MultiKeyboardGraph()

        dictionary = UniformDictionary()
        englishDictionary = EnglishDictionary.restore(path="../local/dictionaries/ed.pkl.gz")

        word_list = []
        for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves(move_sequence=args.moves_list, graph=graph, dictionary=dictionary, max_num_results=None)):    
            word_list.append((guess, englishDictionary.get_score_for_string(guess, False)))
        word_list.sort(key=(lambda x: x[1]), reverse=True)
        word_list = filter(lambda x: x[1] > 0, word_list)
        for (word, score) in word_list:
            print(word)
            print("score: ", end="")
            print(score)
        # aggregate off, used for prefixes

