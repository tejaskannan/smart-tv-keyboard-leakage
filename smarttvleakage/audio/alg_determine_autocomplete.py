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






# Non-autocomplete Dict
ms_dict_non = {}
#ms_dict_non["a"] = [1]
# Dec of Ind
ms_dict_non["we"] = [1, 1]
ms_dict_non["hold"] = [6, 4, 1, 6]
ms_dict_non["these"] = [4, 2, 4, 2, 2]
ms_dict_non["truths"] = [4, 1, 3, 2, 2, 4]
ms_dict_non["to"] = [4, 4]
ms_dict_non["be"] = [6, 4]
ms_dict_non["self"] = [2, 2, 7, 5]
ms_dict_non["evident"] = [2, 3, 6, 6, 1, 5, 3]
ms_dict_non["that"] = [4, 2, 5, 5]
ms_dict_non["all"] = [1, 8, 0]
ms_dict_non["men"] = [8, 6, 5]
ms_dict_non["are"] = [1, 4, 1]
ms_dict_non["created"] = [4, 3, 1, 3, 5, 2, 1]
ms_dict_non["equal"] = [2, 2, 6, 7, 8]
ms_dict_non["they"] = [4, 2, 4, 3]
# ms_dict_non["endowed"] = [2, 5, 4, 7, 7, 1, 1]
ms_dict_non["by"] = [6, 3]
ms_dict_non["their"] = [4, 2, 4, 5, 4]
ms_dict_non["creator"] = [4, 3, 1, 3, 5, 4, 5]
ms_dict_non["with"] = [1, 6, 3, 2]
ms_dict_non["certain"] = [4, 2, 1, 1, 5, 8, 4]
ms_dict_non["unalienable"] = [6, 3, 6, 8, 2, 5, 5, 6, 5, 5, 7]
ms_dict_non["rights"] = [3, 4, 4, 1, 2, 4]
ms_dict_non["life"] = [9, 2, 5, 2]
ms_dict_non["liberty"] = [9, 2, 5, 4, 1, 1, 1]
ms_dict_non["and"] = [1, 6, 4]
# ms_dict_non["the"] = [4, 2, 4]
ms_dict_non["pursuit"] = [9, 3, 3, 3, 6, 1, 3]
ms_dict_non["of"] = [8, 6]
ms_dict_non["happiness"] = [6, 5, 10, 0, 2, 4, 5, 2, 0]
ms_dict_non["secure"] = [2, 2, 2, 6, 3, 1]
ms_dict_non["governments"] = [5, 5, 7, 3, 1, 4, 1, 6, 5, 3, 4]
ms_dict_non["instituted"] = [7, 4, 5, 4, 3, 3, 2, 2, 2, 1]
ms_dict_non["among"] = [1, 7, 4, 5, 2]
ms_dict_non["deriving"] = [3, 1, 1, 4, 6, 6, 4, 2]
ms_dict_non["just"] = [7, 1, 6, 4]
ms_dict_non["powers"] = [9, 1, 7, 1, 1, 3]
ms_dict_non["from"] = [4, 1, 5, 4]
ms_dict_non["consent"] = [4, 8, 5, 5, 2, 5, 3]
ms_dict_non["governed"] = [5, 5, 7, 3, 1, 4, 5, 1]
ms_dict_non["whenever"] = [1, 5, 4, 5, 5, 3, 3, 1]
ms_dict_non["any"] = [1, 6, 2]
ms_dict_non["form"] = [4, 6, 5, 5]
ms_dict_non["becomes"] = [6, 4, 2, 8, 4, 6, 2]
ms_dict_non["destructive"] = [3, 1, 2, 4, 1, 3, 6, 4, 3, 6, 3]
# ms_dict_non["ends"] = [2, 5, 4, 1]
ms_dict_non["it"] = [7, 3]
ms_dict_non["is"] = [7, 7]
ms_dict_non["right"] = [3, 4, 4, 1, 2]
ms_dict_non["people"] = [9, 7, 6, 1, 2, 7]
ms_dict_non["alter"] = [1, 8, 5, 2, 1]
ms_dict_non["or"] = [8, 5]

# Gettys
ms_dict_non["a"] = [1]
ms_dict_non["add"] = [1, 2, 0]
ms_dict_non["ago"] = [1, 4, 5]
ms_dict_non["be"] = [6, 4]
ms_dict_non["brave"] = [6, 3, 4, 4, 3]
ms_dict_non["brought"] = [6, 3, 5, 2, 3, 1, 2]
ms_dict_non["carrot"] = [4, 3, 4, 0, 5, 4]
ms_dict_non["cause"] = [4, 3, 7, 6, 2]
ms_dict_non["civil"] = [4, 7, 6, 6, 2]
ms_dict_non["continent"] = [4, 8, 5, 3, 3, 4, 5, 5, 3]
ms_dict_non["dedicate"] = [3, 1, 1, 6, 7, 3, 5, 2]
ms_dict_non["devotion"] = [3, 1, 3, 7, 4, 3, 1, 5]
ms_dict_non["did"] = [3, 6, 6]
ms_dict_non["do"] = [3, 7]
ms_dict_non["earth"] = [2, 3, 4, 1, 2]
ms_dict_non["endure"] = [2, 5, 4, 5, 3, 1]
ms_dict_non["equal"] = [2, 2, 6, 7, 8]
ms_dict_non["fathers"] = [4, 3, 4, 2, 3, 1, 3]
ms_dict_non["final"] = [4, 5, 4, 6, 8]
ms_dict_non["for"] = [4, 6, 5]
ms_dict_non["forget"] = [4, 6, 5, 2, 3, 2]
ms_dict_non["forth"] = [4, 6, 5, 1, 2]
ms_dict_non["freedom"] = [4, 1, 1, 0, 1, 7, 4]
#p2
ms_dict_non["god"] = [5, 5, 7]
ms_dict_non["ground"] = [5, 2, 5, 2, 3, 4]
ms_dict_non["hallow"] = [6, 5, 8, 0, 1, 7]
ms_dict_non["have"] = [6, 5, 4, 3]
ms_dict_non["here"] = [6, 3, 1, 1]
ms_dict_non["highly"] = [6, 3, 4, 1, 3, 4]
ms_dict_non["in"] = [7, 4]
ms_dict_non["last"] = [9, 8, 1, 4]
ms_dict_non["live"] = [9, 2, 6, 3]
ms_dict_non["lives"] = [9, 2, 6, 3, 2]
ms_dict_non["long"] = [9, 1, 5, 2]
ms_dict_non["measure"] = [8, 6, 3, 1, 6, 3, 1]
ms_dict_non["met"] = [8, 6, 2]
ms_dict_non["nation"] = [7, 5, 4, 3, 1, 5]
ms_dict_non["never"] = [7, 5, 3, 3, 1]
ms_dict_non["new"] = [7, 5, 1]
ms_dict_non["nobly"] = [7, 5, 6, 5, 4]
ms_dict_non["nor"] = [7, 5, 5]
ms_dict_non["note"] = [7, 5, 4, 2]
ms_dict_non["now"] = [7, 5, 8]
ms_dict_non["proper"] = [9, 6, 5, 1, 7, 1]
ms_dict_non["rather"] = [3, 4, 5, 2, 4, 1]
ms_dict_non["remember"] = [3, 1, 6, 6, 6, 2, 4, 1]
ms_dict_non["say"] = [2, 1, 6]
ms_dict_non["sense"] = [2, 2, 5, 5, 2]
ms_dict_non["should"] = [2, 4, 4, 2, 3, 6]
# ms_dict_non["it"] = []
#page 3
ms_dict_non["so"] = [2, 8]
ms_dict_non["take"] = [4, 5, 7, 6]
ms_dict_non["they"] = [4, 2, 4, 3]
ms_dict_non["thus"] = [4, 2, 2, 6]
ms_dict_non["us"] = [6, 6]
ms_dict_non["war"] = [1, 2, 4]
ms_dict_non["whether"] = [1, 5, 4, 2, 2, 4, 1]
ms_dict_non["which"] = [1, 5, 3, 7, 4]
ms_dict_non["work"] = [1, 7, 5, 5]
ms_dict_non["year"] = [5, 3, 3, 4]


# Autocomplete Dict
ms_dict_auto = {}

# Dec of Ind
ms_dict_auto["we"] = [1, 1]
ms_dict_auto["hold"] = [6, 1, 3, 1]
ms_dict_auto["these"] = [4, 1, 0, 6, 1]
ms_dict_auto["truths"] = [4, 2, 1, 2, 0, 2]
ms_dict_auto["to"] = [4, 1]
ms_dict_auto["be"] = [6, 1]
ms_dict_auto["self"] = [2, 1, 8, 1]
ms_dict_auto["evident"] = [2, 1, 2, 0, 0, 0, 2]
ms_dict_auto["that"] = [4, 1, 1, 0]
ms_dict_auto["all"] = [1, 1, 10]
ms_dict_auto["men"] = [8, 1, 2]
ms_dict_auto["are"] = [1, 1, 0]
ms_dict_auto["created"] = [4, 1, 3, 1, 1, 0, 0]
ms_dict_auto["equal"] = [2, 3, 1, 2, 0]
ms_dict_auto["they"] = [4, 1, 0, 0]
# ms_dict_auto["endowed"] = []
ms_dict_auto["by"] = [6, 1]
ms_dict_auto["their"] = [4, 1, 0, 1, 0]
ms_dict_auto["creator"] = [4, 1, 3, 1, 1, 1, 0]
ms_dict_auto["with"] = [1, 1, 0, 0]
ms_dict_auto["certain"] = [4,3, 1, 0, 0, 0, 0]
ms_dict_auto["unalienable"] = [6, 1, 9, 9, 1, 0, 0, 0, 0, 0, 0]
ms_dict_auto["rights"] = [3, 1, 0, 0, 0, 0]
ms_dict_auto["life"] = [9, 1, 1, 0]
ms_dict_auto["liberty"] = [9, 1, 5, 1, 0, 0, 0]
ms_dict_auto["and"] = [1, 1, 0]
# ms_dict_auto["the"] = []
ms_dict_auto["pursuit"] = [9, 1, 8, 1, 1, 0, 1]
ms_dict_auto["of"] = [8, 1]
ms_dict_auto["happiness"] = [6, 1, 2, 6, 1, 0, 0, 0, 10]
ms_dict_auto["secure"] = [2, 1, 0, 1, 0, 1]
ms_dict_auto["governments"] = [5, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0]
ms_dict_auto["instituted"] = [7, 1, 1, 1, 1, 1, 0, 0, 1, 0]
ms_dict_auto["among"] = [1, 8, 1, 0, 0]
ms_dict_auto["deriving"] = [3, 1, 2, 1, 1, 0, 0, 0]
ms_dict_auto["just"] = [7, 1, 0, 0]
ms_dict_auto["powers"] = [9, 1, 2, 0, 0, 1]
ms_dict_auto["from"] = [4, 1, 0, 0]
ms_dict_auto["consent"] = [4, 1, 1, 1, 1, 1, 0]
ms_dict_auto["governed"] = [5, 1, 4, 1, 0, 0, 2, 0]
ms_dict_auto["whenever"] = [1, 1, 2, 0, 0, 0, 0, 0]
ms_dict_auto["any"] = [1, 1, 1]
ms_dict_auto["form"] = [4, 1, 0, 5]
ms_dict_auto["becomes"] = [6, 1, 0, 1, 0, 0, 0]
ms_dict_auto["destructive"] = [3, 1, 3, 1, 0, 1, 0, 0, 0, 1, 0]
# ms_dict_auto["ends"] = []
ms_dict_auto["it"] = [7, 1]
ms_dict_auto["is"] = [7, 1]
ms_dict_auto["right"] = [3, 1, 0, 0, 0]
ms_dict_auto["people"] = [9, 1, 0, 0, 0, 0]
ms_dict_auto["alter"] = [1, 1, 0, 1, 0]
ms_dict_auto["or"] = [8, 1]


# Gettys, p1
ms_dict_auto["a"] = [1]
ms_dict_auto["add"] = [1, 3, 0]
ms_dict_auto["ago"] = [1, 5, 1]
ms_dict_auto["be"] = [6, 1]
ms_dict_auto["brave"] = [6, 4, 1, 1, 0]
ms_dict_auto["brought"] = [6, 6, 4, 1, 1, 0, 0, 0]
ms_dict_auto["carrot"] = [4, 1, 0, 5, 1, 0]
ms_dict_auto["cause"] = [4, 1, 8, 1, 0]
ms_dict_auto["civil"] = [4, 8, 7, 1, 0]
ms_dict_auto["continent"] = [4, 1, 1, 0, 1, 0, 1, 0, 0]
ms_dict_auto["dedicate"] = [3, 1, 1, 1, 0, 0, 0, 0]
ms_dict_auto["devotion"] = [3, 1, 2, 1, 0, 2, 0, 0]
ms_dict_auto["did"] = [3, 1, 0]
ms_dict_auto["do"] = [3, 1]
ms_dict_auto["earth"] = [2, 4, 1, 0, 0]
ms_dict_auto["endure"] = [2, 1, 2, 4, 1, 0]
ms_dict_auto["equal"] = [2, 3, 1, 2, 0]
ms_dict_auto["fathers"] = [4, 4, 1, 0, 0, 0, 2]
ms_dict_auto["final"] = [4, 1, 0, 1, 0]
ms_dict_auto["for"] = [4, 1, 0]
ms_dict_auto["forget"] = [4, 1, 0, 0, 0, 0]
ms_dict_auto["forth"] = [4, 1, 0, 4, 1]
ms_dict_auto["freedom"] = [4, 1, 1, 2, 1, 0, 0]
#p2
ms_dict_auto["god"] = [5, 1, 2]
ms_dict_auto["ground"] = [5, 1, 1, 0, 0, 0]
ms_dict_auto["hallow"] = [6, 1, 5, 0, 2, 1]
ms_dict_auto["have"] = [6, 1, 0, 0]
ms_dict_auto["here"] = [6, 1, 0, 0]
ms_dict_auto["highly"] = [6, 1, 0, 0, 0, 0]
ms_dict_auto["in"] = [7, 1]
ms_dict_auto["last"] = [9, 1, 0, 0]
ms_dict_auto["live"] = [9, 1, 6, 1]
ms_dict_auto["lives"] = [9, 1, 6, 1, 0]
ms_dict_auto["long"] = [9, 1, 1, 0]
ms_dict_auto["measure"] = [8, 1, 0, 6, 1, 0, 1]
ms_dict_auto["met"] = [8, 1, 1]
ms_dict_auto["nation"] = [7, 1, 2, 2, 0, 0]
ms_dict_auto["never"] = [7, 1, 0, 0, 0]
ms_dict_auto["new"] = [7, 1, 1]
ms_dict_auto["nobly"] = [7, 1, 1, 1, 1]
ms_dict_auto["nor"] = [7, 1, 4]
ms_dict_auto["note"] = [7, 1, 1, 1]
ms_dict_auto["now"] = [7, 1, 0]
ms_dict_auto["proper"] = [9, 1, 1, 1, 1, 0]
ms_dict_auto["rather"] = [3, 5, 1, 0, 0, 0]
ms_dict_auto["remember"] = [3, 1, 1, 0, 0, 0, 0, 0]
ms_dict_auto["say"] = [2, 1, 0]
ms_dict_auto["sense"] = [2, 1, 2, 2, 0]
ms_dict_auto["should"] = [2, 1, 1, 0, 0, 0]
# ms_dict_auto["it"] = []
#3
ms_dict_auto["so"] = [2, 1]
ms_dict_auto["take"] = [4, 1, 0, 0]
ms_dict_auto["they"] = [4, 1, 6, 1]
ms_dict_auto["thus"] = [4, 1, 0, 0]
ms_dict_auto["us"] = [6, 1]
ms_dict_auto["war"] = [1, 1, 2]
ms_dict_auto["whether"] = [1, 1, 2, 1, 0, 0, 0]
ms_dict_auto["which"] = [1, 1, 8, 1, 0]
ms_dict_auto["work"] = [1, 8, 1, 0]
ms_dict_auto["year"] = [5, 1, 0, 1]




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
    # Path issue?
    englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")


    word_list = []
    for idx, (guess, score, candidates_count) in enumerate(get_words_from_moves(move_sequence=ms, graph=graph, dictionary=dictionary, max_num_results=SAMPLES)):    
        raw_score = englishDictionary.get_score_for_string(guess, False) 
        score = adjust_for_len(raw_score, guess, strategy)
        word_list.append((guess, score))
    
    
    word_list.sort(key=(lambda x: x[1]), reverse=True)
    #word_list = list(filter(lambda x: x[1] > 0, word_list))

    return word_list


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

