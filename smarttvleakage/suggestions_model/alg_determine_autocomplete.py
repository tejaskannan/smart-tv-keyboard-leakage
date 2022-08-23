from argparse import ArgumentParser
from typing import List, Dict, Tuple

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import UniformDictionary, EnglishDictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.suggestions_model.manual_score_dict import get_word_from_ms, build_ms_dict, build_msfd
from smarttvleakage.utils.constants import KeyboardType, SmartTVType



def adjust_for_len(raw : float, word : str, strategy : int) -> float:
    """returns a score adjusted for length, provided a strategy number"""
    if strategy == 0: # linear
        return raw * len(word)
    if strategy == 1: # quadratic
        return raw * pow(len(word), 2)
    if strategy == 2: # exponential
        return raw * pow(2, len(word))

    return raw

def get_score_from_ms(ms : List[int], strategy : int,
                    msfd : Dict[List[int], Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Returns the highest scoring word (and it's score) for a given ms"""
    word, raw_score = get_word_from_ms(ms, msfd)
    score = adjust_for_len(raw_score, word, strategy)
    return [(word, score)]

# used for finding best parameters
# return the best partitioning for a list of scores
def get_best_cutoff(scores : List[Tuple[float, str]]) -> Tuple[int, float, int]:
    """Used for finding best parameters, returns the best cutoff score"""
    scores.sort(key=lambda x: x[0])
    best = (0, 0)
    for i in range(len(scores)):
        sucs = 0
        for (_, ty) in scores[:i]:
            if ty == "auto":
                sucs = sucs+1
        for (_, ty) in scores[i:]:
            if ty == "non":
                sucs = sucs+1
        if sucs > best[1]:
            best = (i, sucs)

    cutoff_point = (scores[best[0]][0]+scores[best[0]+1][0]) / 2
    return (best[0], cutoff_point, best[1])

def test_strategy(strategy : int,
                ms_dict_auto : Dict[str, List[int]],
                ms_dict_non : Dict[str, List[int]],
                msfd : Dict[List[int], Tuple[str, float]]) -> Tuple[int, float, int, str]:
    """Tests a word scoring strategy"""
    scores = []
    print("testing strategy: " + str(strategy))
    print("testing auto")
    for key in ms_dict_auto:
        word, score = get_score_from_ms(ms_dict_auto[key], strategy, msfd)[0]
        print(word)
        print(score)
        scores.append((score, "auto"))
    print("testing non")
    for key in ms_dict_non:
        word, score = get_score_from_ms(ms_dict_non[key], strategy, msfd)[0]
        print(word)
        print(score)
        scores.append((score, "non"))

    cut, cut_point, result = get_best_cutoff(scores)
    print("cut: " + str(cut))
    print("cut point: " + str(cut_point))
    print("result: " + str(result))
    print("total: " + str(len(scores)))
    return (cut, cut_point, result, str(len(scores)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False)
    parser.add_argument("--ms-path-non", type=str, required=False)
    parser.add_argument("--msfd-path", type=str, required=False)
    parser.add_argument("--moves-list", type=int, required=False, nargs="+",
    help="A space-separated sequence of the number of moves.")
    args = parser.parse_args()

    ms_dict_auto = build_ms_dict(args.ms_path_auto)
    ms_dict_non = build_ms_dict(args.ms_path_non)
    msfd = build_msfd(args.msfd_path)

    # tests
    test = 3

    if test == 3:
        strategies = [0, 1, 2]
        strategy_scores = {}
        for strategy in strategies:
            strategy_scores[strategy] = test_strategy(strategy, ms_dict_auto, ms_dict_non, msfd)

        for key, val in strategy_scores.items():
            cut, cut_point, result, total = val
            print("strategy: " + str(key))
            print("cut: " + str(cut))
            print("cut point: " + str(cut_point))
            print("result: " + str(result))
            print("total: " + str(total))

    if test == 2:
        # init score dict
        score_dict = {}
        for key in ms_dict_auto:
            score_dict[key] = []
        for key in ms_dict_non:
            if key not in ms_dict_auto:
                score_dict[key] = []

        print("testing auto")
        for key, val in ms_dict_auto.items():
            word, score = get_score_from_ms(val, 2, msfd)[0]
            print(word)
            print(score)
            score_dict[key].append(("auto", score))
        print("testing non")
        non_scores = []
        for key, val in ms_dict_non.items():
            word, score = get_score_from_ms(val, 2, msfd)[0]
            print(word)
            print(score)
            score_dict[key].append(("non", score))

        non_wins = 0
        auto_wins = 0
        non_total = 0
        auto_total = 0
        for key, val in score_dict.items():
            score_list = val
            print(key)
            if len(score_list) == 0:
                continue
            if len(score_list) == 1:
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

    if test == 0:
        graph = MultiKeyboardGraph(KeyboardType.SAMSUNG)
        dictionary = UniformDictionary()
        englishDictionary = EnglishDictionary.restore(path="../local/dictionaries/ed.pkl.gz")
        word_list = []
        for idx, (guess, score, candidates_count) in enumerate(
            get_words_from_moves(
                move_sequence=args.moves_list, graph=graph,
                dictionary=dictionary, tv_type=SmartTVType.SAMSUNG, max_num_results=None)):
            word_list.append((guess, englishDictionary.get_score_for_string(guess, False)))
        word_list.sort(key=(lambda x: x[1]), reverse=True)
        word_list = filter(lambda x: x[1] > 0, word_list)
        for (word, score) in word_list:
            print(word)
            print("score: ", end="")
            print(score)
        # aggregate off, used for prefixes
