import time

from smarttvleakage.suggestions_model.generate_cc import (generate_date,
                                                        generate_sec, add_acc, finish_cc)
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType

from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.audio.constants import SAMSUNG_KEY_SELECT

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import NumericDictionary, restore_dictionary
from smarttvleakage.suggestions_model.bin_mod import build_valid_cc_list

# Build Testing Dicts
def build_cc_ms_dict(kb, count : int, ty : str = "k") -> dict[str, list[int]]:
    """Build a dictionary of valid cc numbers"""
    ms_dict = {}
    valid_bins = build_valid_cc_list("cc_bin_dict000")
    for i in range(min(count, len(valid_bins))):
        b = int(valid_bins[i])
        cc = str(finish_cc(add_acc(b)))
        print(b)
        print(cc)

        path = []
        for m in findPath(cc, False, False, 0, 0, 0, kb):
            path.append(m.num_moves)
        ms_dict[cc] = path

    return ms_dict

def build_date_ms_dict(kb, count : int) -> dict[str, list[int]]:
    """Build a dictionary of valid 4-digit date strings"""
    ms_dict = {}
    for i in range(count):
        print(i)
        date = generate_date()
        print(date)
        path = []
        for m in findPath(date, 0, False, False, 0, 0, kb):
            path.append(m.num_moves)
        ms_dict[date] = path

    return ms_dict

def build_sec_ms_dict(kb, count : int) -> dict[str, list[int]]:
    """Build a dictionary of valid 3/4 digit cvv codes"""
    ms_dict = {}
    for i in range(count):
        print(i)
        sec = generate_sec()
        path = []
        for m in findPath(sec, 0, False, False, 0, 0, kb):
            path.append(m.num_moves)
        ms_dict[sec] = path

    return ms_dict


def test_dict(dictionary, ms_dict, max_num_results : int = 300) -> tuple[int, int, int, int]:
    """Test a dictionary on an ms_dict, returns success, t10, total, and total rank sum"""
    graph = MultiKeyboardGraph(KeyboardType.SAMSUNG)

    results = {}
    for key, ms in ms_dict.items():
        moves = [Move(num_moves=num_moves,
                    end_sound=SAMSUNG_KEY_SELECT) for num_moves in ms]

        guesses = 0
        found = False
        final_candidates_count = 0
        for _, (guess, _, candidates_count) in enumerate(
            get_words_from_moves(moves, graph=graph, dictionary=dictionary,
                                tv_type=SmartTVType.SAMSUNG, max_num_results=max_num_results,
                                precomputed=None)):
            if not guess.isnumeric():
                print("non numeric")
                continue

            guesses += 1
            if key == guess:
                results[key] = (guesses, candidates_count)
                found = True
                break
            final_candidates_count = candidates_count
        if not found:
            results[key] = (max_num_results+1, final_candidates_count)

    top_10 = list(filter((lambda x : x[0] <= 10), results.values()))
    success = list(filter((lambda x : x[0] <= max_num_results), results.values()))
    rank_sum = sum(map((lambda x : x[0]), results.values()))
    return (len(success), len(top_10), len(results.items()), rank_sum)

if __name__ == '__main__':
    kb = MultiKeyboardGraph(KeyboardType.SAMSUNG)

    #Tests
    # 0 - cc
    # 1 - date
    # 2 - sec
    test = 2

    if test == 0:
        print("building dict")
        ms_dict = build_cc_ms_dict(kb, 30, ty="discover")
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing num_val")
        nv_success, nv_tt, nv_total, nv_rank_sum = test_dict(
                                                    restore_dictionary("credit_card"), ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("nv total: " + str(nv_total))
        print("nv success: " + str(nv_success))
        print("nv tt: " + str(nv_tt))
        print("nv average: " + str(nv_rank_sum/nv_total))

    if test == 1:
        print("building dict")
        ms_dict = build_date_ms_dict(MultiKeyboardGraph(KeyboardType.SAMSUNG), 300)
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing exp")
        exp_success, exp_tt, exp_total, exp_rank_sum = test_dict(restore_dictionary("exp_date"), ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("exp total: " + str(exp_total))
        print("exp success: " + str(exp_success))
        print("exp tt: " + str(exp_tt))
        print("exp average: " + str(exp_rank_sum/exp_total))

    if test == 2:

        print("building dict")
        ms_dict = build_sec_ms_dict(MultiKeyboardGraph(KeyboardType.SAMSUNG), 300)
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing zip")
        val_success, val_tt, val_total, val_rank_sum = test_dict(restore_dictionary("cvv"), ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("val total: " + str(val_total))
        print("val success: " + str(val_success))
        print("val tt: " + str(val_tt))
        print("val average: " + str(val_rank_sum/val_total))
