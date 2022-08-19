from argparse import ArgumentParser
from smarttvleakage.dictionary.dictionaries import EnglishDictionary

from smarttvleakage.suggestions_model.generate_cc import (generate_date,
                                                        generate_sec, add_acc, finish_cc,
                                                        build_valid_cc_list, generate_zip)
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.constants import KeyboardType, SmartTVType

from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.audio.constants import SAMSUNG_KEY_SELECT

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import NumericDictionary, restore_dictionary

# Build Testing Dicts
def build_cc_ms_dict(kb, count : int) -> dict[str, list[int]]:
    """Build an ms dictionary of valid cc numbers"""
    ms_dict = {}
    valid_bins = build_valid_cc_list("cc_bin_dict000")
    for i in range(min(count, len(valid_bins))):
        b = int(valid_bins[i])
        cc = str(finish_cc(add_acc(b)))
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

def build_zip_ms_dict(kb, count : int, zip_path : str) -> dict[str, list[int]]:
    """Build an ms dictionary of valid zip codes"""
    ms_dict = {}
    for _ in range(count):
        zip_code = generate_zip(zip_path)
        path = []
        for m in findPath(zip_code, False, False, 0, 0, 0, kb):
            path.append(m.num_moves)
        ms_dict[zip_code] = path
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
    parser = ArgumentParser()
    parser.add_argument("--zip-text-path", type=str, required=False) #zip .txt
    parser.add_argument("--zip-dict-path", type=str, required=False) #zip dict pkl.gz
    args = parser.parse_args()

    if args.zip_text_path is None:
        args.zip_text_path = "suggestions_model/local/zip_codes.txt"
    if args.zip_dict_path is None:
        args.zip_dict_path = "suggestions_model/local/zip_codes.txt"
    elif args.zip_dict_path == "build":
        zipDictionary = EnglishDictionary(5)
        zipDictionary.build(
             "suggestions_model/local/zip_codes.txt", 5, True)
        zipDictionary.save("suggestions_model/local/dictionaries/zip_codes.pkl.gz")
        args.zip_dict_path = "suggestions_model/local/dictionaries/zip_codes.pkl.gz"

    kb = MultiKeyboardGraph(KeyboardType.SAMSUNG)

    #Tests
    # 0 - cc
    # 1 - date
    # 2 - sec
    # 3 - zip
    # 4 - total
    test = 4

    #uni success: 50485
    #uni tt: 50409
    #uni rank average: 37619967.12157895
    #spec success: 591
    #spec tt: 577
    #spec rank average: 2171.5645
    if test == 4:
        print("building dicts")
        ms_dict_cc = build_cc_ms_dict(kb, 100)
        ms_dict_date = build_date_ms_dict(MultiKeyboardGraph(KeyboardType.SAMSUNG), 100)
        ms_dict_cvv = build_sec_ms_dict(MultiKeyboardGraph(KeyboardType.SAMSUNG), 100)
        ms_dict_zip = build_zip_ms_dict(
                                MultiKeyboardGraph(KeyboardType.SAMSUNG), 100, args.zip_text_path)
        print("built dicts")

        print("testing uniform")
        uni_success_cc, uni_tt_cc, uni_total_cc, uni_rank_sum_cc = (
            test_dict(NumericDictionary(), ms_dict_cc))
        uni_success_date, uni_tt_date, uni_total_date, uni_rank_sum_date = (
            test_dict(NumericDictionary(), ms_dict_date))
        uni_success_cvv, uni_tt_cvv, uni_total_cvv, uni_rank_sum_cvv = (
            test_dict(NumericDictionary(), ms_dict_cvv))
        uni_success_zip, uni_tt_zip, uni_total_zip, uni_rank_sum_zip = (
            test_dict(NumericDictionary(), ms_dict_zip))
        print("testing spec")
        spec_success_cc, spec_tt_cc, spec_total_cc, spec_rank_sum_cc = (
            test_dict(restore_dictionary("credit_card"), ms_dict_cc))
        spec_success_date, spec_tt_date, spec_total_date, spec_rank_sum_date = (
            test_dict(restore_dictionary("exp_date"), ms_dict_date))
        spec_success_cvv, spec_tt_cvv, spec_total_cvv, spec_rank_sum_cvv = (
            test_dict(restore_dictionary("cvv"), ms_dict_cvv))
        spec_success_zip, spec_tt_zip, spec_total_zip, spec_rank_sum_zip = (
            test_dict(zipDictionary, ms_dict_zip))

        uni_success = uni_success_cc + uni_rank_sum_cvv + uni_rank_sum_date + uni_rank_sum_zip
        uni_tt = uni_tt_cc + uni_total_date + uni_rank_sum_cvv + uni_rank_sum_zip
        uni_average = (
            (uni_rank_sum_cc * uni_rank_sum_date * uni_rank_sum_cvv * uni_rank_sum_zip)/
            (uni_total_cc * uni_total_date * uni_total_cvv * uni_total_zip))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni rank average: " + str(uni_average))

        spec_success = spec_success_cc + spec_rank_sum_cvv + spec_rank_sum_date + spec_rank_sum_zip
        spec_tt = spec_tt_cc + spec_total_date + spec_rank_sum_cvv + spec_rank_sum_zip
        spec_average = (
            (spec_rank_sum_cc * spec_rank_sum_date * spec_rank_sum_cvv * spec_rank_sum_zip)/
            (spec_total_cc * spec_total_date * spec_total_cvv * spec_total_zip))
        print("spec success: " + str(spec_success))
        print("spec tt: " + str(spec_tt))
        print("spec rank average: " + str(spec_average))

    if test == 0:
        print("building dict")
        ms_dict = build_cc_ms_dict(kb, 30)
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing cc")
        nv_success, nv_tt, nv_total, nv_rank_sum = test_dict(
                                                    restore_dictionary("credit_card"), ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("cc dict total: " + str(nv_total))
        print("cc dict success: " + str(nv_success))
        print("cc dict tt: " + str(nv_tt))
        print("cc dict average: " + str(nv_rank_sum/nv_total))

    if test == 1:
        print("building dict")
        ms_dict = build_date_ms_dict(MultiKeyboardGraph(KeyboardType.SAMSUNG), 300)
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing exp")
        exp_success, exp_tt, exp_total, exp_rank_sum = test_dict(
                                                                restore_dictionary("exp_date"), ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("exp date dict total: " + str(exp_total))
        print("exp date dict success: " + str(exp_success))
        print("exp date dict tt: " + str(exp_tt))
        print("exp date dict average: " + str(exp_rank_sum/exp_total))

    if test == 2:
        print("building dict")
        ms_dict = build_sec_ms_dict(MultiKeyboardGraph(KeyboardType.SAMSUNG), 300)
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing cvv dict")
        val_success, val_tt, val_total, val_rank_sum = test_dict(restore_dictionary("cvv"), ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("cvv dict total: " + str(val_total))
        print("cvv dict success: " + str(val_success))
        print("cvv dict tt: " + str(val_tt))
        print("cvv dict average: " + str(val_rank_sum/val_total))

    if test == 3:
        print("building dict")
        ms_dict = build_zip_ms_dict(
                                MultiKeyboardGraph(KeyboardType.SAMSUNG), 300, args.zip_text_path)
        print("built dict")

        print("testing numeric")
        uni_success, uni_tt, uni_total, uni_rank_sum = test_dict(NumericDictionary(), ms_dict)
        print("testing zip dict")
        zipDictionary = EnglishDictionary(5)
        zipDictionary.build(args.zip_dict_path, 5, True)
        val_success, val_tt, val_total, val_rank_sum = test_dict(zipDictionary, ms_dict)

        print("uni total: " + str(uni_total))
        print("uni success: " + str(uni_success))
        print("uni tt: " + str(uni_tt))
        print("uni average: " + str(uni_rank_sum/uni_total))

        print("zip dict total: " + str(val_total))
        print("zip dict success: " + str(val_success))
        print("zip dict tt: " + str(val_tt))
        print("zip dict average: " + str(val_rank_sum/val_total))
