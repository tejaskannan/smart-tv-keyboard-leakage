
import string
from argparse import ArgumentParser
from typing import List, Dict, Tuple

from smarttvleakage.keyboard_utils.word_to_move import findPath

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.utils.constants import KeyboardType

from smarttvleakage.dictionary import EnglishDictionary
from smarttvleakage.utils.file_utils import read_json

from smarttvleakage.suggestions_model.manual_score_dict import build_ms_dict






def eval_ms(key : Tuple[str, str], ms : List[int], ms_dict_non, ms_dict_auto) -> int:
    """evaluates simulated ms against gt, allowing off-by-one for autos"""
    word = key[0]
    ty = key[1]

    if ty == "non":
        gt = ms_dict_non[word]
        if len(gt) != len(ms):
            return 0
        for i, m in enumerate(ms):
            if gt[i] != m:
                return 0
        return 1
    # ty == "auto"
    gt = ms_dict_auto[word]
    if len(gt) != len(ms):
        return 0
    for i, m in enumerate(ms):
        if abs(gt[i] - m) > 1:
            return 0
    return 1






def grab_words(count : int, path : str) -> List[str]:
    """Returns a list of [count] words from the file at path"""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            w = line.split(" ")[0]

            skip = 0
            for c in w:
                if c not in string.ascii_letters:
                    skip = 1
            if skip == 1:
                continue

            words.append(line.split(" ")[0])
            count -= 1
            if count <= 0:
                break
        f.close()

    return words

## todo** fix hardcoded autocomplete singlesuggestions file
def get_autos(e_dict, prefix : str) -> List[str]:
    """Returns simulated autocomplete suggestions using an EnglishDictionary"""
    if len(prefix) == 1: # then use single suggestions
        single_suggestions = read_json("graphs/autocomplete.json")
        return single_suggestions[prefix.lower()]

    char_dict = e_dict.get_letter_counts(prefix, None)
    char_list = []
    for c in char_dict:
        char_list.append((c, char_dict[c]))
    char_list.sort(key=(lambda x: x[1]), reverse=True)

    suggestions = []
    for i in range(4):
        if i < len(char_list):
            suggestions.append(char_list[i][0])
        else:
            break

    return suggestions

def find_path_auto(e_dict, word : str) -> List[int]:
    """Simulates the path for a word using autocomplete"""
    path = []
    keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)
    mode = keyboard.get_start_keyboard_mode()
    prev = START_KEYS[mode]
    last_auto = 0
    for i in list(word.lower()):
        if len(path) > 0:
            autos = get_autos(e_dict, word[:len(path)])
            if autos == []: # Ensure that autos is not empty
                autos.append("\t")

            if i == autos[0]:
                if last_auto == 1:
                    path.append(0)
                else:
                    path.append(1)
                last_auto = 1
            elif i in autos:
                path.append(1)

                last_auto = 1
            else:
                distance = keyboard.get_moves_from_key(prev, i, False, False, mode)
                print(distance)
                while distance == -1:
                    path.append(keyboard.get_moves_from_key(prev, "<CHANGE>", False, False, mode))
                    prev = "<CHANGE>"
                    mode = get_keyboard_mode(prev, mode, keyboard_type=KeyboardType.SAMSUNG)
                    prev = START_KEYS[mode]
                    distance = keyboard.get_moves_from_key(prev, i, False, False, mode)
                if last_auto == 1:
                    distance += 1
                if i == " ":
                    path.append(distance)
                else:
                    path.append(distance)
                prev = i
                last_auto = 0
        else:
            distance = keyboard.get_moves_from_key(prev, i, False, False, mode)
            print(distance)
            while distance == -1:
                #print(i)
                #print(path)
                path.append(keyboard.get_moves_from_key(prev, "<CHANGE>", False, False, mode))
                prev = "<CHANGE>"
                mode = get_keyboard_mode(prev, mode, keyboard_type=KeyboardType.SAMSUNG)
                prev = START_KEYS[mode]
                distance = keyboard.get_moves_from_key(prev, i, False, False, mode)
            if i == " ":
                path.append(distance)
            else:
                path.append(distance)
            prev = i
    return path





def simulate_ms(dict, word : str, auto : bool) -> List[int]:
    """Simulates a move sequence"""
    keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)
    if not auto:
        move = findPath(word, 0, False, False, 0, 0, keyboard)
        ms = []
        for m in move:
            ms.append(m.num_moves)
    else:
        ms = find_path_auto(dict, word)
    return ms


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False)
    parser.add_argument("--ms-path-non", type=str, required=False)
    parser.add_argument("--ed-path", type=str, required=False)
    args = parser.parse_args()

    # tests
    # 2 - deprecated strategy test
    # 3 - strategy test
    test = 3

    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
        ms_dict_auto = build_ms_dict(args.ms_path_auto)
    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_auto)
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


    if test == 1:
        # test non words
        for key in ms_dict_non:
            sim_ms = simulate_ms(englishDictionary, key, False)
            if sim_ms != ms_dict_non[key]:
                print("failed sim on: " + key)
                print("gt: ", end = "")
                print(ms_dict_non[key])
                print("sim: ", end="")
                print(sim_ms)


        # test auto words
        for key in ms_dict_auto:
            if len(key) == 5:
                sim_ms = simulate_ms(englishDictionary, key, True)
                if sim_ms != ms_dict_auto[key]:
                    print("failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim: ", end="")
                    print(sim_ms)
                else:
                    print("success on " + key)
