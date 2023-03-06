
import string
import random 

from argparse import ArgumentParser
from typing import List, Dict, Tuple

from smarttvleakage.keyboard_utils.word_to_move import findPath

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.utils.constants import KeyboardType, Direction


from smarttvleakage.dictionary import EnglishDictionary, restore_dictionary
from smarttvleakage.utils.file_utils import read_json

from smarttvleakage.suggestions_model.manual_score_dict import build_ms_dict

from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.audio.sounds import SAMSUNG_DELETE, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT





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
def get_autos(e_dict, ss_path : str, prefix : str, count : int = 4) -> List[str]:
    """Returns simulated autocomplete suggestions using an EnglishDictionary"""
    if len(prefix) == 1: # then use single suggestions
        single_suggestions = read_json(ss_path)
        return single_suggestions[prefix.lower()]

    char_dict = e_dict.get_letter_counts(prefix, None)
    char_list = []
    for c in char_dict:
        char_list.append((c, char_dict[c]))
    char_list.sort(key=(lambda x: x[1]), reverse=True)

    suggestions = []
    for i in range(count):
        if i < len(char_list):
            suggestions.append(char_list[i][0])
        else:
            break

    return suggestions

def find_path_auto(e_dict, ss_path, word : str) -> List[int]:
    """Simulates the path for a word using autocomplete"""
    path = []
    keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)
    mode = keyboard.get_start_keyboard_mode()
    prev = START_KEYS[mode]
    last_auto = 0
    for i in list(word.lower()):
        if len(path) > 0:
            autos = get_autos(e_dict, ss_path, word[:len(path)])
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


def add_mistakes(ms : List[int], count : int) -> List[int]:
    if len(ms) == 0:
        return ms
        
    while (count > 0):
        place = random.randint(0, len(ms)-1)
        ms[place] += 2
        count -= 1

    return ms
    


def simulate_ms(dict, ss_path : str, word : str,
    auto : bool, mistakes : int = 0) -> List[int]:
    """Simulates a move sequence"""
    keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)
    if not auto:
        move = findPath(word, False, False, False, 0, 0, 0, keyboard, "q")
        ms = []
        for m in move:
            ms.append(m.num_moves)
    else:
        ms = find_path_auto(dict, ss_path, word)

    add_mistakes(ms, mistakes)

    return ms


def add_mistakes_to_ms_dict(dict):
    new_dict = {}
    for word, ms in dict.items():
        rand = random.randint(0, 99)
        if rand < 75:
            new_dict[word] = ms
        elif rand < 90:
            new_dict[word] = add_mistakes(ms, 1)
        elif rand < 97:
            new_dict[word] = add_mistakes(ms, 2)
        else:
            new_dict[word] = add_mistakes(ms, 3)

    return new_dict


# For each move, 
# k - key select
# f - normal done
# s - suggestions done
# d - delete
# c - change keyboard / caps
def build_gt_ms_dict():
    # 0 - web, 1 - pass
    ms = {}
    
    #Subject B
    ms["B"] = {}
    ms["B"][1] = {}

    #WEB
    ms["B"][1][0] = [(8, "k"), (1, "k"), (5, "k"), (6, "k"), (1, "k"), (0, "s")]
    ms["B"][1][1] = [(4, "k"), (1, "k"), (6, "k"), (1, "k"), (0, "k"), (2, "k"), (0, "k"), (0, "s")]
    ms["B"][1][2] = [(9, "k"), (2, "k"), (2, "k"), (6, "k"), (1, "k"), (0, "k"), (0, "k"), (8, "f")]
    ms["B"][1][3] = [(4, "k"), (2, "k"), (5, "k"), (6, "k"), (3, "k"), (5, "f")]
    ms["B"][1][4] = [(3, "k"), (6, "k"), (4, "k"), (1, "k"), (0, "k"), (0, "s")]
    ms["B"][1][5] = [(4, "k"), (5, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"), (0, "k"), (5, "f")]
    ms["B"][1][6] = [(1, "k"), (6, "k"), (0, "k"), (1, "k"), (0, "k"), (2, "k"), (9, "f")]
    ms["B"][1][7] = [(8, "k"), (2, "k"), (5, "k"), (1, "k"), (1, "k"), (5, "f")]
    ms["B"][1][8] = [(1, "k"), (1, "k"), (1, "k"), (1, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"), (3, "f")]
    ms["B"][1][9] = [(1, "k"), (8, "k"), (1, "k"), (2, "k"), (0, "k"), (5, "f")]


    # PASS
    ms["B"][0] = {}
    ms["B"][0][0] = [(4, "k"), (2, "k"), (3, "k"), (3, "k"), (6, "k"), (6, "k"), (1, "k"),
                     (5, "k"), (5, "k"), (4, "f")] # double check
    ms["B"][0][1] = [(13, "k"), (3, "k"), (8, "k"), (4, "k"), (5, "k"), (7, "k"), (3, "k"), (8, "f")]
    ms["B"][0][2] = [(3, "k"), (1, "k"), (3, "k"), (6, "k"), (6, "k"), (7, "k"), (3, "k"), (9, "f")]
    ms["B"][0][3] = [(1, "k"), (9, "k"), (9, "k"), (5, "k"), (5, "k"), (5, "d"), (3, "k"), (3, "k"), 
                     (4, "k"), (6, "f")]
    ms["B"][0][4] = [(9, "k"), (2, "k"), (4, "k"), (6, "k"), (2, "k"), (3, "s"), (1, "k"), (1, "k"), 
                     (1, "k"), (1, "s")]
    ms["B"][0][5] = [(3, "k"), (1, "k"), (1, "k"), (10, "k"), (7, "k"), (4, "k"), (7, "k"), (12, "d"), 
                     (7, "k"), (1, "k"), (6, "f")]
    ms["B"][0][6] = [(7, "k"), (5, "k"), (3, "k"), (5, "k"), (8, "k"), (7, "k"), (6, "k"), (1, "s")]
    ms["B"][0][7] = [(6, "k"), (4, "k"), (3, "k"), (2, "k"), (1, "k"), (1, "k"), (10, "k"), (10, "k"), 
                     (1, "2")]
    ms["B"][0][8] = [(2, "k"), (8, "k"), (1, "k"), (9, "k"), (1, "k"), (3, "k"), (8, "k"), (4, "f")]
    ms["B"][0][9] = [(2, "k"), (1, "k"), (1, "k"), (5, "k"), (4, "k"), (4, "k"), (5, "k"), (6, "f")]




    # Subject C
    ms["C"] = {}
    ms["C"][1] = {}

    #WEB
    ms["C"][1][0] = [(1, "k"), (4, "k"), (0, "k"), (1, "k"), (0, "k"), (0, "k"), (9, "k"),
                    (1, "k"), (0, "k"), (0, "k"), (2, "s")]
    ms["C"][1][1] = [(4, "k"), (1, "k"), (2, "k"), (0, "k"), (0, "k"), (5, "k"), (0, "k"),
                    (2, "k"), (1, "s")]
    ms["C"][1][2] = [(5, "k"), (1, "k"), (3, "k"), (1, "k"), (2, "k"), (2, "k"), (2, "k"),
                    (3, "s")]
    ms["C"][1][3] = [(8, "k"), (8, "k"), (4, "k"), (4, "k"), (1, "k"), (13, "f")]
    ms["C"][1][4] = [(2, "k"), (1, "k"), (5, "k"), (1, "k"), (0, "k"), (2, "k"), (7, "k"),
                    (4, "k"), (3, "s")]
    ms["C"][1][5] = [(5, "k"), (2, "k"), (4, "k"), (5, "k"), (3, "k"), (1, "k"), (0, "k"),
                    (0, "k"), (0, "k"), (2, "k"), (0, "k"), (1, "s")]
    ms["C"][1][6] = [(2, "k"), (6, "k"), (1, "k"), (7, "k"), (1, "k"), (0, "k"), (0, "k"),
                    (1, "s")]
    ms["C"][1][7] = [(1, "k"), (6, "k"), (6, "k"), (1, "k"), (0, "k"), (7, "f")]
    ms["C"][1][8] = [(2, "k"), (5, "k"), (1, "k"), (1, "k"), (0, "k"), (1, "s")]
    ms["C"][1][9] = [(4, "k"), (3, "k"), (5, "k"), (3, "k"), (1, "k"), (1, "s")]

    # PASSWORDs

    ms["C"][0] = {}
    ms["C"][0][0] = [(2, "k"), (1, "k"), (4, "k"), (1, "k"), (12, "k"), (3, "k"), (1, "k"),
                     (3, "k"), (9, "f")]
    ms["C"][0][1] = [(9, "k"), (6, "k"), (1, "k"), (2, "k"), (2, "k"), (5, "k"), (4, "k"),
                    (1, "k"), (1, "k"), (20, "c"), (3, "k"), (4, "c"), (8, "k"), (5, "k"),
                    (8, "k"), (12, "f")]
    ms["C"][0][2] = [(2, "c"), (3, "k"), (1, "k"), (12, "k"), (10, "c"), (9, "k"), (7, "k"),
                    (5, "k"), (3, "k"), (4, "k"), (4, "f")]
    ms["C"][0][3] = [(9, "k"), (7, "k"), (5, "k"), (4, "k"), (4, "k"), (3, "k"), (2, "k"),
                    (1, "k"), (4, "k"), (6, "k"), (2, "k"), (9, "f")]
    ms["C"][0][4] = [(8, "k"), (6, "k"), (1, "k"), (8, "k"), (2, "k"), (1, "k"), (4, "k"),
                    (6, "k"), (7, "k"), (11, "k"), (12, "d"), (11, "k"), (1, "s")]
    ms["C"][0][5] = [(13, "k"), (7, "k"), (7, "k"), (8, "k"), (6, "k"), (1, "k"), (7, "k"),
                    (8, "k"), (2, "f")]
    ms["C"][0][6] = [(3, "k"), (3, "k"), (5, "k"), (8, "k"), (3, "k"), (5, "k"), (0, "k"),
                    (0, "k"), (8, "f")]
    ms["C"][0][7] = [(6, "k"), (3, "k"), (3, "k"), (3, "k"), (5, "k"), (9, "c"), (7, "k"),
                    (7, "c"), (8, "k"), (5, "k"), (3, "k"), (4, "k"), (1, "k"), (10, "f")]
    ms["C"][0][8] = [(4, "k"), (8, "k"), (8, "k"), (8, "k"), (9, "k"), (2, "c"), (7, "k"),
                    (7, "c"), (2, "k"), (8, "k"), (4, "k"), (4, "k"), (6, "k"), (5, "k"), (10, "f")]
    ms["C"][0][9] = [(1, "c"), (4, "k"), (2, "k"), (4, "c"), (6, "k"), (3, "k"), (1, "k"),
                    (1, "k"), (4, "c"), (3, "k"), (2, "k"), (11, "f")]
    



    # Subject D
    ms["D"] = {}
    
    # Web
    ms["D"][1] = {}
    ms["D"][1][0] = [(4, "k"), (3, "k"), (6, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"),
                     (7, "f")]
    ms["D"][1][1] = [(5, "k"), (1, "k"), (0, "k"), (2, "k"), (0, "k"), (4, "k"), (1, "k"),
                     (2, "s")]
    ms["D"][1][2] = [(9, "k"), (1, "k"), (6, "k"), (1, "k"), (0, "k"), (6, "k"), (1, "k"),
                     (0, "s")]
    ms["D"][1][3] = [(6, "k"), (6, "k"), (1, "k"), (4, "k"), (8, "k"), (1, "s")]
    ms["D"][1][4] = [(2, "k"), (3, "k"), (5, "k"), (3, "k"), (0, "k"), (0, "k"), (11, "f")]
    ms["D"][1][5] = [(6, "k"), (1, "k"), (2, "k"), (0, "k"), (1, "k"), (2, "s")]
    ms["D"][1][6] = [(3, "k"), (1, "k"), (11, "k"), (1, "k"), (0, "k"), (1, "s")]
    ms["D"][1][7] = [(6, "k"), (1, "k"), (2, "k"), (8, "k"), (1, "k"), (0, "k"), (0, "k"),
                     (3, "k"), (2, "k"), (0, "s")]
    ms["D"][1][8] = [(9, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"), (6, "f")]
    ms["D"][1][9] = [(4, "k"), (1, "k"), (0, "k"), (2, "k"), (0, "k"), (9, "f")]


    # PASS
    ms["D"][0] = {}

    ms["D"][0][0] = [(2, "k"), (8, "k"), (7, "k"), (1, "k"), (6, "k"), (6, "k"), (1, "k"),
                     (4, "k"), (1, "s")]
    ms["D"][0][1] = [(0, "k"), (1, "k"), (1, "k"), (1, "k"), (1, "k"), (6, "k"), (6, "k"),
                     (3, "k"), (0, "k"), (11, "d"), (3, "f")] # Show Password...
    ms["D"][0][2] = [(2, "c"), (11, "k"), (10, "k"), (6, "k"), (3, "k"), (2, "k"), (4, "k"),
                     (6, "k"), (12, "k"), (15, "k"), (0, "k"), (0, "k"), (1, "s")]
    ms["D"][0][3] = [(8, "k"), (7, "k"), (4, "k"), (1, "k"), (3, "k"), (4, "k"), (3, "k"),
                     (1, "k"), (1, "s")]
    ms["D"][0][4] = [(9, "k"), (8, "k"), (2, "k"), (4, "k"), (1, "k"), (2, "k"), (7, "k"),
                     (3, "k"), (1, "s")]
    ms["D"][0][5] = [(1, "k"), (4, "k"), (4, "c"), (3, "k"), (1, "k"), (4, "c"), (9, "k"),
                     (7, "k"), (10, "k"), (1, "d"), (10, "f")]
    ms["D"][0][6] = [(8, "k"), (6, "k"), (4, "k"), (4, "k"), (4, "k"), (6, "k"), (2, "k"),
                     (7, "k"), (8, "k"), (1, "f")]
    ms["D"][0][7] = [(1, "k"), (0, "k"), (0, "k"), (12, "k"), (4, "k"), (2, "k"), (1, "k"),
                     (1, "k"), (1, "s")]
    ms["D"][0][8] = [(2, "k"), (3, "k"), (3, "k"), (1, "k"), (2, "k"), (6, "k"), (4, "k"),
                     (6, "k"), (1, "s")]
    ms["D"][0][9] = [(4, "k"), (4, "k"), (5, "k"), (3, "k"), (6, "k"), (4, "k"), (6, "k"),
                     (7, "k"), (2, "k"), (3, "k"), (19, "f")] # 15 show password... before last
    
    


    # Subject H

    ms["H"] = {}

    # Web
    ms["H"][1] = {}
    ms["H"][1][0] = [(1, "k"), (1, "k"), (0, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"),
                     (0, "k"), (0, "k"), (0, "k"), (0, "s")]
    ms["H"][1][1] = [(6, "k"), (1, "k"), (2, "k"), (4, "k"), (1, "k"), (4, "k"), (0, "k"),
                     (0, "k"), (1, "s")]
    ms["H"][1][2] = [(5, "k"), (1, "k"), (2, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"),
                     (2, "s")]
    ms["H"][1][3] = [(8, "k"), (1, "k"), (5, "k"), (4, "k"), (1, "k"), (0, "s")]
    ms["H"][1][4] = [(2, "k"), (1, "k"), (2, "k"), (2, "k"), (0, "k"), (1, "k"), (1, "k"),
                     (1, "k"), (1, "s")]
    ms["H"][1][5] = [(3, "k"), (1, "k"), (9, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"),
                     (0, "k"), (0, "k"), (1, "k"), (0, "k"), (12, "f")]
    ms["H"][1][6] = [(0, "k"), (1, "k"), (1, "k"), (1, "k"), (0, "k"), (0, "k"), (0, "k"),
                     (1, "s")]
    ms["H"][1][7] = [(1, "k"), (1, "k"), (8, "k"), (1, "k"), (0, "k"), (7, "f")]
    ms["H"][1][8] = [(2, "k"), (1, "k"), (1, "k"), (1, "k"), (0, "k"), (1, "s")]
    ms["H"][1][9] = [(4, "k"), (3, "k"), (8, "k"), (1, "k"), (0, "k"), (1, "s")]



    # Pass
    ms["H"][0] = {}
    ms["H"][0][0] = [(2, "k"), (1, "k"), (4, "k"), (1, "k"), (6, "k"), (3, "k"), (1, "k"),
                     (3, "k"), (11, "f")]
    ms["H"][0][1] = [(9, "k"), (6, "k"), (1, "k"), (4, "k"), (2, "k"), (5, "k"), (4, "k"),
                     (1, "k"), (1, "k"), (5, "c"), (15, "c"), (16, "c"), (3, "k"), (4, "c"),
                     (8, "k"), (5, "k"), (8, "k"), (1, "s")]
    ms["H"][0][2] = [(2, "c"), (3, "k"), (3, "k"), (8, "k"), (10, "c"), (9, "k"), (9, "c"),
                     (0, "c"), (8, "k"), (5, "k"), (3, "k"), (6, "k"), (1, "s")]
    ms["H"][0][3] = [(7, "k"), (7, "k"), (5, "k"), (4, "k"), (4, "k"), (3, "k"), (2, "k"),
                     (1, "k"), (4, "k"), (5, "k"), (2, "k"), (1, "s")]
    ms["H"][0][4] = [(8, "k"), (8, "k"), (1, "k"), (8, "k"), (2, "k"), (1, "k"), (4, "k"),
                     (9, "k"), (1, "k"), (7, "k"), (10, "k"), (1, "s")]
    ms["H"][0][5] = [(5, "k"), (11, "k"), (10, "k"), (8, "k"), (6, "k"), (1, "k"), (6, "k"),
                     (8, "k"), (2, "f")]
    ms["H"][0][6] = [(3, "k"), (3, "k"), (5, "k"), (8, "k"), (3, "k"), (5, "k"), (0, "k"),
                     (0, "k"), (5, "f")]
    ms["H"][0][7] = [(6, "k"), (3, "k"), (3, "k"), (3, "k"), (5, "k"), (9, "c"), (7, "k"),
                     (7, "c"), (8, "k"), (5, "k"), (3, "k"), (3, "k"), (1, "k"), (7, "f")]
    ms["H"][0][8] = [(4, "k"), (9, "k"), (10, "k"), (9, "k"), (7, "k"), (2, "c"), (7, "k"),
                     (7, "c"), (2, "k"), (6, "k"), (4, "k"), (4, "k"), (6, "k"), (5, "k"),
                     (1, "s")]
    ms["H"][0][9] = [(1, "c"), (4, "k"), (2, "k"), (4, "c"), (6, "k"), (3, "k"), (1, "k"),
                     (1, "k"), (4, "c"), (3, "k"), (2, "k"), (11, "f")]


    return ms




def build_moves(ms):
    moves = []
    for m in ms:
        ty = m[1]
        print(ty)
        if ty in ["k", "s"]:
            sound = "key_select"
        elif ty in ["d"]:
            sound = "select"
        else:
            sound = "select"

        moves.append(Move(num_moves=m[0], end_sound=sound, directions=Direction.ANY))
    return moves

# tests for comparing lists




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-path-auto", type=str, required=False)
    parser.add_argument("--ms-path-non", type=str, required=False)
    parser.add_argument("--ed-path", type=str, required=False)
    parser.add_argument("--ss-path", type=str, required=False)
    args = parser.parse_args()

    # tests
    # 2 - deprecated strategy test
    # 3 - strategy test

    # 10 - autocomplete suggestions test
    test = 10



    if args.ms_path_auto is None:
        args.ms_path_auto = "suggestions_model/local/ms_dict_auto.pkl.gz"
        ms_dict_auto = build_ms_dict(args.ms_path_auto)
    if args.ms_path_non is None:
        args.ms_path_non = "suggestions_model/local/ms_dict_non.pkl.gz"
        ms_dict_non = build_ms_dict(args.ms_path_auto)
    if args.ed_path is None:
        englishDictionary = restore_dictionary(
            "suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "build":
        englishDictionary = EnglishDictionary(50)
        englishDictionary.build(
            "suggestions_model/local/dictionaries/enwiki-20210820-words-frequency.txt", 50, True, False)
        englishDictionary.save("suggestions_model/local/dictionaries/ed.pkl.gz")
    elif args.ed_path == "ry":
        rockyouDictionary = NgramDictionary.restore("suggestions_model/local/rockyou_dict_updated.pkl.gz")
    elif args.ed_path != "skip":
        englishDictionary = EnglishDictionary.restore(args.ed_path)
    if args.ss_path is None:
        args.ss_path = "graphs/autocomplete.json"




    if test == 10:
        suggest_dict = {}

        suggest_dict["m"] = ["a", "o", "y", "e"]
        suggest_dict["mo"] = ["n", "r", "v", "t"]
        suggest_dict["moo"] = ["s", "t", "d", "n"]
        suggest_dict["moor"] = ["o", "h", "e", "s"]
        suggest_dict["moore"] = ["d"] # more

        suggest_dict["f"] = ["i", "e", "r", "o"]
        suggest_dict["fe"] = ["b", "w", "l", "a"]
        suggest_dict["fed"] = ["s", "e"] #
        suggest_dict["fede"] = ["x", "r"] #
        suggest_dict["feder"] = ["l", "i", "e", "a"]
        suggest_dict["federe"] = ["r"]
        suggest_dict["federer"] = []

        suggest_dict["p"] = ["u", "r", "e", "l"]
        suggest_dict["po"] = ["w", "s", "i", "l"]
        suggest_dict["pop"] = ["s", "e", "u"]
        suggest_dict["popu"] = ["p", "l"]
        suggest_dict["popul"] = ["a", "o", "i"]
        suggest_dict["popula"] = ["c", "t", "r"]
        suggest_dict["popular"] = ["l", "i"]

        suggest_dict["t"] = ["i", "a", "o", "h"]
        suggest_dict["tr"] = ["i", "o", "u", "y"]
        suggest_dict["tri"] = ["c", "a", "p", "e"]
        suggest_dict["trie"] = ["r", "n", "s", "d"]
        suggest_dict["tries"] = ["t"]

        suggest_dict["r"] = ["o", "u", "e", "i"]
        suggest_dict["ro"] = ["u", "l", "c", "a"]
        suggest_dict["roy"] = ["c", "s", "'", "a"]
        suggest_dict["roya"] = ["l"]
        suggest_dict["royal"] = ["t","i"]

        # Skip "t"
        suggest_dict["to"] = ["g", "n", "d", "l"]
        suggest_dict["ton"] = ["y", "i", "g", "e"]
        suggest_dict["toni"] = ["c", "g"]
        suggest_dict["tonig"] = ["h"]
        suggest_dict["tonigh"] = ["t"]
        suggest_dict["tonight"] = ["'"]

        suggest_dict["a"] = ["s", "r", "l", "n"]
        suggest_dict["at"] = ["r", "c", "o", "e"]
        suggest_dict["att"] = ["o", "i", "e", "a"]
        suggest_dict["atte"] = ["m", "n", "s"]
        suggest_dict["atten"] = ["b", "d", "t", "u"]
        suggest_dict["attend"] = ["a", "i", "e", "s"]

        # Skip "m"
        suggest_dict["me"] = ["t", "s", "n", "a"]
        suggest_dict["med"] = ["a", "l", "s", "i"]
        suggest_dict["medi"] = ["a", "t", "u", "c"]
        suggest_dict["media"] = ["'", "n"]

        suggest_dict["a"] = ["s", "r", "l", "n"]
        suggest_dict["an"] = ["o", "s", "y", "d"]
        suggest_dict["ana"] = ["t", "g", "r", "l"]
        suggest_dict["anal"] = ["g", "y", "o"]
        suggest_dict["analy"] = ["z", "s", "t"]
        suggest_dict["analys"] = ["t", "i", "e"]
        suggest_dict["analysi"] = ["n", "s"]
        suggest_dict["analysis"] = []

        suggest_dict["w"] = ["i", "a", "e", "h"]
        suggest_dict["wo"] = ["m", "w", "r", "u"]
        suggest_dict["wor"] = ["t", "l", "k", "d"]
        suggest_dict["worl"] = ["e", "d"]
        suggest_dict["world"] = ["s", "'", "l", "w"]




        for count in [2, 3, 4, 5, 6, 7, 8, 9]:
            found = 0
            total_gt = 0
            total_guess = 0
            for key, val in suggest_dict.items():
                autos = get_autos(e_dict=englishDictionary, ss_path=args.ss_path, prefix=key, count=count)

                if len(val) != 0:
                    for c in val:
                        if c in autos:
                            found += 1
                        total_gt += 1
                total_guess += 4
            
            print("count: " + str(count))
            print("acc: " + str(found/total_gt))
            print("over_guess avg: " + str((total_guess - found) / (total_guess/count)))




    if test == 1:
        # test non words
        for key in ms_dict_non:
            sim_ms = simulate_ms(englishDictionary, args.ss_path, key, False)
            if sim_ms != ms_dict_non[key]:
                print("failed sim on: " + key)
                print("gt: ", end = "")
                print(ms_dict_non[key])
                print("sim: ", end="")
                print(sim_ms)


        # test auto words
        for key in ms_dict_auto:
            if len(key) == 5:
                sim_ms = simulate_ms(englishDictionary, args.ss_path, key, True)
                if sim_ms != ms_dict_auto[key]:
                    print("failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim: ", end="")
                    print(sim_ms)
                else:
                    print("success on " + key)
