import string
from argparse import ArgumentParser
from typing import List, Dict, Tuple
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph

from smarttvleakage.utils.constants import KeyboardType

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.file_utils import save_pickle_gz, read_pickle_gz


# manual score frequency dicts
def make_msfd(path : str) -> Dict[str, Tuple[str, float]]:
    """todo"""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        d[line.split(";")[0]] = (line.split(";")[1], float(line.split(";")[2]))

    return d

def save_msfd(input_path : str, save_path : str):
    """todo"""
    msfd = make_msfd(input_path)
    save_pickle_gz(msfd, save_path)

def build_msfd(path : str) -> Dict[str, Tuple[str, float]]:
    """todo"""
    return read_pickle_gz(path)




def save_ms_dict(save_path_start : str):
    """Saves hardcoded ms dicts"""
    # Non-autocomplete Dict
    ms_dict_non = {}
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
    ms_dict_non["fathers"] = [4, 3, 5, 2, 4, 1, 3]
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
    ms_dict_non["here"] = [6, 4, 1, 1]
    ms_dict_non["highly"] = [6, 3, 4, 1, 3, 4]
    ms_dict_non["in"] = [7, 4]
    ms_dict_non["last"] = [9, 8, 1, 4]
    ms_dict_non["live"] = [9, 2, 6, 3]
    ms_dict_non["lives"] = [9, 2, 6, 3, 2]
    ms_dict_non["long"] = [9, 1, 5, 2]
    ms_dict_non["measure"] = [8, 6, 3, 1, 6, 3, 1]
    ms_dict_non["met"] = [8, 6, 2]
    ms_dict_non["nation"] = [7, 6, 5, 3, 1, 5]
    ms_dict_non["never"] = [7, 5, 3, 3, 1]
    ms_dict_non["new"] = [7, 5, 1]
    ms_dict_non["nobly"] = [7, 5, 6, 5, 4]
    ms_dict_non["nor"] = [7, 5, 5]
    ms_dict_non["note"] = [7, 5, 4, 2]
    ms_dict_non["now"] = [7, 5, 7]
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
    ms_dict_auto["thus"] = [4, 1, 0, 0]
    ms_dict_auto["us"] = [6, 1]
    ms_dict_auto["war"] = [1, 1, 2]
    ms_dict_auto["whether"] = [1, 1, 2, 1, 0, 0, 0]
    ms_dict_auto["which"] = [1, 1, 8, 1, 0]
    ms_dict_auto["work"] = [1, 8, 1, 0]
    ms_dict_auto["year"] = [5, 1, 0, 1]

    save_pickle_gz(ms_dict_non, save_path_start + "_non.pkl.gz")
    save_pickle_gz(ms_dict_auto, save_path_start + "_auto.pkl.gz")

def build_ms_dict(path : str, take : int = 0) -> Dict[str, List[int]]:
    """Builds ms dict from path, takes either positive number or passes over negative"""
    base = read_pickle_gz(path)
    if take == 0:
        return base
    if take > 0: # take the first x
        ms_dict = {}
        for key in base:
            ms_dict[key] = base[key]
            take -= 1
            if take == 0:
                break
        return ms_dict
    # skip the first x
    ms_dict = {}
    for key in base:
        if take < 0:
            take += 1
            continue
        ms_dict[key] = base[key]
    return ms_dict


def get_word_from_ms(ms : List[int], msfd : Dict[str, Tuple[str, int]]) -> Tuple[str, float]:
    """Gets the most common word and freq. score from a move sequence"""
    ms_string = ""
    for m in ms:
        ms_string += str(m) + ","
    if ms_string in msfd:
        return msfd[ms_string]
    return ("", 0)


def build_rockyou_ms_dict(path, count, passover : int = 0):
    """Builds a ms dict from rockyou path"""
    rockyou_ms_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        i = 0
        for line in f:

            # to avoid breaking
            skip = 0
            for c in line:
                if c not in string.printable:
                    skip = 1
            if skip == 1:
                continue

            if passover > 0:
                passover -= 1
                continue

            path = []
            kb = MultiKeyboardGraph(KeyboardType.SAMSUNG)
            line = line.replace("\n", "").replace(" ", "").replace(";", "")
            print(line)
            for m in findPath(line, False, False, 0, 0, 0, kb):
                path.append(m.num_moves)
            rockyou_ms_dict[line] = path
            i += 1
            if i >= count:
                break
    return rockyou_ms_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--msfd-path", type=str, required=False)
    args = parser.parse_args()


    #save_msfd()
    #save_ms_dict("suggestions_model/local/ms_dict")

    i = 0
    msfd = build_msfd(args.msfd_path)
    for key in msfd:
        print(key)
        i += 1
        if i > 10:
            break

    print(get_word_from_ms([3, 5, 4], msfd)[0])
    print(float(get_word_from_ms([3, 5, 4], msfd)[1]))
    print(get_word_from_ms([6, 5, 1], msfd))
    print(get_word_from_ms([3, 5, 4, 1, 3], msfd))
    print(get_word_from_ms([1, 1, 1, 1], msfd))

    print("\n")
    print(get_word_from_ms([2, 1, 0, 0], msfd))
    print(get_word_from_ms([2, 8, 4, 6], msfd))
    print(get_word_from_ms([2, 3, 4, 6, 1, 5, 3], msfd))
    
