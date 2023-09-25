from distutils.command.build import build
import string
from argparse import ArgumentParser
from typing import List, Dict, Tuple
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph

from smarttvleakage.utils.constants import KeyboardType

from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.utils.file_utils import save_pickle_gz, read_pickle_gz




def build_ms_dict(path : str, take : int = 0, passover : int = 0) -> Dict[str, List[int]]:
    """Builds ms dict from path, takes either positive number or passes over negative"""
    base = read_pickle_gz(path)
    if take == 0:
        return base
    if take > 0: # take the first x
        ms_dict = {}
        for key in base:
            if passover > 0:
                passover -= 1
                continue

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


#def get_word_from_ms(ms : List[int], msfd : Dict[str, Tuple[str, int]]) -> Tuple[str, float]:
#    """Gets the most common word and freq. score from a move sequence"""
#    ms_string = ""
#    for m in ms:
#        ms_string += str(m) + ","
#    if ms_string in msfd:
#        return msfd[ms_string]
#    return ("", 0)
