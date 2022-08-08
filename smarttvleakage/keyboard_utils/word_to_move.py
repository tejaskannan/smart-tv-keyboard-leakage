import argparse
import csv
import numpy as np
import json
import random
from typing import List

from smarttvleakage.audio.move_extractor import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, SAMSUNG_STANDARD


def findPath(word, error, shortcuts, wraparound) -> List[Move]:
    path: List[Move] = []
    mode = SAMSUNG_STANDARD
    prev = START_KEYS[mode]
    keyboard_type = KeyboardType.SAMSUNG

    keyboard = MultiKeyboardGraph(keyboard_type=keyboard_type)

    for character in list(word.lower()):
        distance = keyboard.get_moves_from_key(start_key=prev,
                                               end_key=character,
                                               use_shortcuts=shortcuts,
                                               use_wraparound=wraparound,
                                               mode=mode)

        while distance == -1:
            print('Start Key: {}, End Key: {}'.format(prev, character))

            path.append(Move(num_moves=keyboard.get_moves_from_key(prev, '<CHANGE>', shortcuts, wraparound, mode),end_sound=SAMSUNG_SELECT))
            prev = '<CHANGE>'
            mode = get_keyboard_mode(prev, mode, keyboard_type=keyboard_type)
            distance = keyboard.get_moves_from_key(prev, character, shortcuts, wraparound, mode)
            return

        path.append(Move(num_moves=distance, end_sound=SAMSUNG_KEY_SELECT))

        if random.random() > (1-error)**float(path[-1][0]):
            if random.random()>0.3*error:
                path[-1]+=2
            else:
                path[-1]+=4

        prev = character

    return path


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', type=str, help='enter the input txt file')
    # parser.add_argument('-o', type=str, help='enter the output jsonl.gz file')
    # parser.add_argument('-e', type=float, help='percent of moves with an error')

    # args = parser.parse_args()
    # words = open(args.i, 'r')
    output = []
    path = findPath('1234 1234',0,False,True)
    print(path)

    # for i in words:
    # 	# path = findPath(i.strip(), args.e, True)
    # 	# output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
    # 	path = findPath(i.strip(), args.e, False, False)
    # 	output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
    # save_jsonl_gz(output, args.o)
