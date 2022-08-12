import argparse
import csv
import numpy as np
import json
import random
import time
from typing import List

from smarttvleakage.audio.move_extractor import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, SAMSUNG_STANDARD
from smarttvleakage.dictionary.dictionaries import REVERSE_CHARACTER_TRANSLATION
from datetime import datetime, timedelta


def findPath(word, shortcuts, wraparound, mr, dr, me, keyboard):
    mistakes = []
    for n in range(me):
        mistakes.append(mr*(dr**n))

    path = []
    mode = SAMSUNG_STANDARD
    prev = START_KEYS[mode]

    for character in list(word.lower()):
        character = REVERSE_CHARACTER_TRANSLATION.get(character, character)
        distance = keyboard.get_moves_from_key(prev, character, shortcuts, wraparound, mode)

        if distance == -1:
            path.append((Move(num_moves=int(keyboard.get_moves_from_key(prev, '<CHANGE>', shortcuts, wraparound, mode)), end_sound=SAMSUNG_SELECT)))
            prev = '<CHANGE>'
            mode = get_keyboard_mode(prev, mode, keyboard_type=KeyboardType.SAMSUNG)
            distance = keyboard.get_moves_from_key(prev, character, shortcuts, wraparound, mode)

        assert distance != -1, 'No path from {} to {}'.format(prev, character)

        if character == '<SPACE>':
            path.append((Move(num_moves=distance, end_sound=SAMSUNG_SELECT)))
        else:
            path.append((Move(num_moves=distance, end_sound=SAMSUNG_KEY_SELECT)))

        rand = random.random()
        for x, j in enumerate(mistakes):
            if rand < j:
                path[-1] = Move(num_moves=path[-1][0] + 2 * (x + 1), end_sound=path[-1][1])

        prev = character

    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='enter the input txt file')
    parser.add_argument('-o', type=str, help='enter the output jsonl.gz file')
    parser.add_argument('-mr', type=float, help='mistake rate')
    parser.add_argument('-dr', type=float, help='decay rate')
    parser.add_argument('-me', type=int, help='max errors in one word')

    args = parser.parse_args()
    # words = open(args.i, 'r')
    words = [args.i]
    output = []
    # path = findPath('wph',0,False,True)
    # print(path)
    now = datetime.now()
    for i in words:
        # path = findPath(i.strip(), args.e, True)
        # output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
        path = findPath(i.strip(), False, False, args.mr, args.dr, args.me)
        output.append({"word": i.strip(), "move_seq": [{"num_moves": j[0], "sound": j[1].name} for j in path]})
        if (now + timedelta(seconds=33)) < datetime.now():
            break

    print(output)
    save_jsonl_gz(output, args.o)
