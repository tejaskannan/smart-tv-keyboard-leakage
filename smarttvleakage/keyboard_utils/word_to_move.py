import argparse
import csv
import numpy as np
import json
import random
import time
from typing import List

from smarttvleakage.audio import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, CHANGE_KEYS, SELECT_KEYS, SingleKeyboardGraph
from smarttvleakage.graphs.keyboard_graph import SAMSUNG_STANDARD, APPLETV_SEARCH_ALPHABET, APPLETV_SEARCH_NUMBERS, APPLETV_SEARCH_SPECIAL, APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL
from smarttvleakage.dictionary.dictionaries import REVERSE_CHARACTER_TRANSLATION
from datetime import datetime, timedelta


def findPath(word: str, use_shortcuts: bool, use_wraparound: bool, use_done: bool, mistake_rate: float, decay_rate: float, max_errors: int, keyboard: MultiKeyboardGraph):
    """
    Get the path taken to input the word.

    :param word: the word that the program finds the path for
    :param use_shortcuts: whether or not the program will use shortcuts
    :param use_wraparound: whether or not the program will use shortcuts
    :param mistake_rate: the base probability of an error
    :param decay_rate: the decay rate of the mistake rate
    :param max_errors: the maximum number of errors for one letter to the next
    :param keyboard: the keyboard that will be used
    :return: a list of Move named tuple
    """
    mistakes = []

    #set up the probability of each number of errors
    for n in range(max_errors):
        mistakes.append(mistake_rate*(decay_rate**n))
    path = []
    mode = keyboard.get_start_keyboard_mode()
    prev = START_KEYS[mode]
    word = list(word.lower())
    if use_done:
        word.append("<DONE>")
    #print(word)

    for character in word:
        character = REVERSE_CHARACTER_TRANSLATION.get(character, character)
        distance = keyboard.get_moves_from_key(prev, character, use_shortcuts, use_wraparound, mode)

        # handle page switching
        if distance == -1:
            in_keyboard = ''
            found_char = False
            is_backwards = False

            # find which page the target key is on
            for possible_keyboard in keyboard.get_keyboards().keys():
                if found_char == True:
                    break
                if character in keyboard.get_keyboards()[possible_keyboard].get_characters():
                    found_char = True
                    in_keyboard = possible_keyboard

            original_mode = mode
            counter = 0
            on_key = prev

            # navigate to the correct page by going to the nearest character that switches to the next page
            # and repeating this until the page is the correct one
            while mode != in_keyboard:
                # print(mode)
                # print(in_keyboard)
                changer = keyboard.get_nearest_link(prev, mode, in_keyboard, use_shortcuts, use_wraparound)
                # print(changer)
                if changer != prev:
                    on_key = changer
                    original_mode = mode
                    counter = 0

                if mode in CHANGE_KEYS.keys():
                    path.append((Move(num_moves=int(keyboard.get_moves_from_key(prev, changer, use_shortcuts, use_wraparound, mode)), end_sound=CHANGE_KEYS[mode])))
                prev = changer
                linked_state = keyboard.get_linked_states(on_key, original_mode)
                # print(on_key)
                # print(original_mode)
                if len(linked_state)<counter:
                    break
                # print(linked_state)
                # print(counter)
                mode = linked_state[counter][1]
                prev = linked_state[counter][0]
                counter+=1

            distance = keyboard.get_moves_from_key(prev, character, use_shortcuts, use_wraparound, mode)

        if character != "<DONE>":
        	assert distance != -1 and distance != None, 'No path from {} to {}'.format(prev, character)

        #assign the correct sound to space
        if character == '<SPACE>' or character == '<DONE>':
        	if distance != -1 and distance != None:
	            if mode in CHANGE_KEYS.keys():
	                path.append((Move(num_moves=distance, end_sound=CHANGE_KEYS[mode])))
	            else:
	                path.append((Move(num_moves=distance, end_sound=SELECT_KEYS[mode])))
        else:
            path.append((Move(num_moves=distance, end_sound=SELECT_KEYS[mode])))

        #determine how many errors there will be by generating a random number and comparing it against the probability of each number of errors
        rand = random.random()
        num_errors = 0
        for x, j in enumerate(mistakes):
            if rand < j:
                num_errors+=2
            else:
                break

        if path != -1 and path != None:
            path[-1] = Move(num_moves=path[-1][0] + num_errors, end_sound=path[-1][1])

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
