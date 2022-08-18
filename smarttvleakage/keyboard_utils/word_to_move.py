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


def findPath(word, shortcuts, wraparound, mr, dr, me, kb):
    mistakes = []
    for n in range(me):
        mistakes.append(mr*(dr**n))
    keyboard = kb
    path = []
    mode = kb.get_start_keyboard_mode()
    prev = START_KEYS[mode]

    for character in list(word.lower()):
        character = REVERSE_CHARACTER_TRANSLATION.get(character, character)
        distance = keyboard.get_moves_from_key(prev, character, shortcuts, wraparound, mode)
        #print(distance)
        #print('link: ', keyboard.get_linked_states(prev, mode))
        if distance == -1:
        	in_keyboard = ''
        	found_char = False
        	is_backwards = False
        	for possible_keyboard in kb.get_keyboards():
        		if found_char == True:
        			break
        		if character in possible_keyboard.get_characters():
        			found_char = True
        			in_keyboard = possible_keyboard
        	#print(keyboard.get_keyboards())
        	#original_prev = prev
        	original_mode = mode
        	counter = 0
        	on_key = prev
        	while keyboard._keyboards[mode] != in_keyboard:
        		changer = keyboard.get_nearest_link(prev, mode, shortcuts, wraparound)
        		if changer != prev:
        			on_key = changer
        			original_mode = mode
        			counter = 0
        		#print(counter)
        		# print(CHANGE_KEYS.keys())
        		if mode in CHANGE_KEYS.keys():
        			# print('\n')
        			# print(CHANGE_KEYS[mode])
        			path.append((Move(num_moves=int(keyboard.get_moves_from_key(prev, changer, shortcuts, wraparound, mode)), end_sound=CHANGE_KEYS[mode], move_times=[])))
        		prev = changer
        		#print(on_key)
        		linked_state = keyboard.get_linked_states(on_key, original_mode)
        		#print(linked_state)
        		mode = linked_state[counter][1]
        		prev = linked_state[counter][0]
        		counter+=1
        	distance = keyboard.get_moves_from_key(prev, character, shortcuts, wraparound, mode)

        assert distance != -1, 'No path from {} to {}'.format(prev, character)

        if character == '<SPACE>':
        	if mode in CHANGE_KEYS.keys():
        		path.append((Move(num_moves=distance, end_sound=CHANGE_KEYS[mode], move_times=[])))
        	else:
        		path.append((Move(num_moves=distance, end_sound=SELECT_KEYS[mode], move_times=[])))
        else:
        	path.append((Move(num_moves=distance, end_sound=SELECT_KEYS[mode], move_times=[])))

        rand = random.random()
        for x, j in enumerate(mistakes):
            if rand < j:
                path[-1] = Move(num_moves=path[-1][0] + 2 * (x + 1), end_sound=path[-1][1], move_times=[])

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
