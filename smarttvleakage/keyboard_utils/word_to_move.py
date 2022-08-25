import argparse
import csv
import numpy as np
import json
import random
import time
from typing import List
from collections import deque
from typing import Dict

from smarttvleakage.audio import Move, SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.utils.constants import KeyboardType, Direction
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, CHANGE_KEYS, SELECT_KEYS, SingleKeyboardGraph
from smarttvleakage.graphs.keyboard_graph import SAMSUNG_STANDARD, APPLETV_SEARCH_ALPHABET, APPLETV_SEARCH_NUMBERS, APPLETV_SEARCH_SPECIAL, APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL
from smarttvleakage.dictionary.dictionaries import REVERSE_CHARACTER_TRANSLATION
from smarttvleakage.utils.constants import Direction
from datetime import datetime, timedelta


def findPath(word: str, use_shortcuts: bool, use_wraparound: bool, use_done: bool, use_direction: bool, mistake_rate: float, decay_rate: float, max_errors: int, keyboard: MultiKeyboardGraph):
    """
    Get the path taken to input the word.

    :param word: the word that the program finds the path for
    :param use_shortcuts: whether or not the program will use shortcuts
    :param use_wraparound: whether or not the program will use shortcuts
    :param use_direction: whether or not the program will consider direction
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
        distance = -1
        direction = []
        if use_direction:
        	direction = bfs_path(prev, character, keyboard.get_adjacency_list(mode, use_shortcuts, use_wraparound))
        	# print('direction: ', direction)
        	if direction == Direction.ANY:
        		if prev == character:
        			distance = 0
        		else:
        			distance = -1
        	else:
        		distance = len(direction)
        else:
        	distance = keyboard.get_moves_from_key(prev, character, use_shortcuts, use_wraparound, mode)
        	direction = Direction.ANY

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
                	if use_direction:
                		temp = bfs_path(prev, changer, keyboard.get_adjacency_list(mode, use_shortcuts, use_wraparound))
                		path.append((Move(num_moves=len(temp), end_sound=CHANGE_KEYS[mode], directions = [i for i in temp])))
                	else:
                		path.append((Move(num_moves=int(keyboard.get_moves_from_key(prev, changer, use_shortcuts, use_wraparound, mode)), end_sound=CHANGE_KEYS[mode], directions=Direction.ANY)))
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

            if use_direction:
            	direction = bfs_path(prev, character, keyboard.get_adjacency_list(mode, use_shortcuts, use_wraparound))
            	if direction == Direction.ANY:
            		if prev == character:
            			distance = 0
            		else:
            			distance = -1
            	else:
            		distance = len(direction)
            		# direction = Direction.ANY
            else:
            	distance = keyboard.get_moves_from_key(prev, character, use_shortcuts, use_wraparound, mode)
            	direction = Direction.ANY

        if character != "<DONE>":
        	# print(distance)
        	# print(direction)
        	assert distance != -1 and distance != None, 'No path from {} to {}'.format(prev, character)

        #assign the correct sound to space
        if character == '<SPACE>' or character == '<DONE>':
        	if distance != -1 and distance != None:
	            if mode in CHANGE_KEYS.keys():
	                path.append((Move(num_moves=distance, end_sound=CHANGE_KEYS[mode], directions=direction)))
	            else:
	                path.append((Move(num_moves=distance, end_sound=SELECT_KEYS[mode], directions=direction)))
        else:
            path.append((Move(num_moves=distance, end_sound=SELECT_KEYS[mode], directions=direction)))

        #determine how many errors there will be by generating a random number and comparing it against the probability of each number of errors
        rand = random.random()
        num_errors = 0
        for x, j in enumerate(mistakes):
            if rand < j:
                num_errors+=2
            else:
                break

        if path != -1 and path != None:
            path[-1] = Move(num_moves=path[-1][0] + num_errors, end_sound=path[-1][1], directions=path[-1][2])

        prev = character
    
    return path


def bfs_path(start_key, target_key, adj_list: Dict[str, Dict[str, str]],):
	visited = [start_key]
	frontier = deque()
	frontier.append(([], start_key))
	final_path = []

	while len(frontier)>0:

		path, key = frontier.popleft()
		# print(key)
		if key == target_key:
			# print(frontier)
			# print(path)
			final_path = path
			break

		neighbors = adj_list[key]
		# print(adj_list)
		# print(neighbors)
		directions = neighbors.keys()
		# print(frontier)

		if 'right' in directions:
			if neighbors['right'] not in visited:
				temp_path = [i for i in path]
				temp_path.append('right')
				# print(temp_path)
				# print(neighbors['right'])
				frontier.append((temp_path, neighbors['right']))
				visited.append(neighbors['right'])
		# print('temp: ', temp_path)
		temp_path = []
		# print(path)
		if 'left' in directions:
			if neighbors['left'] not in visited:
				temp_path = [i for i in path]
				temp_path.append('left')
				frontier.append((temp_path, neighbors['left']))
				visited.append(neighbors['left'])
		temp_path = []
		if 'up' in directions:
			if neighbors['up'] not in visited:
				temp_path = [i for i in path]
				temp_path.append('up')
				frontier.append((temp_path, neighbors['up']))
				visited.append(neighbors['up'])
		temp_path = []
		if 'down' in directions:
			if neighbors['down'] not in visited:
				temp_path = [i for i in path]
				temp_path.append('down')
				frontier.append((temp_path, neighbors['down']))
				visited.append(neighbors['down'])
	returner_path = []
	if final_path == []:
		return Direction.ANY
	else:
		same_in_row = 0
		switch = False
		prev = ''
		for idx, direction in enumerate(final_path):
			if direction != prev or switch:
				if switch:
					returner_path.append(Direction.VERTICAL)
				else:
					same_in_row = 0
					returner_path.append(Direction.ANY)
			else:
				if same_in_row >= 3:
					switch = True
					for x, i in enumerate(returner_path):
						returner_path[x] = Direction.HORIZONTAL
				else:
					returner_path.append(Direction.ANY)
					same_in_row += 1
			prev = direction

		# print(final_path)
		# print('\n')
		# print(returner_path)
		# print('\n')
		return returner_path


if __name__ == '__main__':
	adj = {}
	with open('/home/abebdm/Desktop/Thing/smart-tv-keyboard-leakage/smarttvleakage/graphs/samsung/samsung_keyboard.json', 'r') as f:
		adj = json.load(f)
	bfs_path('q', 'j', adj['adjacency_list'])
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i', type=str, help='enter the input txt file')
    # parser.add_argument('-o', type=str, help='enter the output jsonl.gz file')
    # parser.add_argument('-mr', type=float, help='mistake rate')
    # parser.add_argument('-dr', type=float, help='decay rate')
    # parser.add_argument('-me', type=int, help='max errors in one word')

    # args = parser.parse_args()
    # # words = open(args.i, 'r')
    # words = [args.i]
    # output = []
    # # path = findPath('wph',0,False,True)
    # # print(path)
    # now = datetime.now()
    # for i in words:
    #     # path = findPath(i.strip(), args.e, True)
    #     # output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
    #     path = findPath(i.strip(), False, False, args.mr, args.dr, args.me)
    #     output.append({"word": i.strip(), "move_seq": [{"num_moves": j[0], "sound": j[1].name} for j in path]})
    #     if (now + timedelta(seconds=33)) < datetime.now():
    #         break

    # print(output)
    # save_jsonl_gz(output, args.o)
