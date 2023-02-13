import argparse
import csv
import numpy as np
import json
import random
import time
from typing import List
from collections import deque
from typing import Dict

from smarttvleakage.audio import Move
from smarttvleakage.audio.sounds import SAMSUNG_SELECT, SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT, DONE_SOUNDS, SPACE_SOUNDS, CHANGE_SOUNDS, KEY_SELECT_SOUNDS
from smarttvleakage.dictionary.dictionaries import DONE, SPACE, CHANGE, NEXT
from smarttvleakage.utils.constants import KeyboardType, Direction
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, CHANGE_KEYS, SELECT_KEYS, SingleKeyboardGraph
from smarttvleakage.graphs.keyboard_graph import SAMSUNG_STANDARD, APPLETV_SEARCH_ALPHABET, APPLETV_SEARCH_NUMBERS, APPLETV_SEARCH_SPECIAL, APPLETV_PASSWORD_STANDARD, APPLETV_PASSWORD_SPECIAL
from smarttvleakage.graphs.keyboard_graph import SAMSUNG_CAPS, SAMSUNG_SPECIAL_ONE, SAMSUNG_SPECIAL_TWO
from smarttvleakage.dictionary.dictionaries import REVERSE_CHARACTER_TRANSLATION
from datetime import datetime, timedelta


def findPath(word: str, use_shortcuts: bool, use_wraparound: bool, use_done: bool, mistake_rate: float, decay_rate: float, max_errors: int, keyboard: MultiKeyboardGraph, start_key: str):
    """
    Get the path taken to input the word.

    :param word: the word that the program finds the path for
    :param use_shortcuts: whether or not the program will use shortcuts
    :param use_wraparound: whether or not the program will use shortcuts
    :param mistake_rate: the base probability of an error
    :param decay_rate: the decay rate of the mistake rate
    :param max_errors: the maximum number of errors for one letter to the next
    :param keyboard: the keyboard that will be used
    :param start_key: the key to start the move sequence on
    :return: a list of Move named tuple
    """
    mistakes = []

    # set up the probability of each number of errors
    for n in range(max_errors):
        mistakes.append(mistake_rate * (decay_rate**n))

    path = []
    mode = keyboard.get_start_keyboard_mode()
    keyboard_type = keyboard.keyboard_type
    prev = start_key
    word = list(word.lower())

    if use_done:
        word.append(DONE)

    for character in word:
        character = REVERSE_CHARACTER_TRANSLATION.get(character, character)
        distance = keyboard.get_moves_from_key(prev, character, use_shortcuts, use_wraparound, mode)

        # handle page switching
        if distance == -1:
            in_keyboard = ''
            found_char = False
            is_backwards = False

            # Find which page the target key is on
            target_keyboard_mode = keyboard.get_keyboard_with_character(character)
            assert target_keyboard_mode is not None, 'The character {} was not found in any keyboard'.format(character)

            original_mode = mode
            counter = 0
            on_key = prev

            # Hard code in the change paths for each keyboard, as each keyboard type has different dynamics.
            if keyboard_type == KeyboardType.SAMSUNG:
                if mode in (SAMSUNG_STANDARD, SAMSUNG_CAPS):
                    num_moves = keyboard.get_moves_from_key(prev, CHANGE, use_shortcuts, use_wraparound, mode)
                    path.append(Move(num_moves=num_moves, end_sound=CHANGE_SOUNDS[keyboard_type], directions=Direction.ANY))
                    on_key = CHANGE

                    if target_keyboard_mode == SAMSUNG_SPECIAL_TWO:
                        num_moves = keyboard.get_moves_from_key(CHANGE, NEXT, use_shortcuts, use_wraparound, SAMSUNG_SPECIAL_ONE)
                        path.append(Move(num_moves=num_moves, end_sound=CHANGE_SOUNDS[keyboard_type], directions=Direction.ANY))
                        on_key = NEXT
                elif mode in (SAMSUNG_SPECIAL_ONE, SAMSUNG_SPECIAL_TWO):
                    if target_keyboard_mode in (SAMSUNG_STANDARD, SAMSUNG_CAPS):
                        num_moves = keyboard.get_moves_from_key(prev, CHANGE, use_shortcuts, use_wraparound, mode)
                        path.append(Move(num_moves=num_moves, end_sound=CHANGE_SOUNDS[keyboard_type], directions=Direction.ANY))
                        on_key = CHANGE
                    elif target_keyboard_mode in (SAMSUNG_SPECIAL_ONE, SAMSUNG_SPECIAL_TWO):
                        num_moves = keyboard.get_moves_from_key(prev, NEXT, use_shortcuts, use_wraparound, mode)
                        path.append(Move(num_moves=num_moves, end_sound=CHANGE_SOUNDS[keyboard_type], directions=Direction.ANY))
                        on_key = NEXT
                    else:
                        raise ValueError('Unknown keyboard mode: {}'.format(target_keyboard_mode))
                else:
                    raise ValueError('Unknown keyboard mode: {}'.format(mode))
            elif keyboard_type in (KeyboardType.APPLE_TV_SEARCH, KeyboardType.APPLE_TV_PASSWORD):
                on_key = keyboard.get_linked_key_to(current_key=prev,
                                                    current_mode=mode,
                                                    target_mode=target_keyboard_mode)
                
                assert on_key is not None, 'No linked key for {} from {} to {}'.format(prev, mode, target_keyboard_mode)

            mode = target_keyboard_mode
            distance = keyboard.get_moves_from_key(on_key, character, use_shortcuts, use_wraparound, mode)

        if character != DONE:
            assert (distance != -1) and (distance is not None), 'No path from {} to {}'.format(prev, character)

        # assign the correct sound to space
        if character in (SPACE, DONE):
            if (distance != -1) and (distance is not None):
                if character == SPACE:
                    path.append((Move(num_moves=distance, end_sound=SPACE_SOUNDS[keyboard_type], directions=Direction.ANY)))
                else:
                    path.append((Move(num_moves=distance, end_sound=DONE_SOUNDS[keyboard_type], directions=Direction.ANY)))
        else:
            path.append((Move(num_moves=distance, end_sound=KEY_SELECT_SOUNDS[keyboard_type], directions=Direction.ANY)))

        # determine how many errors there will be by generating a random number and comparing it against the probability of each number of errors
        rand = random.random()
        num_errors = 0
        for m in mistakes:
            if rand < m:
                num_errors += 2
            else:
                break

        if (path != -1) and (path is not None):
            path[-1] = Move(num_moves=path[-1].num_moves + num_errors, end_sound=path[-1].end_sound, directions=Direction.ANY)

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
