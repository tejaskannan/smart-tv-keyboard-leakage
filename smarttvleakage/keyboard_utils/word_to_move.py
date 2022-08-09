import argparse
import csv
import numpy as np
import json
import random
from smarttvleakage.audio.move_extractor import Move, Sound
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from datetime import datetime, timedelta


def findPath(word, shortcuts, wraparound, mr, dr, me):
	mistakes = []
	for n in range(me):
		mistakes.append(mr*(dr**n))
	path = []
	mode = KeyboardMode.STANDARD
	prev = START_KEYS[mode]
	keyboard = MultiKeyboardGraph()
	page_1 = True
	for i in list(word.lower()):
		distance = keyboard.get_moves_from_key(prev, i, shortcuts, wraparound, mode)
		print(distance)
		while distance == -1:
			path.append((Move(num_moves=float(keyboard.get_moves_from_key(prev, "<CHANGE>", shortcuts, wraparound, mode)), end_sound=Sound.SELECT)))
			prev = '<CHANGE>'
			mode = get_keyboard_mode(prev, mode)
			prev = START_KEYS[mode]
			distance = keyboard.get_moves_from_key(prev, i, shortcuts, wraparound, mode)
		if i == ' ':
			path.append((Move(num_moves=distance, end_sound=Sound.SELECT)))
		else:
			path.append((Move(num_moves=distance, end_sound=Sound.KEY_SELECT)))
		rand = random.random()
		for x, j in enumerate(mistakes):
			if rand < j:
				path[-1] = Move(num_moves=path[-1][0]+2*(x+1), end_sound=path[-1][1])
		prev = i
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
		if now+timedelta(seconds=33) < datetime.now():
			break
	print(output)
	save_jsonl_gz(output, args.o)
