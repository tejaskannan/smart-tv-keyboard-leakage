import argparse
import csv
import numpy as np
import json
import random
from smarttvleakage.audio.move_extractor import Move, Sound
from smarttvleakage.utils.file_utils import save_jsonl_gz
from smarttvleakage.utils.transformations import get_keyboard_mode
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS

def findPath(word, error, shortcuts, wraparound):
	# active = []
	path = []
	mode = KeyboardMode.STANDARD
	# if not wraparound:
	# 	f = open('../graphs/samsung/samsung_keyboard.csv')
	# 	active = list(csv.reader(f))
	# 	f.close()
	# 	f = open('../graphs/samsung/samsung_keyboard_special_1.csv')
	# 	inactive = list(csv.reader(f))
	# 	f.close()
	# else:
	# 	f = open('../graphs/samsung/samsung_keyboard_wraparound.csv')
	# 	active = list(csv.reader(f))
	# 	f.close()
	# 	f = open('../graphs/samsung/samsung_keyboard_special_1_wraparound.csv')

	# 	inactive = list(csv.reader(f))
	# 	f.close()
	# prev = 'q'
	prev = START_KEYS[mode]
	keyboard = MultiKeyboardGraph()
	for i in list(word.lower()):
		distance = keyboard.get_moves_from_key(prev, i, shortcuts, wraparound, mode)
		while distance == -1:
			#print(i)
			path.append((Move(num_moves=float(keyboard.get_moves_from_key(prev, "<CHANGE>", shortcuts, wraparound, mode)),end_sound=Sound.SELECT)))
			prev = '<CHANGE>'
			mode = get_keyboard_mode(prev, mode)
			distance = keyboard.get_moves_from_key(prev, i, shortcuts, wraparound, mode)
			#print(distance)
		path.append((Move(num_moves=distance, end_sound=Sound.KEY_SELECT)))
		if random.random() > (1-error)**float(path[-1][0]):
			if random.random()>0.3*error:
				path[-1]+=2
			else:
				path[-1]+=4
	return path

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type=str, help='enter the input txt file')
	parser.add_argument('-o', type=str, help='enter the output jsonl.gz file')
	parser.add_argument('-e', type=float, help='percent of moves with an error')

	args = parser.parse_args()
	words = open(args.i, 'r')
	output = []
	for i in words:
		# path = findPath(i.strip(), args.e, True)
		# output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
		path = findPath(i.strip(), args.e, False, False)
		output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
	save_jsonl_gz(output, args.o)
