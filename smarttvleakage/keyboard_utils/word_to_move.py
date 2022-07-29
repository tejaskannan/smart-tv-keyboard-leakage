import argparse
import csv
import numpy as np
import json
import random
from smarttvleakage.audio.move_extractor import Move, Sound
from smarttvleakage.utils.file_utils import save_jsonl_gz

def findPath(word, error, wraparound):
	active = []
	path = []
	if not wraparound:
		f = open('../graphs/samsung/samsung_keyboard.csv')
		active = list(csv.reader(f))
		f.close()
		f = open('../graphs/samsung/samsung_keyboard_special_1.csv')
		inactive = list(csv.reader(f))
		f.close()
	else:
		f = open('../graphs/samsung/samsung_keyboard_wraparound.csv')
		active = list(csv.reader(f))
		f.close()
		f = open('../graphs/samsung/samsung_keyboard_special_1_wraparound.csv')

		inactive = list(csv.reader(f))
		f.close()
	prev = 'q'
	for i in list(word.lower()):
		if i in active[0]:
			prev_index = active[0].index(prev)
			cur_index = active[0].index(i)
			path.append(Move(num_moves=float(active[prev_index+1][cur_index]),end_sound=Sound.KEY_SELECT))
		else:
			prev_index = active[0].index(prev)
			cur_index = active[0].index('<CHANGE>')
			path.append(Move(num_moves=float(active[prev_index+1][cur_index]),end_sound=Sound.SELECT))
			cur_index = inactive[0].index(i)
			path.append(Move(num_moves=float(inactive[inactive[0].index('<CHANGE>')+1][cur_index]),end_sound=Sound.KEY_SELECT))
			temp = inactive
			inactive = active
			active = temp
		prev = i
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
		path = findPath(i.strip(), args.e, False)
		output.append({"word":i.strip(), "move_seq":[{"num_moves":j[0], "sound":j[1].name} for j in path]})
	save_jsonl_gz(output, args.o)
