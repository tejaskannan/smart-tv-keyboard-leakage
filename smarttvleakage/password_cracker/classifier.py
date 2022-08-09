import re
import argparse
from smarttvleakage.keyboard_utils.unpack_jsonl_gz import read_moves
from smarttvleakage.audio.move_extractor import Move, Sound


def get_possible(move, thing, pos):
	output = []
	pos_temp = []
	for i in pos:
		if move[0] == 0:
			for j in thing[i]:
				if j not in output:
					output.append(j)
			pos_temp.append(i)
		else:
			counter = move[0] + i
			if counter >= len(thing):
				if counter % 2 == len(thing) % 2:
					counter=len(thing)-2
				else:
					counter=len(thing)-1
			while counter>=i+1:
				for j in thing[counter]:
					if j not in output:
						output.append(j)
				pos_temp.append(counter)
				counter-=2
			counter = i-move[0]
			if counter<0:
				if counter%2 == 0:
					counter = 0
				else:
					counter = 1
			while counter<=i-1:
				for j in thing[counter]:
					if j not in output:
						output.append(j)
				pos_temp.append(counter)
				counter+=2
	output.insert(0, '[')
	output.append(']')
	return [''.join(output), pos_temp]


def find_regex(moves1):
	standard = [['q'],
				['1', 'a', 'w'],
				['2', 'e', 's', 'z'],
				['3', 'd', 'r', 'x'],
				['4', 'c', 'f', 't'],
				['5', 'g', 'v', 'y'],
				['6', 'b', 'h', 'u'],
				['7', 'i', 'j', 'n'],
				['8', 'k', 'm', 'o'],
				[',', '9', 'l', 'p'],
				['.', '0', '^', '~'],
				['*', '/', '?', '@'],
				['!'],
				['\\-']]
	special = [[],
				['!'],
				["'", '1', '@'],
				['"', '#', '\\-', '2'],
				['$', '+', '3', ':'],
				['/', '4', ';'],
				[',', '5', '^'],
				['&', '6', '=', '?'],
				['%', '*', '7', '<'],
				['(', '8', '>', '\\\\'],
				[')', '9', '{'],
				['0', '\\[', '}'],
				['\\]']]

	escape = '.+*?^$()[]{}|\\'
	moves = [moves1]
	page_1 = '?l?d^*~@!\\,./-'
	page_2 = '?d!@#$/^&*()[]\'";:\\,??<>{}-+=%\\'
	pos = [[0] for i in moves]
	words = []
	regex = [[] for i in moves]
	pos = [[0]for i in moves]
	for idx, move_list in enumerate(moves):
		page2 = False
		for move in move_list:
			if move[1] == Sound.SELECT:
				if page2:
					page2 = False
					pos[idx] = [0]
				else:
					page2 = True
					pos[idx] = [0]
			else:
				thing = []
				if page2:
					thing = get_possible(move, special, pos[idx])
				else:
					thing = get_possible(move, standard, pos[idx])
				regex[idx].append(thing[0])
				pos[idx] = thing[1]

	totals = []
	averages = []
	for idx_exp, expression in enumerate(regex):
		totals.append([])
		for idx_letter, letter in enumerate(expression):
			totals[idx].append(0)
			prev = ''
			for character in letter:
				if character != '\\':
					totals[idx_exp][idx_letter]+=1
				elif prev == '\\':
					totals[idx_exp][idx_letter]+=1
				prev = k
	for word in totals:
		averages.append(1)
		for letter in word:
			averages[-1]*=letter
	total = 0
	count = 0
	for word in averages:
		total+=word
		count+=1

	words_combined = ' '.join(words)
	incorrect = 0
	return regex


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type=str, required=True, help='input jsonl.gz file with move sequence')
	args = parser.parse_args()
	moves = read_moves(args.i)
	regex = find_regex(moves)
	with open('masks.txt', 'w') as f:
		for exp in regex:
			f.write(''.join(exp))
