import re
import argparse
from smarttvleakage.keyboard_utils.unpack_jsonl_gz import read_moves
from smarttvleakage.audio import Move, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT
import itertools
import random

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
				counter = int(counter)
				#print('counter: ',counter)
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
	#random.shuffle(output)
	output.insert(0, '[')
	output.append(']')
	return [''.join(output), pos_temp]


def find_regex(moves1, spaces, average):
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
				['*', '/', '\\?', '@'],
				['!'],
				['\\-']]

	standard_space = [[],
					['c'],
					['d', 'm', 'v', 'x', 'z'],
					[',', '/', 'a', 'b', 'e', 'f', 'j', 'n', 's'],
					['.', '3', 'g', 'h', 'k', 'q', 'r', 'u', 'w'],
					['1', '2', '4', '7', '\\?', 'i', 'l', 't', 'y'],
					['5', '6', '8', 'o', '~'],
					['\\-', '9', '@', 'p'],
					['!', '0', '^'],
					['*']]

	standard_after_change = [[], ['q'], ['1', 'a', 'w'], ['2', 'e', 's', 'z'], ['3', 'd', 'r', 'x'], ['4', 'c', 'f', 't'], ['5', 'g', 'v', 'y'], ['6', 'b', 'h', 'm', 'u'], [',', '/', '7', 'i', 'j', 'n'], ['.', '8', 'k', 'o'], ['9', '?', 'l', 'p'], ['0', '^', '~'], ['*', '-', '@'], ['!']]


	special = [[],
		    ['!'],
		    ["'", '1', '@'],
		    ['\\"', '#', '\\-', '2'],
		    ['$', '+', '3', ':'],
		    ['/', '4', ';'],
		    [',', '5', '^'],
		    ['&', '6', '=', '\\?'],
		    ['%', '*', '7', '<'],
		    ['(', '8', '>', '\\\\'],
		    [')', '9', '{'],
		    ['0', '\\[', '}'],
		    ['\\]']]

	special_space = [[],
					[],
					['+', '\\-', ':', '\\\\'],
					['\\"', '#', '%', "'", ';', '<', '='],
					['!', '$', '&', ',', '3', '>', '\\?', '@'],
					['*', '/', '1', '2', '4', '7', '^', '{'],
					['(', '5', '6', '8', '}'],
					[')', '9'],
					['0', '\\['],
					['\\]']]


	# escape = '.+*?^$()[]{}|\\'
	moves = [moves1]
	# page_1 = '?l?d^*~@!\\,./-'
	# page_2 = '?d!@#$/^&*()[]\'";:\\,??<>{}-+=%\\'
	pos = [[0] for i in moves]
	words = []
	regex = [[] for i in moves]
	for idx, move_list in enumerate(moves):
		page2 = False
		thing = []
		space_used = False
		changed = False
		for move_idx, move in enumerate(move_list):
			thing = []
			if move[1] == SAMSUNG_SELECT:
				if move_idx in spaces:
					# print('\n')
					# print(move)
					# print('\n')
					if page2:
						thing = ['[ ]', [0]]
					else:
						thing = ['[ ]', [0]]
					space_used = True
					#move_list[move_idx+1] = Move(num_moves = move_list[move_idx+1][0]-1, end_sound = move_list[move_idx+1][1], directions = move_list[move_idx+1][2])
				else:
					if page2:
						# print('1')
						# print(move)
						# print(spaces)
						page2 = False
						pos[idx] = [0]
						space_used = False
						changed = True
					else:
						# print('1')
						# print(move)
						# print(spaces)
						page2 = True
						pos[idx] = [0]
						space_used = False
						changed = True
			else:
				thing = []
				if page2:
					if space_used:
						# print('1')
						thing = get_possible(move, special_space, pos[idx])
					else:
						# print('2')
						thing = get_possible(move, special, pos[idx])
						# print(thing)
				else:
					if space_used:
						# print('3')
						thing = get_possible(move, standard_space, pos[idx])
					elif changed:
						# print('4')
						thing = get_possible(move, standard_after_change, pos[idx])
					else:
						# print('5')
						thing = get_possible(move, standard, pos[idx])

			if thing != []:
				regex[idx].append(thing[0])
				pos[idx] = thing[1]
			# print(regex[idx])

	# totals = []
	# averages = []
	# original = []

	# for word in regex:
	# 	totals.append([])
	# 	original.append([])
	# 	for character in word:
	# 		totals[-1].append(0)
	# 		prev = ''
	# 		for letter in list(character):
	# 			if letter != '\\':
	# 				#print(totals[-1][-1])
	# 				totals[-1][-1]+=1
	# 			elif prev == '\\':
	# 				totals[-1][-1]+=1
	# 			prev = letter
	# 		original[-1].append(71)
	# #print(totals)
	# for idx, word in enumerate(totals):
	# 	thing = 1
	# 	og_thing = 1
	# 	for character in word:
	# 		thing *= character
	# 		og_thing *= 71
	# 	totals[idx] = thing
	# 	original[idx] = og_thing
	# # print('\n')
	# # print(totals)
	# # print(original)
	# output = [totals, original]
	# for idx, num in enumerate(totals):
	# 	# print(original[idx]/num)
	# 	output.append(original[idx]/num)
	# if average:
	# 	return output


	# for idx_exp, expression in enumerate(regex):
	# 	totals.append([])
	# 	for idx_letter, letter in enumerate(expression):
	# 		totals[idx].append(0)
	# 		prev = ''
	# 		for character in letter:
	# 			if character != '\\':
	# 				totals[idx_exp][idx_letter]+=1
	# 			elif prev == '\\':
	# 				totals[idx_exp][idx_letter]+=1
	# 			prev = character


	# words_combined = ' '.join(words)
	# incorrect = 0
	# regex_temp = [['' for j in i] for i in regex]
	# pos_temp = [[1] for i in moves]
	# num_select = 0
	# for pos_idx, move_seq in enumerate(moves):
	# 	for idx, move in enumerate(move_seq):
	# 		# print('idx: ', idx)
	# 		if move[1] == SAMSUNG_SELECT:
	# 			for i in range(idx, 0, -1):
	# 				if move_seq[i][1] == SAMSUNG_SELECT:
	# 					continue
	# 				thing = get_possible(move_seq[i], standard, pos_temp[pos_idx])
	# 				regex_temp[pos_idx][i-1-num_select] = thing[0]
	# 				pos_temp[pos_idx] = thing[1]
	# 			num_select+=1
	# 	for idx,expression in enumerate(regex[pos_idx]):
	# 		if regex_temp[pos_idx][idx] != '':
	# 			if len(regex_temp[pos_idx][idx])<len(expression):
	# 				regex[pos_idx][idx] = regex_temp[pos_idx][idx]
	# print(regex)
	return regex


def get_selects(moves):
	with_select = []
	output = []
	# print(moves)
	for idx, move in enumerate(moves):
		# print(move)
		# print('\n')
		if move[1] == SAMSUNG_SELECT:
			with_select.append(idx)
			#print('1')
	for L in range(0, len(with_select)+1):
	    for subset in itertools.combinations(with_select, L):
	        # print(subset)
	        output.append(subset)
	return output


def get_regex(moves):
	regexes = []
	for move_sequence in moves:
		#print(move_sequence)
		regexes.append([])
		combinations = get_selects(move_sequence)
		for combination in combinations:
			regexes[-1].append(find_regex(move_sequence, combination, False))
	return regexes


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type=str, required=True, help='input jsonl.gz file with move sequence')
	args = parser.parse_args()
	moves = read_moves(args.i)
	regex = find_regex(moves)
	with open('masks.txt', 'w') as f:
		for exp in regex:
			f.write(''.join(exp))

