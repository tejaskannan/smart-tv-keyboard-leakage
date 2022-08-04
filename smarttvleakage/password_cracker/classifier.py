import re
import argparse
from smarttvleakage.keyboard_utils.unpack_jsonl_gz import read_moves
from smarttvleakage.audio.move_extractor import Move, Sound

# spc = {"!": 1, "\"": 3, "#": 3, "$": 4, "%": 8, "&": 7, "'": 2, "(": 9, ")": 10, "*": 8, "+": 4, ",": 6, "-": 3, "/": 5, "0": 11, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, ":": 4, ";": 5, "<": 8, "<<": 12, "<BACK>": 12, "<CENT>": 14, "<DIV>": 6, "<DOWN>": 14, "<EURO>": 10, "<LANGUAGE>": 3, "<LEFT>": 13, "<MULT>": 5, "<NEXT>": 1, "<POUND>": 11, "<RIGHT>": 15, "<SETTINGS>": 4, "<SPACE>": 5, "<UP>": 13, "<YEN>": 12, "=": 7, ">": 9, ">>": 13, "?": 7, "@": 2, "[": 11, "\\": 9, "]": 12, "^": 6, "{": 10, "}": 11}
# std = {"!": 12, "*": 11, ",": 9, "-": 13, ".": 10, "/": 11, "0": 10, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "<BACK>": 11, "<CANCEL>": 14, "<CAPS>": 2, "<CHANGE>": 1, "<COM>": 10, "<DELETEALL>": 12, "<DONE>": 13, "<DOWN>": 13, "<LANGUAGE>": 2, "<LEFT>": 12, "<RETURN>": 12, "<RIGHT>": 14, "<SETTINGS>": 3, "<SPACE>": 5, "<UP>": 12, "<WWW>": 9, "?": 11, "@": 11, "^": 10, "a": 1, "b": 6, "c": 4, "d": 3, "e": 2, "f": 4, "g": 5, "h": 6, "i": 7, "j": 7, "k": 8, "l": 9, "m": 8, "n": 7, "o": 8, "p": 9, "r": 3, "s": 2, "t": 4, "u": 6, "v": 5, "w": 1, "x": 3, "y": 5, "z": 2, "~": 10}
# special = [[] for i in range(max(spc.values()))]
# standard = [[] for i in range(max(std.values()))]
# for i in spc:
# 	special[spc[i]-1].append(i)
# for i in std:
# 	standard[std[i]-1].append(i)
# print(special)
# print('\n')
# print(standard)

standard = [['q'], ['1', 'a', 'w'], ['2', 'e', 's', 'z'], ['3', 'd', 'r', 'x'], ['4', 'c', 'f', 't'], ['5', 'g', 'v', 'y'], ['6', 'b', 'h', 'u'], ['7', 'i', 'j', 'n'], ['8', 'k', 'm', 'o'], [',', '9', 'l', 'p'], ['.', '0', '^', '~'], ['*', '/', '?', '@'], ['!'], ['\\-']]
special = [[],['!'], ["'", '1', '@'], ['"', '#', '\\-', '2'], ['$', '+', '3', ':'], ['/', '4', ';'], [',', '5', '^'], ['&', '6', '=', '?'], ['%', '*', '7', '<'], ['(', '8', '>', '\\\\'], [')', '9', '{'], ['0', '\\[', '}'], ['\\]']]

escape = '.+*?^$()[]{}|\\'

page_1 = '?l?d^*~@!\\,./-'
page_2 = '?d!@#$/^&*()[]\'";:\\,??<>{}-+=%\\'

def get_possible(move, thing, pos):
	output = []
	pos_temp = []
	#print(move)
	#print(pos)
	for i in pos:
		#print(i)
		if move[0] == 0:
			for j in thing[i]:
				if j not in output:
					output.append(j)
			pos_temp.append(i)
		else:
			# print(move[0])
			# print(i)
			counter = move[0]+i
			if counter>=len(thing):
				if counter%2 == len(thing)%2:
					counter=len(thing)-2
				else:
					counter=len(thing)-1
			while counter>=i+1:
				# print(counter)
				# print(len(thing))
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
				# print(counter)
				# print(len(thing))
				for j in thing[counter]:
					if j not in output:
						output.append(j)
				pos_temp.append(counter)
				counter+=2
	output.insert(0,'[')
	output.append(']')
	return [''.join(output),pos_temp]

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True)
args = parser.parse_args()

moves = read_moves(args.i)
first_moves = []
pos = [[0] for i in moves]

words = []
for x,i in enumerate(moves):
	words.append(i.pop(0))
	if i[0][1] == Sound.KEY_SELECT:
		first_moves.append(i[0])
	else:
		first_moves.append(Move(num_moves=i[1][0], end_sound=Sound.SELECT))

regex = [[] for i in moves]
# for x,i in enumerate(first_moves):
# 	thing = ''
# 	if i[1] == Sound.SELECT:
# 		thing = get_possible(i, special, pos[x])
# 	else:
# 		thing = get_possible(i, standard, pos[x])
# 	regex[x].append(thing[0])
# 	pos[x] = thing[1]
# pos = [[0]for i in moves]
for x,i in enumerate(moves):
	page2 = False
	for j in i:
		if j[1] == Sound.SELECT:
			if page2:
				page2 = False
				pos[x] = [0]
			else:
				page2 = True
				pos[x] = [0]
		else:
			thing = []
			if page2:
				thing = get_possible(j, special, pos[x])
			else:
				thing = get_possible(j, standard, pos[x])
			regex[x].append(thing[0])
			pos[x] = thing[1]

# for i in regex:
# 	print(len(i))
# for x,i in enumerate(moves):
# 	page2 = False
# 	for j in i:
# 		if j[1] == Sound.KEY_SELECT:
# 			if page2:
# 				regex[x].append(page_2)
# 			else:
# 				regex[x].append(page_1)
# 		else:
# 			if page2:
# 				page2 = False
# 			else:
# 				page2 = True
# print(regex)
totals = []
averages = []
for x,i in enumerate(regex):
	#print(regex[x].pop(0))
	totals.append([])
	#print(regex[x])
	for y,j in enumerate(i):
		totals[x].append(0)
		prev = ''
		for k in j:
			if k != '\\':
				totals[x][y]+=1
			elif prev == '\\':
				totals[x][y]+=1
			prev = k
		#print(totals[x])
		#print(j)
			# totals[-1].append(len(k))
	#print(regex[x].pop(0))
	#regex[x].insert(0,r'\b')
	#regex[x].insert(0,r'(')
	#regex[x].append(r'\b')
	#regex[x].append(r')')

# print(len(totals))
# print(len(regex))
# print(regex[0])
# print(len(regex[0]))
# print(totals)
for i in totals:
	# print(i)
	averages.append(1)
	# print(averages)
	for j in i:
		# print(j)
		#averages*=j
		averages[-1]*=j
total = 0
count = 0
#print(averages)
for i in averages:
	#print(total)
	total+=i
	count+=1
# print(total/count)

words_combined = ' '.join(words)
incorrect = 0
# for x,i in enumerate(regex):
# 	#print(i)
# 	#print('\n')
# 	#reg = ''.join(i)
# 	#print(reg)
# 	#temp = re.findall(''.join(i),words_combined)
# 	if words[x] in temp:
# 		incorrect+=1
print(incorrect)

with open('masks.txt', 'w') as f:
	for exp in regex:
		f.write(''.join(exp))


#HASHCAT
# rn = regex[0]
# # for i in regex:
# # 	print(len(i))
# wordrn = words[0]
# rn_move = moves[0]
# rn_page = []
# rn.pop(0)
# rn.pop(-1)
# rn_temp = [i for i in rn]
# #print(len(rn))
# print(wordrn)
# shortest = [rn_temp.pop(0), rn_temp.pop(0)]

# for i in rn:
# 	if len(shortest[0])>len(shortest[1]):
# 		shortest[0],shortest[1] = shortest[1],shortest[0]
# 	if len(i)<len(shortest[1]):
# 		shortest[1] = i
# print(shortest)
# page2 = False
# for i in rn_move:
# 	if i[1]==Sound.SELECT:
# 		if page2:
# 			page2 = False
# 		else:
# 			page2 = True
# 	else:
# 		if page2:
# 			rn_page.append(1)
# 		else:
# 			rn_page.append(0)
# print(rn_page)
# output = []
# for idx,exp in enumerate(rn):
# 	print(exp)
# 	if exp not in shortest:
# 		if rn_page[idx]==0:
# 			output.append(1)
# 		else:
# 			output.append(2)
# 	elif exp == shortest[1]:
# 		output.append(3)
# 	else:
# 		output.append(4)
# print(output)
# charsets = []

# charsets.append(page_1)
# charsets.append(page_2)
# for idx,exp in enumerate(shortest):
# 	charsets.append([])
# 	for letter in list(exp):
# 		if letter == ',':
# 			charsets[idk+2].append('\\')
# 		charsets[idx+2].append(letter)
# 		if letter == '?':
# 			charsets[idx+2].append('?')
# 	charsets[idx+2].pop(0)
# 	charsets[idx+2].pop()
# 	charsets[idx+2] = ''.join(charsets[idx+2])
# print(charsets)
# mask = ''
# mask+='[?1,'
# mask+=charsets[0]
# mask+='][?2,'
# mask+=charsets[1]
# mask+='][?3,'
# mask+=charsets[2]
# mask+='][?4,'
# mask+=charsets[3]
# mask+=']'
# print(mask)
# for num in output:
# 	mask+='?'
# 	mask+=str(num)
# print(mask)

# with open('mask.hcmask', 'w') as f:
# 	f.write(mask)