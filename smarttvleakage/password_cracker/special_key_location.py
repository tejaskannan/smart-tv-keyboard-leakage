import argparse
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.audio import Move, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT
from typing import Set, List, Dict, Optional, Iterable, Tuple
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, START_KEYS, SAMSUNG_STANDARD, SAMSUNG_SPECIAL_ONE
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SPACE, SELECT_SOUND_KEYS
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
import string
from smarttvleakage.keyboard_utils.unpack_jsonl_gz import read_moves
from smarttvleakage.utils.file_utils import save_jsonl_gz


special_chars = string.punctuation

def test_with_search(target, move_sequence: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary):
	guesses = []
	for (guess, score, candidates_count) in get_words_from_moves(move_sequence, graph=graph, dictionary=dictionary, max_num_results=100):
		#print('1')
		guesses.append((guess,score))
	#print(guesses)
	total = 0
	w = 0
	output = []
	for i in range(len(target)):
		output.append(0)
		for (guess, score) in guesses:
			if i>len(guess)-1:
				continue
			#print(guess)
			#print(score)
			# print(i)
			# print(guess)
			# print('\n')
			# print(len(guess))
			# print("\n")
			if list(guess)[i] in special_chars:
				# print(score)
				output[i]+=score
		#print("\n")
	for (guess,score) in guesses:
		total+=score
	for x,i in enumerate(output):
		output[x]=float(i)/total
	print(output)
	return output


if __name__ == '__main__':
	# idx_of_last_sound = -1
	# for idx, (_,sound) in enumerate(move_seq):
	# 	if sound == SAMSUNG_SELECT and idx_of_last_sound == -1:
	# 		idx_of_last_sound = idx
	# 	elif sound == SAMSUNG_SELECT and idx_of_last_sound+2 == idx and idx_of_last_sound != -1:
	# 		print('There is probably a special character at index {}'.format(idx-1))
	# 		idx_of_last_sound = -1


	parser = argparse.ArgumentParser()
	parser.add_argument('--dictionary-path', type=str, required=True)
	parser.add_argument('--moves-list', type=str, required=True)
	parser.add_argument('--o', type=str, required=True)
	args = parser.parse_args()

	graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)

	moves = read_moves(args.moves_list)

	if args.dictionary_path == 'uniform':
		dictionary = UniformDictionary()
	else:
		dictionary = EnglishDictionary.restore(path=args.dictionary_path)

        dictionar.set_characters(graphs.get_characters())

	output = []
	for z,move_seq in enumerate(moves):
		word = move_seq.pop(0)
		print(word)
		if len(move_seq)>12:
			continue
		output.append({"word":word,"prediction":test_with_search(word, move_seq, graph, dictionary)})
		print('\n')
		#turn into json lines where one key is target word and other key is thoughts

	save_jsonl_gz(output, args.o)
