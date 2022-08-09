import argparse
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.audio.move_extractor import Sound, Move
from typing import Set, List, Dict, Optional, Iterable, Tuple
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SPACE, SELECT_SOUND_KEYS
from smarttvleakage.utils.transformations import filter_and_normalize_scores, get_keyboard_mode, get_string_from_keys
import string
from smarttvleakage.keyboard_utils.unpack_jsonl_gz import read_moves
from smarttvleakage.utils.file_utils import save_jsonl_gz


special_chars = string.punctuation


def test_with_search(target, move_sequence: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary):
	guesses = []
	for (guess, score, candidates_count) in get_words_from_moves(move_sequence, graph=graph, dictionary=dictionary, max_num_results=100):
		guesses.append((guess, score))
	total = 0
	w = 0
	output = []
	for i in range(len(target)):
		output.append(0)
		for (guess, score) in guesses:
			if i > len(guess) - 1:
				continue
			if list(guess)[i] in special_chars:
				output[i] += score
	for (guess, score) in guesses:
		total += score
	for x, i in enumerate(output):
		output[x] = float(i) / total
	print(output)
	return output


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dictionary-path', type=str, required=True)
	parser.add_argument('--moves-list', type=str, required=True)
	parser.add_argument('--o', type=str, required=True)
	args = parser.parse_args()

	graph = MultiKeyboardGraph()

	moves = read_moves(args.moves_list)

	if args.dictionary_path == 'uniform':
		dictionary = UniformDictionary()
	else:
		dictionary = EnglishDictionary.restore(path=args.dictionary_path)

	output = []
	for z, move_seq in enumerate(moves):
		word = move_seq.pop(0)
		print(word)
		if len(move_seq) > 12:
			continue
		output.append({"word": word, "prediction": test_with_search(word, move_seq, graph, dictionary)})
		print('\n')
	save_jsonl_gz(output, args.o)
