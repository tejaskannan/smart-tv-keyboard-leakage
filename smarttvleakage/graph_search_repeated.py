import search_without_autocomplete
import argparse
import csv
from dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from datetime import datetime, timedelta
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.keyboard_utils.unpack_jsonl_gz import read_moves

parser = argparse.ArgumentParser()
parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
parser.add_argument('--moves-list', type=str, required=True, help='jsonl. of words and the moves')
parser.add_argument('--output', type=str, required=True, help='CSV output file')
parser.add_argument('--tv-type', type=str, required=True, choices=[tv_type.name.lower() for tv_type in SmartTVType], help='The type of TV to use.')
#parser.add_argument('-cutoff', type=int, required=True, help='When to stop if a word is taking too long')
args = parser.parse_args()

graph = MultiKeyboardGraph()
characters = graph.get_characters()

if args.dictionary_path == 'uniform':
    dictionary = UniformDictionary(characters=characters)
else:
    dictionary = EnglishDictionary.restore(characters=characters, path=args.dictionary_path)

f = open(args.moves_list)
c = csv.reader(f, delimiter = '\t')
out = open(args.output, 'w')

c=read_moves(args.moves_list)
print(c)

out1 = []
for row in c:
  print(row)
  answer = row.pop(0)
  for idx, (guess, score, candidates_count) in enumerate(search_without_autocomplete.get_words_from_moves(row, graph=graph, dictionary=dictionary, max_num_results=None)):
    print(guess)
    if answer == guess:
        temp = []
        temp.append(answer)
        temp.append(idx+1)
        temp.append(candidates_count)
        print('Found {}. Rank {}. # Considered Strings: {}'.format(guess, idx + 1, candidates_count))
        out.write('\n')
        out1.append(temp)
        break
f.close()
w = csv.writer(out, delimiter='\t')
for i in out1:
  w.writerow(i)
out.close()
