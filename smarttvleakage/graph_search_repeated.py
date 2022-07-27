import search_without_autocomplete
import argparse
import csv
from dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from datetime import datetime, timedelta
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.audio.move_extractor import Move, Sound

parser = argparse.ArgumentParser()
parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
parser.add_argument('--moves-list', type=str, required=True, help='jsonl. of words and the moves')
parser.add_argument('--output', type=str, required=True, help='CSV output file')
#parser.add_argument('-cutoff', type=int, required=True, help='When to stop if a word is taking too long')
args = parser.parse_args()

graph = MultiKeyboardGraph()

if args.dictionary_path == 'uniform':
    dictionary = UniformDictionary()
else:
    dictionary = EnglishDictionary.restore(path=args.dictionary_path)

f = open(args.moves_list)
c = csv.reader(f, delimiter = '\t')
out = open(args.output, 'w')

c=[]
for i in read_jsonl_gz(args.moves_list):
    temp = []
    temp.append(i["word"])
    for j in i["move_seq"]:
        temp.append(Move(num_moves=int(j["num_moves"]),end_sound=Sound[j["sound"]]))
    c.append(temp)
print(c)

out1 = []
for row in c:
  answer = row.pop(0)
  for idx, (guess, score, candidates_count) in enumerate(search_without_autocomplete.get_words_from_moves(row, graph=graph, dictionary=dictionary, max_num_results=100)):
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
out1f.close()