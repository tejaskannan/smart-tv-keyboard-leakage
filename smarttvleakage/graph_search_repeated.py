import graph_search
import argparse
import csv


parser = ArgumentParser()
parser.add_argument('--dictionary-path', type=str, required=True, help='Path to the dictionary pkl.gz file.')
parser.add_argument('--moves-list', type=str, required=True, help='A tab delimited csv of words and the moves')
parser.add_argument('--output', type=str, required=True, help='CSV output file')
parser.add_argument('-cuttoff', type=int, required=True, help='When to stop if a word is taking too long')
args = parser.parse_args()

graph = MultiKeyboardGraph()

if args.dictionary_path == 'uniform':
    dictionary = UniformDictionary()
else:
    dictionary = EnglishDictionary.restore(path=args.dictionary_path)

f = open(args.moves_list)
c = csv.reader(f, delimiter = '\t')
out = open(args.output, 'w')
out1 = []
for row in c:
  answer = row.pop(0)
  for x in range(len(row)):
    row[x] = int(float(row[x]))
  for idx, (guess, score, candidates_count) in enumerate(graph_search.get_words_from_moves(num_moves=row, graph=graph, dictionary=dictionary, max_num_results=None)):
    if idx+1>3000000:
        print('Not Found')
        break
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