import json
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help='enter the graph')
parser.add_argument('-o', type=str, help='enter the output csv')
args = parser.parse_args()

f = open(args.f)
kb = json.load(f)
f.close()
#print(kb)



def combine_dicts(dict1, dict2):
	for keys in dict2.keys():
		for i in dict2[keys]:
			dict1[keys].append(i)
	return dict1

possibilities = []
base_adjacency = kb['adjacency_list']
#print(base_adjacency)
shortcuts = kb['shortcuts']
wraparound = kb['wraparound']
possibilities.append(('normal',base_adjacency))
possibilities.append(('shortcuts',combine_dicts(base_adjacency, shortcuts)))
possibilities.append(('wraparound',combine_dicts(base_adjacency, wraparound)))
possibilities.append(('both',combine_dicts(combine_dicts(base_adjacency, shortcuts),wraparound)))
output = []

for (type,keyboard) in possibilities:
	keys=sorted(keyboard.keys())
	size=len(keys)

	#Create adjacency matrix

	M = [ [0]*size for i in range(size) ]

	for a,b in [(keys.index(a), keys.index(b)) for a, row in keyboard.items() for b in row]:
	     M[a][b] = 2 if (a==b) else 1


	for c in range(len(M)):
		for r in range(len(M[c])):
			if M[c][r] == 0:
				if r!=c:
					M[c][r] = float(M[c][r])
					M[c][r] = np.inf

	#make into numpy array
	a = np.asarray(M)
	graph = csr_matrix(M)

	#run floyd warshall
	dist_matrix, predecessors = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)
	dist_matrix = dist_matrix.tolist()
	dist_matrix.insert(0, keys)

	output.append((type,dist_matrix))

for (name,matrix) in output:
	#save
	filename = '_'+name+'.csv'
	f = open(args.o.replace('.csv',filename), 'w+')
	csvWriter = csv.writer(f,delimiter=',')
	csvWriter.writerows(matrix)
	f.close()
