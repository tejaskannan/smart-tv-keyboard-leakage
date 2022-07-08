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

f = open('samsung_keyboard_special_1.json')
kb = json.load(f)
f.close()

keys=sorted(kb.keys())
print(keys)
size=len(keys)

#Create adjacency matrix

M = [ [0]*size for i in range(size) ]

for a,b in [(keys.index(a), keys.index(b)) for a, row in kb.items() for b in row]:
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

#save
f = open(args.o, 'w+')
csvWriter = csv.writer(f,delimiter=',')
csvWriter.writerows(dist_matrix)
f.close()