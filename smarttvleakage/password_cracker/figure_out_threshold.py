from smarttvleakage.utils.file_utils import read_jsonl_gz
import argparse
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json


def read_jsonl(path):
	output = []
	for x,i in enumerate(read_jsonl_gz(path)):
		output.append([])
		output[x].append(i["word"])
		output[x].append(i["prediction"])
	return output

def get_prediction(prediction, threshold):
	output = []
	#print(prediction)
	for i in prediction:
		if i>threshold:
			output.append(1)
		else:
			output.append(0)
	return output

def get_prediction_with_positions(prediction, thresholds):
	output = []
	for i in prediction:
		for x,j in enumerate(i):
			if x == len(i)-1:
				b = -2
			elif x>10:
				b=-1
			else:
				b=x
			if j>thresholds[b]:
				output.append(1)
			else:
				output.append(0)
	return output

def binary_search(truth,prediction):
	threshold = 0.5
	threshold_change=0.25
	while threshold_change>1e-4:
		lower = accuracy_score(truth,get_prediction(prediction,threshold-threshold_change))
		upper = accuracy_score(truth,get_prediction(prediction,threshold+threshold_change))
		# print(lower)
		# print(upper)
		# print(threshold)
		# print(threshold_change)
		if lower>upper:
			#print('lower')
			threshold = threshold-threshold_change
		else:
			#print('upper')
			threshold = threshold+threshold_change
		threshold_change*=0.5
		#print('\n')
	return threshold

def binary_search_2(truth,prediction,upper_certainty, lower_certainty):
	upper_threshold=0.5
	lower_threshold=0.5
	threshold_change = 0.25
	while precision_score(truth,get_prediction(prediction,upper_threshold))<upper_certainty and threshold_change>1e-5:
		#print(upper_threshold)
		lower = precision_score(truth,get_prediction(prediction,upper_threshold-threshold_change))
		upper = precision_score(truth,get_prediction(prediction,upper_threshold+threshold_change))
		if lower>upper:
			# print('lower')
			# print(lower)
			upper_threshold = upper_threshold-threshold_change
		else:
			# print('upper')
			# print(upper)
			upper_threshold = upper_threshold+threshold_change
		threshold_change*=0.5
	for x,i in enumerate(truth):
		truth[x] = 1-truth[x]

	for x,i in enumerate(prediction):
		prediction[x]=1-i

	threshold_change=0.25
	while precision_score(truth,get_prediction(prediction,lower_threshold))<lower_certainty and threshold_change>1e-5:
		lower = precision_score(truth,get_prediction(prediction,upper_threshold-threshold_change))
		upper = precision_score(truth,get_prediction(prediction,upper_threshold+threshold_change))
		if lower>upper:
			#print('lower')
			lower_threshold = lower_threshold-threshold_change
		else:
			#print('upper')
			lower_threshold = lower_threshold+threshold_change
		threshold_change*=0.5
	print(precision_score(truth,get_prediction(prediction,upper_threshold)))
	print(precision_score(truth,get_prediction(prediction,lower_threshold)))
	return(upper_threshold,lower_threshold)

def test_ten_percent_up(target, upper_threshold):
	prediction = []
	truth = []
	idx=[]
	for i in list(target[0]):
		if i in special_chars:
			truth.append(1)
		else:
			truth.append(0)
	for i in target[1]:
		if i>upper_threshold:
			prediction.append(1)
		else:
			prediction.append(0)
	for x,i in enumerate(truth):
		if i!=prediction[x] and i==0:
			idx.append(x)
	return idx

def test_ten_percent_down(target, lower_threshold):
	prediction = []
	truth = []
	idx=[]
	for i in list(target[0]):
		if i in special_chars:
			truth.append(1)
		else:
			truth.append(0)
	for i in target[1]:
		if i>lower_threshold:
			prediction.append(1)
		else:
			prediction.append(0)
	for x,i in enumerate(truth):
		if i!=prediction[x] and i==1:
			idx.append(x)
	return idx


special_chars = string.punctuation
letters = string.ascii_lowercase
numbers = string.digits

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True)
parser.add_argument('-i1', type=str, required=True)
args = parser.parse_args()


truth = [[] for i in range(11)]
prediction = [[] for i in range(11)]

# data = read_jsonl(args.i)
# for i in data:
# 	b = ''
# 	for x,j in enumerate(list(i[0])):
# 		b=j
# 		if j in special_chars:
# 			truth[x].append(1)
# 		else:
# 			truth[x].append(0)
# 	if b in special_chars:
# 		truth[10].append(1)
# 	else:
# 		truth[10].append(0)
# 	for x,j in enumerate(i[1]):
# 		prediction[x].append(j)
# 	prediction[10].append(i[1][-1])

# threshold_for_position = [0 for i in range(11)]

# for i in range(len(threshold_for_position)):
# 	threshold_for_position[i] = binary_search(truth[i],prediction[i])

# truth_full = []
# prediction_full = []

# for i in truth:
# 	for j in i:
# 		truth_full.append(j)
# for i in prediction:
# 	for j in i:
# 		prediction_full.append(j)

# threshold = binary_search(truth_full,prediction_full)
# threshold_for_position.append(threshold)
# print(threshold_for_position)
# print('accuracy for letter: ',accuracy_score(truth_full, get_prediction_with_positions(prediction, threshold_for_position)))

# print(threshold)
# prediction1 = get_prediction(prediction_full, threshold)
# print('accuracy: ',accuracy_score(truth_full,prediction1))
# print('precision: ',precision_score(truth_full,prediction1))
# print('recall: ',recall_score(truth_full,prediction1))
# print('\n')
# prediction2 = get_prediction(prediction_full, 0)
# print('accuracy: ',accuracy_score(truth_full,prediction2))
# print('precision: ',precision_score(truth_full,prediction2))
# print('recall: ',recall_score(truth_full,prediction2))

# thresholds = binary_search_2(truth_full, prediction_full, 0.99, 0.01)
# print(thresholds)
# mistakes = {"upper": [], "lower": []}
# for i in data:
# 	if test_ten_percent_up(i, thresholds[0])!=[]:
# 		mistakes["upper"].append([i,test_ten_percent_up(i, thresholds[0])])
# 	if test_ten_percent_down(i, thresholds[1])!=[]:
# 		mistakes["lower"].append([i,test_ten_percent_down(i, thresholds[0])])

# with open('mistakes.json', 'w') as f:
# 	json.dump(mistakes, f)

data = read_jsonl(args.i)
letters_truth = []
numbers_truth = []
special_truth = []

letters_pred = []
numbers_pred = []
special_pred = []

for i in data:
	print(i)
	letter = False
	number = False
	special = False
	for z,j in enumerate(list(i[0])):
		if j in letters:
			if letter == False:
				letters_truth.append(1)
			letter = True
		elif j in special_chars:
			if special == False:
				special_truth.append(1)
			special = True
		elif j in numbers:
			if number == False:
				numbers_truth.append(1)
			number = True
	if letter == False:
		letters_truth.append(0)
	if number == False:
		numbers_truth.append(0)
	if special == False:
		special_truth.append(0)

	for z,j in enumerate(i[1]):
		print(j)
		print(z)
		if z%3==0:
			letters_pred.append(j)
		elif z%3==1:
			numbers_pred.append(j)
		elif z%3==2:
			special_pred.append(j)

letters_threshold = binary_search(letters_truth, letters_pred)
numbers_threshold = binary_search(numbers_truth, numbers_pred)
special_threshold = binary_search(special_truth, special_pred)
print('letter: ', letters_threshold)
print('accuracy: ',accuracy_score(letters_truth, get_prediction(letters_pred,letters_threshold)))
print('number: ', numbers_threshold)
print('accuracy: ',accuracy_score(numbers_truth, get_prediction(numbers_pred,numbers_threshold)))
print('special: ', special_threshold)
print('accuracy: ',accuracy_score(special_truth, get_prediction(special_pred,special_threshold)))

special_truth_1 = []
for x,i in enumerate(data):
	special_truth_1.append([])
	for j in list(i[0]):
		if j in special_chars:
			special_truth_1[x].append(1)
		else:
			special_truth_1[x].append(0)

data1 = read_jsonl(args.i1)

special_truth_1 = []
special_pred_1 = []
for x,i in enumerate(data1):
	special_truth_1.append([])
	for j in list(i[0]):
		if j in special_chars:
			special_truth_1[x].append(1)
		else:
			special_truth_1[x].append(0)
	special_pred_1.append([])
	for j in i[1]:
		if j > 0.5467529296875:
			special_pred_1[x].append(1)
		else:
			special_pred_1[x].append(0)

special_pred_2 = []
for z,i in enumerate(special_pred_1):
	special_pred_2.append(0)
	for j in i:
		if j == 1:
			special_pred_2[-1] = 1
			break

special_truth_2 = []
for x,i in enumerate(special_truth_1):
	special_truth_2.append(0)
	for j in i:
		if j==1:
			special_truth_2[x-1]
print(len(special_truth_2))
print(len(special_pred_2))

print('special 2: ', accuracy_score(special_truth_2,special_pred_2))