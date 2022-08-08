import argparse
import string
import random

parser = argparse.ArgumentParser()
parser.add_argument('-o', type=str, required=True)
parser.add_argument('-n', type=int, required=True)
parser.add_argument('-l', type=int, required=True)
args = parser.parse_args()

n=int(args.n/2)

chars = string.ascii_lowercase+string.digits
chars = list(chars)
special = string.punctuation
special = list(special)
special.remove('`')
special.remove('_')
special.remove('|')
all_chars = chars+special

passwords = [[] for i in range(args.n)]
for i in range(n):
	# length = int(random.random()*5)+5
	length = args.l
	for j in range(length):
		passwords[i].append(random.choice(chars))
	passwords[i] = ''.join(passwords[i])
for i in range(n):
	length = args.l
	passwords[i+n].append(random.choice(special))
	for j in range(length-1):
		passwords[i+n].append(random.choice(all_chars))
	random.shuffle(passwords[i+n])
	passwords[i+n] = ''.join(passwords[i+n])
random.shuffle(passwords)

with open(args.o, 'w+') as f:
	f.writelines('\n'.join(passwords))
print('1')