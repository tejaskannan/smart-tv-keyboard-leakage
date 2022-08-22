import argparse
import string
import random


chars = string.ascii_lowercase+string.digits
chars = list(chars)
special = string.punctuation
special = list(special)
special.remove('`')
special.remove('_')
special.remove('|')
special.append(' ')
all_chars = chars+special


def generate_password(number, length):
	passwords = [[] for i in range(number)]
	for i in range(number):
		for j in range(length):
			passwords[i].append(random.choice(all_chars))
		passwords[i] = ''.join(passwords[i])
	return passwords
	# n = int(number / 2)
	# passwords = [[] for i in range(number)]
	# for i in range(number):
	# 	for j in range(length):
	# 		passwords[i].append(random.choice(chars))
	# for i in range(n):
	# 	passwords[i+n].append(random.choice(special))
	# 	for j in range(length-1):
	# 		passwords[i+n].append(random.choice(all_chars))
	# 	random.shuffle(passwords[i+n])
	# for i, _ in enumerate(passwords):
	# 	passwords[i] = ''.join(passwords[i])
	# random.shuffle(passwords)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', type=str, required=True, help='output text file for the passwords')
	parser.add_argument('-n', type=int, required=True, help='number of passwords to be generated')
	parser.add_argument('-l', type=int, required=True, help='length of each password')
	passwords = generate_password(args.n, args.l)
	args = parser.parse_args()
	with open(args.o, 'w+') as f:
		f.writelines('\n'.join(passwords))
	print(passwords)
