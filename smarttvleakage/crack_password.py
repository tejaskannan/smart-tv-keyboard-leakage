import argparse
import subprocess
import sys
from datetime import datetime, timedelta
#from smarttvleakage.keyboard_utils.word_to_move import findPath

parser = argparse.ArgumentParser()
parser.add_argument('-password', type=str, required=True)
args = parser.parse_args()

hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', args.password], capture_output=True, text=True)
print(hashed.stdout)

move_sequence = subprocess.run(['python3', 'keyboard_utils/word_to_move.py', '-i', args.password, '-o', 'password_moves.jsonl.gz', '-mr', '0', '-dr', '0', '-me', '0'])

mask = subprocess.run(['python3', 'password_cracker/classifier.py', '-i', 'password_moves.jsonl.gz'])

masks = []
with open('masks.txt', 'r') as f:
	for line in f:
		masks.append(line)

mask_line = "--mask='"+masks[0]

password = subprocess.run(['~/john-1.9.0-jumbo-1/run/john', mask_line, hashed.stdout], capture_output=True, text=True)
print(password.stdout)