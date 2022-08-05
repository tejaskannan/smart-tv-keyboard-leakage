print('hello')
import argparse
import subprocess
import sys
from datetime import datetime, timedelta

now = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('-password', type=str, required=True)
args = parser.parse_args()

print('1')
hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', args.password.strip()], capture_output=True, text=True)
print(hashed.stdout)
print('2')
move_sequence = subprocess.run(['python3', 'keyboard_utils/word_to_move.py', '-i', args.password, '-o', 'password_moves.jsonl.gz', '-mr', '0', '-dr', '0', '-me', '0'])
print('3')
mask = subprocess.run(['python3', 'password_cracker/classifier.py', '-i', 'password_moves.jsonl.gz'])
print('4')
masks = []
with open('masks.txt', 'r') as f:
	for line in f:
		masks.append(line)
print('5')
mask_line = "--mask='"+masks[0]+"'"

with open('hashed_password.txt', 'w') as f:
    f.write(hashed.stdout)
#hashed_pass = "'"+hashed.stdout+"'"
print(mask_line)
#print(['/home/abebdm/john-1.9.0-jumbo-1/run/john', mask_line, '/home/abebdm/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'])
password = subprocess.run("/home/abebdm/john-1.9.0-jumbo-1/run/john {} /home/abebdm/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt".format(mask_line), shell=True)
print('6')
#print(password.stdout)
print(datetime.now()-now)
with open('time.txt', 'w') as f:
    f.write(str(datetime.now()-now))
print(' '.join(password.args))
out = subprocess.run(['/home/abebdm/john-1.9.0-jumbo-1/run/john', '--show', '/home/abebdm/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'])