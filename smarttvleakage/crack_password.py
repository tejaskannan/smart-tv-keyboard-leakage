import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.password_cracker.classifier import find_regex
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('-password', type=str, required=True)
args = parser.parse_args()

hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', args.password.strip()], capture_output=True, text=True)

with open('hashed_password.txt', 'w') as f:
    f.write(hashed.stdout)

beginning_datetime = datetime.now()
beginning_perf = perf_counter()

masks = find_regex(findPath(args.password, False, False, 0, 0, 0))

mask_line = "-mask='"+masks[0]+"'"

john_datetime = datetime.now()
john_perf = perf_counter()

password = subprocess.run(['/home/abebdm/john-1.9.0-jumbo-1/run/john', mask_line, '/home/abebdm/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'])

after_perf = perf_counter()
after_datetime = datetime.now()

with open('times_{}.txt'.format(args.password), 'w') as f:
	f.write('Datetime from start: ', after_datetime-beginning_datetime, '\n')
	f.write('Perf_counter from start: ', after_perf-beginning_perf, '\n')
	f.write('Datetime of jtr: ', after_datetime-beginning_john, '\n')
	f.write('Perf_counter of jtr: ', after_perf-john_perf, '\n')
