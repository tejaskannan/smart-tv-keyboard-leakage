import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.password_cracker.classifier import get_regex
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.constants import KeyboardType
from time import perf_counter
import os

parser = argparse.ArgumentParser()
parser.add_argument('-password', type=str, required=True)
args = parser.parse_args()

kb = MultiKeyboardGraph(KeyboardType.APPLE_TV_PASSWORD)

if os.path.exists('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot'):
	os.remove('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot')

hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', args.password.strip()], capture_output=True, text=True)
print(hashed)
with open('hashed_password.txt', 'w') as f:
    f.write(hashed.stdout)

beginning_perf = perf_counter()

masks = get_regex([findPath(args.password, False, False, 0, 0, 0, kb)])

for mask in masks:
	print(mask[0][0])
	mask_line = "-mask='"+''.join(mask[0][0])+"'"
	print(mask_line)
	# john_datetime = datetime.now()
	# john_perf = perf_counter()

	password = subprocess.run(['/home/abebdm/john-1.9.0-jumbo-1/run/john', mask_line, '/home/abebdm/Desktop/Thing/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'])
	print('\n')
	print(password.args)
	
	if os.stat("/home/abebdm/john-1.9.0-jumbo-1/run/john.pot").st_size > 0:
		break

after_perf = perf_counter()

with open('times_{}.txt'.format(args.password), 'w') as f:
	f.write('Perf_counter from start: ', after_perf-beginning_perf, '\n')
	f.write('Datetime of jtr: ', after_datetime-beginning_john, '\n')
	f.write('Perf_counter of jtr: ', after_perf-john_perf, '\n')
