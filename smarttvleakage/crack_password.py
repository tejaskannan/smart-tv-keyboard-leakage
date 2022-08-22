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
from smarttvleakage.keyboard_utils.generate_passwords import generate_password


parser = argparse.ArgumentParser()
parser.add_argument('-password', type=str, required=True)
args = parser.parse_args()

# kb = MultiKeyboardGraph(KeyboardType.APPLE_TV_SEARCH)
kb = MultiKeyboardGraph(KeyboardType.SAMSUNG)
error_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
passwords = generate_password(10, 5)
times = [[] for i in error_rates]

for password in passwords:
	for error in error_rates:
		if os.path.exists('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot'):
			os.remove('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot')

		hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', args.password.strip()], capture_output=True, text=True)

		with open('hashed_password.txt', 'w') as f:
		    f.write(hashed.stdout)
		# print(password)
		path = findPath(password, False, False, error, 0.9, 10, kb)

		beginning_perf = perf_counter()

		masks = get_regex([path])

		for mask in masks:
			mask_line = "-mask='"+''.join(mask[0][0])+"'"

			password = subprocess.run(['/home/abebdm/john-1.9.0-jumbo-1/run/john', mask_line, '/home/abebdm/Desktop/Thing/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'])
			
			if os.stat("/home/abebdm/john-1.9.0-jumbo-1/run/john.pot").st_size > 0:
				break

		after_perf = perf_counter()
		times[error_rates.index(error)].append(after_perf-beginning_perf)

print('\n')
print(times)

with open('times.csv', 'w') as f:
	csvwriter = csv.writer(f)
	csvwriter.writerows(times)
