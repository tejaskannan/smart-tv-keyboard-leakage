import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from smarttvleakage.keyboard_utils.word_to_move import findPath
from smarttvleakage.password_cracker.classifier import find_regex
from time import perf_counter
import json
from smarttvleakage.keyboard_utils.generate_passwords import make_passwords
import random
import csv
import os

passwords = make_passwords(15,5)
for x,i in enumerate(passwords):
    if i == []:
        passwords.remove(i)
error_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
#decay_rates = [0.9, 0.8, 0.75, 0.7]

#passwords = [j for i in passwords for j in i]

#parse.ArgumentParser()
#parser.add_argument('-password', type=str, required=True)
#args = parser.parse_args()

times = [[] for i in error_rate]

times_dict = {}
print(passwords)
print(error_rate)
for i in passwords:
    print(i)
    for j in error_rate:
        os.remove('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot')
        print(j)
        #print(i)
        hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', i.strip()], capture_output=True, text=True)

        with open('hashed_password.txt', 'a') as f:
          f.write(hashed.stdout)

        path = findPath(i, False, False, j, 0.9, 10)

        beginning_perf = perf_counter()

        masks = find_regex(path)
        #print(masks[0])
        mask_line = "-mask='"+''.join(masks[0])+"'"
        #print(mask_line)


        john_perf = perf_counter()

        password = subprocess.run(['/home/abebdm/john-1.9.0-jumbo-1/run/john', mask_line, '/home/abebdm/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'])

        after_perf = perf_counter()

        times_dict[i] = [after_perf-beginning_perf, after_perf-john_perf]
        times[error_rate.index(j)].append(after_perf-beginning_perf)

#with open('times.json', 'w') as f:
#    json.dump(times_dict, f)

with open('times.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerows(times)
