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
import csv

# kb = MultiKeyboardGraph(KeyboardType.APPLE_TV_SEARCH)
kb = MultiKeyboardGraph(KeyboardType.SAMSUNG)
# error_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
error_rates = [0]
# passwords = generate_password(10, 5)
passwords = ['a l s']
times = [[] for i in error_rates]
# passwords = ['=xh{4h']

with open('times_passwords.txt', 'w') as f:
        for i in passwords:
                f.write(i)
                f.write('\n')

for password in passwords:
        for error in error_rates:
                print('\n')
                print(password)
                print('\n')
                if os.path.exists('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot'):
                        os.remove('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot')

                hashed = subprocess.run(['openssl', 'passwd', '-5', '-salt', 'agldf', password], capture_output=True, text=True)

                with open('hashed_password.txt', 'w') as f:
                    f.write(hashed.stdout)
                # print(password)
                path = findPath(password, True, False, False, False, error, 0.9, 10, kb)
                # print(len(path))
                print('\n')
                print('\n')
                print(path)
                print('\n')
                path_no_select = 0
                for i in path:
                        if i[1] != 'select':
                                path_no_select+=1
                # print(path_no_select)
                beginning_perf = perf_counter()

                masks = get_regex([path])
                # print(masks)
                # # print('\n')
                # for mask in masks:
                #         for mask1 in mask:
                #                 #print(mask1)
                #                 for mask2 in mask1:
                #                         #print(mask2)
                #                         # print(len(mask2))

                for mask in masks[0]:
                        mask_line = "-mask='"+''.join(mask[0])+"'"
                        print('\n')
                        # print(password)
                        print(mask_line)
                        print('\n')
                        passwrd = subprocess.run(['/home/abebdm/john-1.9.0-jumbo-1/run/john {} /home/abebdm/Desktop/Thing/smart-tv-keyboard-leakage/smarttvleakage/hashed_password.txt'.format(mask_line)], shell=True)
                        
                        if os.path.exists('/home/abebdm/john-1.9.0-jumbo-1/run/john.pot'):
                                if os.stat("/home/abebdm/john-1.9.0-jumbo-1/run/john.pot").st_size > 0:
                                        break

                after_perf = perf_counter()
                times[error_rates.index(error)].append(after_perf-beginning_perf)

print('\n')
print(times)


with open('times_random.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(times)