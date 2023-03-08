
from typing import List, Dict, Tuple
from argparse import ArgumentParser

import random
import json

import numpy as np

from smarttvleakage.suggestions_model.simulate_ms import simulate_ms
from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.dictionary import EnglishDictionary, restore_dictionary

# fields = ['credit_card', 'security_code', 'zip_code', 'exp_month', 'exp_year']


def gen_year():
    year = 24
    year += random.randint(0, 5)
    return str(year)

def gen_month():
    month = random.randint(1, 12)
    if month < 10:
        return "0" + str(month)
    return str(month)

def gen_sec(digits):
    sec = ""
    for i in range(digits):
        sec += str(random.randint(0, 9))
    return sec


# testing if adding population counts overflows
def test_prob_gen_zip(path):
    with open(path) as f:

        zips = {}

        lines = f.readlines()
        for line in lines:
            data = line.split(' ')
            zips[data[0]] = int(data[1])

    #print(zips["93405"])
    #print(len(list(zips.items())))
    #print(sum(list(zips.values())))
    fail = 0

    leng = len(list(zips.items()))
    vals = list(zips.values())
    last = 0
    for i in range(1000):
        test_sum = sum(vals[:(leng // (1000-i))])
        print(test_sum)
        if test_sum < last:
            fail = 1
        last = test_sum

    print("fail? " + str(fail))
    return ""

    

def build_zip(path):
    with open(path) as f:

        zip_data = []

        total = 0

        lines = f.readlines()
        for line in lines:
            total += int(line.split(' ')[1])
        print("total: " + str(total))

        for line in lines:
            splits = line.split(' ')

            zip = splits[0]
            prob = int(splits[1]) / total
            raw = int(splits[1])

            zip_data.append((zip, prob, raw))
        
        f.close()
        
    return zip_data

    
def test__gen_zip(zip_data):
    probs = list(map(lambda x : x[1], zip_data))
    zips = list(map(lambda x : x[0], zip_data))
    raws = list(map(lambda x : x[2], zip_data))
    
    for i in range(10):
        choice = np.random.choice(range(len(zip_data)), p=probs)
        print("zip: " + zips[choice])
        print("prob: " + str(probs[choice]))
        print("raw: " + str(raws[choice]))

    print("sum: " + str(sum(probs)))


    
def gen_zip(zips, probs):

    choice = np.random.choice(zips, p=probs)
    return choice









def build_cc_list(path):

    cc_list = []
    skipped = 0

    with open(path) as f:
        ccs = json.load(f)
        for card in ccs:
            cc = card["CreditCard"]

            # check if a duplicate
            if cc["CardNumber"] in cc_list:
                skipped += 1
                print("skipping: " + str(cc["CardNumber"]))
                continue

            cc_list.append(str(cc["CardNumber"]))

        print("skipped total: " + str(skipped))

        f.close()
    return cc_list, skipped




def build_move_list(str, ed, ss):
    ms = simulate_ms(ed, ss, str, False)
    moves = []
    for m in ms:
        move = {}
        move["num_moves"] = m
        move["end_sound"] = "key_select"
        move["directions"] = "any"
        moves.append(move)
    return moves
    





def make_cc_json(cc_path, zip_path, output_path_base, digits, ed, ss):

    zip_data = build_zip(zip_path)
    zips = list(map(lambda x : x[0], zip_data))
    probs = list(map(lambda x : x[1], zip_data))

    cc_list, skipped = build_cc_list(cc_path)
    if skipped > 0:
        print("skipped: " + str(skipped))
        return
    

    fn = 0
    counter = 0
    info = {}
    info["cards"] = []

    for cc in cc_list:
        print("fn: " + str(fn) + "counter: " + str(counter))

        ccd = {}
        ccd["ccn"] = cc
        ccd["cvv"] = gen_sec(digits)
        ccd["zip_code"] = gen_zip(zips, probs)
        ccd["exp_month"] = gen_month()
        ccd["exp_year"] = gen_year()

        ccd["ccn_moves"] = build_move_list(ccd["ccn"], ed, ss)
        ccd["cvv_moves"] = build_move_list(ccd["cvv"], ed, ss)
        ccd["zip_code_moves"] = build_move_list(ccd["zip_code"], ed, ss)
        ccd["exp_month_moves"] = build_move_list(ccd["exp_month"], ed, ss)
        ccd["ccn_year_moves"] = build_move_list(ccd["exp_year"], ed, ss)


        info["cards"].append(ccd)

        counter += 1
        if counter == 500: # now write a file

            output_path = output_path_base + str(fn) + ".json"
            with open(output_path, "w") as f:
                json.dump(info, f)
                f.close()

            fn += 1
            counter = 0
            info = {}
            info["cards"] = []

    
    if counter > 0: # one final write
        output_path = output_path_base + str(fn) + ".json"
        with open(output_path, "w") as f:
            json.dump(info, f)
            f.close()

    return info










if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cc-path", type=str, required=False) 
    parser.add_argument("--zip-path", type=str, required=False) 
    parser.add_argument("--ed-path", type=str, required=False)
    parser.add_argument("--ss-path", type=str, required=False) #single suggestions .json
    parser.add_argument("--digits", type=str, required=False)

    args = parser.parse_args()

    if args.ed_path is None:
        ed = restore_dictionary("suggestions_model/local/dictionaries/ed.pkl.gz")
    if args.ss_path is None:
        ss = "graphs/autocomplete.json"


    #build_cc_list(args.cc_path)
    #zip_data = build_zip(args.zip_path)
    #test__gen_zip(zip_data)

    make_cc_json(args.cc_path, args.zip_path, "cc_gen/amex_r2/mc_", int(args.digits), ed, ss)

    # https://www.getcreditcardnumbers.com/generated-credit-card-numbers