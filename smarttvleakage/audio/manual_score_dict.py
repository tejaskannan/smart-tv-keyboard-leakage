import time
import io
import os.path
import string
from argparse import ArgumentParser
from queue import PriorityQueue
from collections import defaultdict, namedtuple
from typing import Set, List, Dict, Optional, Iterable, Tuple

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph, KeyboardMode, START_KEYS
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.keyboard_utils.word_to_move import findPath

graph = MultiKeyboardGraph()
dictionary = UniformDictionary()
# englishDictionary = EnglishDictionary.restore(path="../local/dictionaries/ed.pkl.gz")
englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")


def default_value():
    return 0

def buildDict(min_count):
    print("building dict...")
    word_counts = defaultdict(default_value)
    path = "..\local\dictionaries\enwiki-20210820-words-frequency.txt"

    with open(path, 'rb') as fin:
        io_wrapper = io.TextIOWrapper(fin, encoding='utf-8', errors='ignore')

        for line in io_wrapper:
            line = line.strip()
            tokens = line.split()

            if len(tokens) == 2:
                count = int(tokens[1])

                if count > min_count:
                    word_counts[tokens[0]] = count
    print("done.")
    return word_counts


def build_file():
    minCount = 500
    word_counts = buildDict(minCount)
    done = []

    path = "manual_score_dict_2.txt"
    with open(path, "w") as f:

        for word in word_counts:
            skip = 0
            for char in word:
                if char not in string.ascii_letters:
                    skip = 1
            if skip == 1:
                continue

            ms = findPath(word, 0)
            if ms in done:
                continue

            done.append(ms)
            line = ""

            for num in ms:
                line = line + str(int(num)) + ","
            line = line + ";"

            line = line + word + ";"

            raw_score = englishDictionary.get_score_for_string(word, False) 
            line = line + str(raw_score) + "\n"

            #raw_score = englishDictionary.get_score_for_string(word, False)
            
            f.write(line)


# maybe build this as a dictionary??

def build_msfd():
    path = "audio/manual_score_dict_2.txt"
    d = {}
    with open(path) as f:
        lines = f.readlines()
    
    for line in lines:
        d[line.split(";")[0]] = (line.split(";")[1], float(line.split(";")[2]))

    return d

    
def get_word_from_ms(ms):
    ms_string = ""
    for m in ms:
        ms_string += str(m) + ","
    ms_string += ";"

    path = "audio/manual_score_dict_2.txt"
    with open(path) as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith(ms_string):
            return (line.split(";")[1], float(line.split(";")[2]))
    return ("", 0)


if __name__ == '__main__':


    print(get_word_from_ms([3, 5, 4])[0])
    print(float(get_word_from_ms([3, 5, 4])[1]))
    print(get_word_from_ms([6, 5, 1]))
    print(get_word_from_ms([3, 5, 4, 1, 3]))
    