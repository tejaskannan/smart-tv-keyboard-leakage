
import csv
import string

from collections import defaultdict
import io 

from smarttvleakage.keyboard_utils.word_to_move import findPath

from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.utils.file_utils import read_json

from smarttvleakage.max.determine_autocomplete import build_model, classify_ms
from smarttvleakage.max.manual_score_dict import build_ms_dict





# test

# ..\local\dictionaries\enwiki-20210820-words-frequency.txt

def default_value():
    return 0

def buildDict(min_count):
    print("building dict...")
    word_counts = defaultdict(default_value)
    path = "local\dictionaries\enwiki-20210820-words-frequency.txt"

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








# evaluates simulated ms against gt, allowing off-by-one for autos
def eval_ms(key : tuple[str, str], ms : list[int]) -> int:
    ms_dict_non = build_ms_dict("non")
    ms_dict_auto = build_ms_dict("auto")

    word = key[0]
    ty = key[1]

    if ty == "non":
        gt = ms_dict_non[word]
        if len(gt) != len(ms):
            return 0
        for i in range(len(ms)):
            if gt[i] != ms[i]:
                return 0
        return 1

    else:
        gt = ms_dict_auto[word]
        if len(gt) != len(ms):
            return 0
        for i in range(len(ms)):
            if i == 0 and gt[i] != ms[i]:
                return 0

            if abs(gt[i] - ms[i]) > 1:
                return 0
        return 1






def grab_words(count : int) -> list[str]:
    words = []
    with open("local/dictionaries/enwiki-20210820-words-frequency-backup.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            w = line.split(" ")[0]

            skip = 0
            for c in w:
                if c not in string.ascii_letters:
                    skip = 1
            if skip == 1:
                continue

            words.append(line.split(" ")[0])
            count -= 1
            if count <= 0:
                break
        f.close()

    return words


def combine_tops(list_1 : list[str], list_2 : list[str]):
    char_dict = {}

    for i in range(len(list_1)):
        if list_1[i] not in char_dict:
            char_dict[list_1[i]] = len(list_1) - i
        else:
            char_dict[list_1[i]] += len(list_1) - i
    
    for i in range(len(list_2)):
        if list_2[i] not in char_dict:
            char_dict[list_2[i]] = len(list_2) - i
        else:
            char_dict[list_2[i]] += len(list_2) - i

    char_list = []
    for c in char_dict:
        char_list.append((c, char_dict[c]))

    char_list.sort(key=(lambda x: x[1]), reverse=True)
    suggestions = []
    for i in range(4):
        if i < len(char_list):
            suggestions.append(char_list[i][0])
        else: 
            break

    return suggestions








def get_autos_weighted_2(dict, prefix : str) -> list[str]:
    # to do

    # for now, uses single suggestions

    if len(prefix) == 1:
        single_suggestions = read_json('../graphs/autocomplete.json')
        return single_suggestions[prefix[len(prefix)-1:]]

    char_dict = {}
    for key in dict:
        if key.startswith(prefix) and key != prefix:
            c = key[len(prefix)]
            if c not in char_dict:
                if len(key) == len(prefix) + 1:
                    char_dict[c] = dict[key] * 3
                else:
                    char_dict[c] = dict[key]
            else:
                if len(key) == len(prefix) + 1:
                    if dict[key] * 10 > char_dict[c]:
                        char_dict[c] = dict[key] * 3
                else:
                    if dict[key] > char_dict[c]:
                        char_dict[c] = dict[key]

    
    char_list = []
    for c in char_dict:
        char_list.append((c, char_dict[c]))
    char_list.sort(key=(lambda x: x[1]), reverse=True)

    suggestions = []
    for i in range(4):
        if i < len(char_list):
            suggestions.append(char_list[i][0])
        else: 
            break
    exp = []
    for i in range(8):
        if i < len(char_list):
            exp.append(char_list[i][0])
        else: 
            break

    #print(prefix + " (2", end="); ")
    #print(exp)
    return suggestions


def get_autos_weighted(dict, prefix : str) -> list[str]:
    # to do

    # for now, uses single suggestions

    if len(prefix) == 1:
        single_suggestions = read_json('../graphs/autocomplete.json')
        return single_suggestions[prefix[len(prefix)-1:]]

    char_dict = {}
    for key in dict:
        if key.startswith(prefix) and key != prefix:
            c = key[len(prefix)]
            if c not in char_dict:
                char_dict[c] = dict[key]
            else:
                if dict[key] > char_dict[c]:
                    char_dict[c] = dict[key]
    
    char_list = []
    for c in char_dict:
        char_list.append((c, char_dict[c]))
    char_list.sort(key=(lambda x: x[1]), reverse=True)

    suggestions = []
    for i in range(4):
        if i < len(char_list):
            suggestions.append(char_list[i][0])
        else: 
            break
    exp = []
    for i in range(8):
        if i < len(char_list):
            exp.append(char_list[i][0])
        else: 
            break

    #print(prefix + " (2", end="); ")
    #print(exp)
    return suggestions









def get_autos(dict, prefix : str, smooth : bool) -> list[str]:
    # to do

    # for now, uses single suggestions

    if len(prefix) == 1:
        single_suggestions = read_json('../graphs/autocomplete.json')
        return single_suggestions[prefix[len(prefix)-1:]]


    char_dict = dict.get_letter_counts(prefix, None, should_smooth=smooth)
    char_list = []
    for key in char_dict:
        char_list.append((key, char_dict[key]))
    char_list.sort(key=(lambda x: x[1]), reverse=True)

    suggestions = []
    for i in range(4):
        if i < len(char_list):
            suggestions.append(char_list[i][0])
        else: 
            break
    exp = []
    for i in range(8):
        if i < len(char_list):
            exp.append(char_list[i][0])
        else: 
            break

    if smooth:
        strat = 1
    else:
        strat = 0
    #print(prefix + " (" + str(strat), end="); ")
    #print(exp)
    return suggestions




def find_path_auto(dict, wcs, word : str, auto_strategy : int, errmsg : bool = False) -> list[float]:
    path = []
    f = open('../graphs/samsung/samsung_keyboard.csv')
    active = list(csv.reader(f))
    f.close()
    f = open('../graphs/samsung/samsung_keyboard_special_1.csv')
    inactive = list(csv.reader(f))
    f.close()
    prev = 'q'
    last_auto = 0


    for i in list(word.lower()):

        if len(path) > 0:

            # get autos using strategy
            if auto_strategy == 0:
                autos = get_autos(dict, word[:len(path)], False)
            elif auto_strategy == 1:
                autos = get_autos(dict, word[:len(path)], True)
            elif auto_strategy == 2:
                autos = get_autos_weighted(wcs, word[:len(path)])
            elif auto_strategy == 3:
                autos = get_autos_weighted_2(wcs, word[:len(path)])
            elif auto_strategy == 4:
                autos_1 = get_autos(dict, word[:len(path)], False)
                autos_2 = get_autos_weighted(wcs, word[:len(path)])
                autos = combine_tops(autos_1, autos_2)
            
            if errmsg:
                print("strategy " + str(auto_strategy) + "; " + word[:len(path)], end=": ")
                print(autos)

            # Ensure that autos is not empty
            if autos != []: #temporary fix
                autos.append("\t")

            if i == autos[0]:
                if last_auto == 1:
                    path.append(0.0)
                else:
                    path.append(1.0)
                last_auto = 1
            elif i in autos:
                path.append(1.0)

                last_auto = 1

            else:
                if i in active[0]:
                    prev_index = active[0].index(prev)
                    cur_index = active[0].index(i)
                    path.append(float(active[prev_index+1][cur_index]) + 1)
                else:
                    prev_index = active[0].index(prev)
                    cur_index = active[0].index('<CHANGE>')
                    path.append(float(active[prev_index+1][cur_index]) + 1)
                    cur_index = inactive[0].index(i)
                    path.append(float(inactive[inactive[0].index('<CHANGE>')+1][cur_index]))
                    temp = inactive
                    inactive = active
                    active = temp
                prev = i
                last_auto = 0

        else: # first move
            if i in active[0]:
                prev_index = active[0].index(prev)
                cur_index = active[0].index(i)
                path.append(float(active[prev_index+1][cur_index]))
            else:
                prev_index = active[0].index(prev)
                cur_index = active[0].index('<CHANGE>')
                path.append(float(active[prev_index+1][cur_index]))
                cur_index = inactive[0].index(i)
                path.append(float(inactive[inactive[0].index('<CHANGE>')+1][cur_index]))
                temp = inactive
                inactive = active
                active = temp
            prev = i
            
    return path
	



def simulate_ms(dict, wcs, word : str, auto : bool, auto_strategy : int, errmsg : bool = False) -> list[int]:
    if not auto:
        ms_string = findPath(word, 0, False)
    else:
        ms_string = find_path_auto(dict, wcs, word, auto_strategy, errmsg)

    ms = []
    for x in ms_string:
        ms.append(int(x))
    return ms


if __name__ == '__main__':
    test = 2

    ms_dict_non = build_ms_dict("non")
    ms_dict_auto = build_ms_dict("auto")
    
    englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")
    wcs = buildDict(100)

    if test == 2:

        correct0 = 0
        correct2 = 0
        correct3 = 0
        total = 0

        fail_list = []

        for key in ms_dict_auto:
            k = (key, "auto")
            if len(key) < 10:
                sim_ms_0 = simulate_ms(englishDictionary, wcs, key, True, 0)
                sim_ms_2 = simulate_ms(englishDictionary, wcs, key, True, 2)
                sim_ms_3 = simulate_ms(englishDictionary, wcs, key, True, 3)

                # test strategy 0
                if eval_ms(k, sim_ms_0) == 0:
                    fail_list.append(key)
                    print("0 failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim 0: ", end="")
                    print(sim_ms_0)
                else:
                    #print("success on " + key)
                    correct0 += 1

                # test strategy 2
                if eval_ms(k, sim_ms_2) == 0:
                    fail_list.append(key)
                    print("2 failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim 2: ", end="")
                    print(sim_ms_2)
                else:
                    #print("success on " + key)
                    correct2 += 1

                # test strategy 3
                if eval_ms(k, sim_ms_3) == 0:
                    fail_list.append(key)
                    print("3 failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim 3: ", end="")
                    print(sim_ms_3)
                else:
                    #print("success on " + key)
                    correct3 += 1
                total += 1
            
            
        print("correct 0: " + str(correct0))
        print("correct 2: " + str(correct2))
        print("correct 3: " + str(correct3))
        print("total: " + str(total))


        # run through fails
        fail_list = list(set(fail_list))
        for key in fail_list:
            print(key)
            sim_ms_0 = simulate_ms(englishDictionary, wcs, key, True, 0, True)
            sim_ms_2 = simulate_ms(englishDictionary, wcs, key, True, 2, True)
            sim_ms_3 = simulate_ms(englishDictionary, wcs, key, True, 3, True)
            print(ms_dict_auto[key])
            print(sim_ms_0)
            print(sim_ms_2)
            print(sim_ms_3)
            print("\n")

        
        print("correct 0: " + str(correct0))
        print("correct 2: " + str(correct2))
        print("correct 3: " + str(correct3))
        print("total: " + str(total))

        # maybe prioritize shorter words?  If it can complete it immediately, try that?




    if test == 0:
        model = build_model()
        correct = (0, 0)
        total = 0

        print("model built")

        words = grab_words(1000)

        for word in words:

            sim_ms_non = simulate_ms(englishDictionary, wcs, word, False, 0)
            if classify_ms(model, sim_ms_non) == 1:
                print("fail to classify simulated non " + word)
                print(sim_ms_non)
            else:
                correct = (correct[0]+1, correct[1])
            sim_ms_auto = simulate_ms(englishDictionary, wcs, word, True, 0)
            if classify_ms(model, sim_ms_auto) == 0:
                print("fail to classify simulated auto " + word)
                print(sim_ms_auto)
            else:
                correct = (correct[0], correct[1] + 1)
            total += 1
        
        print("correct nons: " + str(correct[0]))
        print("correct autos: " + str(correct[1]))
        print("total: " + str(total))

    if test == 1:
        # test non words
        for key in ms_dict_non:
            sim_ms = simulate_ms(englishDictionary, wcs, key, False, 0)
            if sim_ms != ms_dict_non[key]:
                print("failed sim on: " + key)
                print("gt: ", end = "")
                print(ms_dict_non[key])
                print("sim: ", end="")
                print(sim_ms)


        # test auto words
        for key in ms_dict_auto:
            if len(key) == 5:
                sim_ms = simulate_ms(englishDictionary, wcs, key, True, 0)
                if sim_ms != ms_dict_auto[key]:
                    print("failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim: ", end="")
                    print(sim_ms)
                else:
                    print("success on " + key)