
import csv
import os.path
import string


from smarttvleakage.keyboard_utils.word_to_move import findPath

from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION
from smarttvleakage.utils.file_utils import read_json

from smarttvleakage.audio.determine_autocomplete import build_model, classify_ms






# Non-autocomplete Dict
ms_dict_non = {}

# Dec of Ind
ms_dict_non["we"] = [1, 1]
ms_dict_non["hold"] = [6, 4, 1, 6]
ms_dict_non["these"] = [4, 2, 4, 2, 2]
ms_dict_non["truths"] = [4, 1, 3, 2, 2, 4]
ms_dict_non["to"] = [4, 4]
ms_dict_non["be"] = [6, 4]
ms_dict_non["self"] = [2, 2, 7, 5]
ms_dict_non["evident"] = [2, 3, 6, 6, 1, 5, 3]
ms_dict_non["that"] = [4, 2, 5, 5]
ms_dict_non["all"] = [1, 8, 0]
ms_dict_non["men"] = [8, 6, 5]
ms_dict_non["are"] = [1, 4, 1]
ms_dict_non["created"] = [4, 3, 1, 3, 5, 2, 1]
ms_dict_non["equal"] = [2, 2, 6, 7, 8]
ms_dict_non["they"] = [4, 2, 4, 3]
# ms_dict_non["endowed"] = [2, 5, 4, 7, 7, 1, 1]
ms_dict_non["by"] = [6, 3]
ms_dict_non["their"] = [4, 2, 4, 5, 4]
ms_dict_non["creator"] = [4, 3, 1, 3, 5, 4, 5]
ms_dict_non["with"] = [1, 6, 3, 2]
ms_dict_non["certain"] = [4, 2, 1, 1, 5, 8, 4]
ms_dict_non["unalienable"] = [6, 3, 6, 8, 2, 5, 5, 6, 5, 5, 7]
ms_dict_non["rights"] = [3, 4, 4, 1, 2, 4]
ms_dict_non["life"] = [9, 2, 5, 2]
ms_dict_non["liberty"] = [9, 2, 5, 4, 1, 1, 1]
ms_dict_non["and"] = [1, 6, 4]
# ms_dict_non["the"] = [4, 2, 4]
ms_dict_non["pursuit"] = [9, 3, 3, 3, 6, 1, 3]
ms_dict_non["of"] = [8, 6]
ms_dict_non["happiness"] = [6, 5, 10, 0, 2, 4, 5, 2, 0]
ms_dict_non["secure"] = [2, 2, 2, 6, 3, 1]
ms_dict_non["governments"] = [5, 5, 7, 3, 1, 4, 1, 6, 5, 3, 4]
ms_dict_non["instituted"] = [7, 4, 5, 4, 3, 3, 2, 2, 2, 1]
ms_dict_non["among"] = [1, 7, 4, 5, 2]
ms_dict_non["deriving"] = [3, 1, 1, 4, 6, 6, 4, 2]
ms_dict_non["just"] = [7, 1, 6, 4]
ms_dict_non["powers"] = [9, 1, 7, 1, 1, 3]
ms_dict_non["from"] = [4, 1, 5, 4]
ms_dict_non["consent"] = [4, 8, 5, 5, 2, 5, 3]
ms_dict_non["governed"] = [5, 5, 7, 3, 1, 4, 5, 1]
ms_dict_non["whenever"] = [1, 5, 4, 5, 5, 3, 3, 1]
ms_dict_non["any"] = [1, 6, 2]
ms_dict_non["form"] = [4, 6, 5, 5]
ms_dict_non["becomes"] = [6, 4, 2, 8, 4, 6, 2]
ms_dict_non["destructive"] = [3, 1, 2, 4, 1, 3, 6, 4, 3, 6, 3]
# ms_dict_non["ends"] = [2, 5, 4, 1]
ms_dict_non["it"] = [7, 3]
ms_dict_non["is"] = [7, 7]
ms_dict_non["right"] = [3, 4, 4, 1, 2]
ms_dict_non["people"] = [9, 7, 6, 1, 2, 7]
ms_dict_non["alter"] = [1, 8, 5, 2, 1]
ms_dict_non["or"] = [8, 5]

# Gettys
ms_dict_non["a"] = [1]
ms_dict_non["add"] = [1, 2, 0]
ms_dict_non["ago"] = [1, 4, 5]
ms_dict_non["be"] = [6, 4]
ms_dict_non["brave"] = [6, 3, 4, 4, 3]
ms_dict_non["brought"] = [6, 3, 5, 2, 3, 1, 2]
ms_dict_non["carrot"] = [4, 3, 4, 0, 5, 4]
ms_dict_non["cause"] = [4, 3, 7, 6, 2]
ms_dict_non["civil"] = [4, 7, 6, 6, 2]
ms_dict_non["continent"] = [4, 8, 5, 3, 3, 4, 5, 5, 3]
ms_dict_non["dedicate"] = [3, 1, 1, 6, 7, 3, 5, 2]
ms_dict_non["devotion"] = [3, 1, 3, 7, 4, 3, 1, 5]
ms_dict_non["did"] = [3, 6, 6]
ms_dict_non["do"] = [3, 7]
ms_dict_non["earth"] = [2, 3, 4, 1, 2]
ms_dict_non["endure"] = [2, 5, 4, 5, 3, 1]
ms_dict_non["equal"] = [2, 2, 6, 7, 8]
ms_dict_non["fathers"] = [4, 3, 5, 2, 4, 1, 3]
ms_dict_non["final"] = [4, 5, 4, 6, 8]
ms_dict_non["for"] = [4, 6, 5]
ms_dict_non["forget"] = [4, 6, 5, 2, 3, 2]
ms_dict_non["forth"] = [4, 6, 5, 1, 2]
ms_dict_non["freedom"] = [4, 1, 1, 0, 1, 7, 4]
#p2
ms_dict_non["god"] = [5, 5, 7]
ms_dict_non["ground"] = [5, 2, 5, 2, 3, 4]
ms_dict_non["hallow"] = [6, 5, 8, 0, 1, 7]
ms_dict_non["have"] = [6, 5, 4, 3]
ms_dict_non["here"] = [6, 4, 1, 1]
ms_dict_non["highly"] = [6, 3, 4, 1, 3, 4]
ms_dict_non["in"] = [7, 4]
ms_dict_non["last"] = [9, 8, 1, 4]
ms_dict_non["live"] = [9, 2, 6, 3]
ms_dict_non["lives"] = [9, 2, 6, 3, 2]
ms_dict_non["long"] = [9, 1, 5, 2]
ms_dict_non["measure"] = [8, 6, 3, 1, 6, 3, 1]
ms_dict_non["met"] = [8, 6, 2]
ms_dict_non["nation"] = [7, 6, 5, 3, 1, 5]
ms_dict_non["never"] = [7, 5, 3, 3, 1]
ms_dict_non["new"] = [7, 5, 1]
ms_dict_non["nobly"] = [7, 5, 6, 5, 4]
ms_dict_non["nor"] = [7, 5, 5]
ms_dict_non["note"] = [7, 5, 4, 2]
ms_dict_non["now"] = [7, 5, 7]
ms_dict_non["proper"] = [9, 6, 5, 1, 7, 1]
ms_dict_non["rather"] = [3, 4, 5, 2, 4, 1]
ms_dict_non["remember"] = [3, 1, 6, 6, 6, 2, 4, 1]
ms_dict_non["say"] = [2, 1, 6]
ms_dict_non["sense"] = [2, 2, 5, 5, 2]
ms_dict_non["should"] = [2, 4, 4, 2, 3, 6]
# ms_dict_non["it"] = []
#page 3
ms_dict_non["so"] = [2, 8]
ms_dict_non["take"] = [4, 5, 7, 6]
ms_dict_non["they"] = [4, 2, 4, 3]
ms_dict_non["thus"] = [4, 2, 2, 6]
ms_dict_non["us"] = [6, 6]
ms_dict_non["war"] = [1, 2, 4]
ms_dict_non["whether"] = [1, 5, 4, 2, 2, 4, 1]
ms_dict_non["which"] = [1, 5, 3, 7, 4]
ms_dict_non["work"] = [1, 7, 5, 5]
ms_dict_non["year"] = [5, 3, 3, 4]


# Autocomplete Dict
ms_dict_auto = {}

# Dec of Ind
ms_dict_auto["we"] = [1, 1]
ms_dict_auto["hold"] = [6, 1, 3, 1]
ms_dict_auto["these"] = [4, 1, 0, 6, 1]
ms_dict_auto["truths"] = [4, 2, 1, 2, 0, 2]
ms_dict_auto["to"] = [4, 1]
ms_dict_auto["be"] = [6, 1]
ms_dict_auto["self"] = [2, 1, 8, 1]
ms_dict_auto["evident"] = [2, 1, 2, 0, 0, 0, 2]
ms_dict_auto["that"] = [4, 1, 1, 0]
ms_dict_auto["all"] = [1, 1, 10]
ms_dict_auto["men"] = [8, 1, 2]
ms_dict_auto["are"] = [1, 1, 0]
ms_dict_auto["created"] = [4, 1, 3, 1, 1, 0, 0]
ms_dict_auto["equal"] = [2, 3, 1, 2, 0]
ms_dict_auto["they"] = [4, 1, 0, 0]
# ms_dict_auto["endowed"] = []
ms_dict_auto["by"] = [6, 1]
ms_dict_auto["their"] = [4, 1, 0, 1, 0]
ms_dict_auto["creator"] = [4, 1, 3, 1, 1, 1, 0]
ms_dict_auto["with"] = [1, 1, 0, 0]
ms_dict_auto["certain"] = [4,3, 1, 0, 0, 0, 0]
ms_dict_auto["unalienable"] = [6, 1, 9, 9, 1, 0, 0, 0, 0, 0, 0]
ms_dict_auto["rights"] = [3, 1, 0, 0, 0, 0]
ms_dict_auto["life"] = [9, 1, 1, 0]
ms_dict_auto["liberty"] = [9, 1, 5, 1, 0, 0, 0]
ms_dict_auto["and"] = [1, 1, 0]
# ms_dict_auto["the"] = []
ms_dict_auto["pursuit"] = [9, 1, 8, 1, 1, 0, 1]
ms_dict_auto["of"] = [8, 1]
ms_dict_auto["happiness"] = [6, 1, 2, 6, 1, 0, 0, 0, 10]
ms_dict_auto["secure"] = [2, 1, 0, 1, 0, 1]
ms_dict_auto["governments"] = [5, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0]
ms_dict_auto["instituted"] = [7, 1, 1, 1, 1, 1, 0, 0, 1, 0]
ms_dict_auto["among"] = [1, 8, 1, 0, 0]
ms_dict_auto["deriving"] = [3, 1, 2, 1, 1, 0, 0, 0]
ms_dict_auto["just"] = [7, 1, 0, 0]
ms_dict_auto["powers"] = [9, 1, 2, 0, 0, 1]
ms_dict_auto["from"] = [4, 1, 0, 0]
ms_dict_auto["consent"] = [4, 1, 1, 1, 1, 1, 0]
ms_dict_auto["governed"] = [5, 1, 4, 1, 0, 0, 2, 0]
ms_dict_auto["whenever"] = [1, 1, 2, 0, 0, 0, 0, 0]
ms_dict_auto["any"] = [1, 1, 1]
ms_dict_auto["form"] = [4, 1, 0, 5]
ms_dict_auto["becomes"] = [6, 1, 0, 1, 0, 0, 0]
ms_dict_auto["destructive"] = [3, 1, 3, 1, 0, 1, 0, 0, 0, 1, 0]
# ms_dict_auto["ends"] = []
ms_dict_auto["it"] = [7, 1]
ms_dict_auto["is"] = [7, 1]
ms_dict_auto["right"] = [3, 1, 0, 0, 0]
ms_dict_auto["people"] = [9, 1, 0, 0, 0, 0]
ms_dict_auto["alter"] = [1, 1, 0, 1, 0]
ms_dict_auto["or"] = [8, 1]


# Gettys, p1
ms_dict_auto["a"] = [1]
ms_dict_auto["add"] = [1, 3, 0]
ms_dict_auto["ago"] = [1, 5, 1]
ms_dict_auto["be"] = [6, 1]
ms_dict_auto["brave"] = [6, 4, 1, 1, 0]
ms_dict_auto["brought"] = [6, 6, 4, 1, 1, 0, 0, 0]
ms_dict_auto["carrot"] = [4, 1, 0, 5, 1, 0]
ms_dict_auto["cause"] = [4, 1, 8, 1, 0]
ms_dict_auto["civil"] = [4, 8, 7, 1, 0]
ms_dict_auto["continent"] = [4, 1, 1, 0, 1, 0, 1, 0, 0]
ms_dict_auto["dedicate"] = [3, 1, 1, 1, 0, 0, 0, 0]
ms_dict_auto["devotion"] = [3, 1, 2, 1, 0, 2, 0, 0]
ms_dict_auto["did"] = [3, 1, 0]
ms_dict_auto["do"] = [3, 1]
ms_dict_auto["earth"] = [2, 4, 1, 0, 0]
ms_dict_auto["endure"] = [2, 1, 2, 4, 1, 0]
ms_dict_auto["equal"] = [2, 3, 1, 2, 0]
ms_dict_auto["fathers"] = [4, 4, 1, 0, 0, 0, 2]
ms_dict_auto["final"] = [4, 1, 0, 1, 0]
ms_dict_auto["for"] = [4, 1, 0]
ms_dict_auto["forget"] = [4, 1, 0, 0, 0, 0]
ms_dict_auto["forth"] = [4, 1, 0, 4, 1]
ms_dict_auto["freedom"] = [4, 1, 1, 2, 1, 0, 0]
#p2
ms_dict_auto["god"] = [5, 1, 2]
ms_dict_auto["ground"] = [5, 1, 1, 0, 0, 0]
ms_dict_auto["hallow"] = [6, 1, 5, 0, 2, 1]
ms_dict_auto["have"] = [6, 1, 0, 0]
ms_dict_auto["here"] = [6, 1, 0, 0]
ms_dict_auto["highly"] = [6, 1, 0, 0, 0, 0]
ms_dict_auto["in"] = [7, 1]
ms_dict_auto["last"] = [9, 1, 0, 0]
ms_dict_auto["live"] = [9, 1, 6, 1]
ms_dict_auto["lives"] = [9, 1, 6, 1, 0]
ms_dict_auto["long"] = [9, 1, 1, 0]
ms_dict_auto["measure"] = [8, 1, 0, 6, 1, 0, 1]
ms_dict_auto["met"] = [8, 1, 1]
ms_dict_auto["nation"] = [7, 1, 2, 2, 0, 0]
ms_dict_auto["never"] = [7, 1, 0, 0, 0]
ms_dict_auto["new"] = [7, 1, 1]
ms_dict_auto["nobly"] = [7, 1, 1, 1, 1]
ms_dict_auto["nor"] = [7, 1, 4]
ms_dict_auto["note"] = [7, 1, 1, 1]
ms_dict_auto["now"] = [7, 1, 0]
ms_dict_auto["proper"] = [9, 1, 1, 1, 1, 0]
ms_dict_auto["rather"] = [3, 5, 1, 0, 0, 0]
ms_dict_auto["remember"] = [3, 1, 1, 0, 0, 0, 0, 0]
ms_dict_auto["say"] = [2, 1, 0]
ms_dict_auto["sense"] = [2, 1, 2, 2, 0]
ms_dict_auto["should"] = [2, 1, 1, 0, 0, 0]
# ms_dict_auto["it"] = []
#3
ms_dict_auto["so"] = [2, 1]
ms_dict_auto["take"] = [4, 1, 0, 0]
ms_dict_auto["they"] = [4, 1, 6, 1]
ms_dict_auto["thus"] = [4, 1, 0, 0]
ms_dict_auto["us"] = [6, 1]
ms_dict_auto["war"] = [1, 1, 2]
ms_dict_auto["whether"] = [1, 1, 2, 1, 0, 0, 0]
ms_dict_auto["which"] = [1, 1, 8, 1, 0]
ms_dict_auto["work"] = [1, 8, 1, 0]
ms_dict_auto["year"] = [5, 1, 0, 1]








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





def get_autos(dict, prefix : str) -> list[str]:
    # to do

    # for now, uses single suggestions

    if len(prefix) == 1:
        single_suggestions = read_json('graphs/autocomplete.json')
        return single_suggestions[prefix[len(prefix)-1:]]


    char_dict = dict.get_letter_counts(prefix, None, should_smooth=False)
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
    
    return suggestions




def find_path_auto(dict, word : str) -> list[float]:
    path = []
    f = open('graphs/samsung_keyboard.csv')
    active = list(csv.reader(f))
    f.close()
    f = open('graphs/samsung_keyboard_special_1.csv')
    inactive = list(csv.reader(f))
    f.close()
    prev = 'q'
    last_auto = 0

    for i in list(word.lower()):

        if len(path) > 0:
            autos = get_autos(dict, word[:len(path)])
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
	



def simulate_ms(dict, word : str, auto : bool) -> list[int]:
    if not auto:
        ms_string = findPath(word, 0)
    else:
        ms_string = find_path_auto(dict, word)

    ms = []
    for x in ms_string:
        ms.append(int(x))
    return ms


if __name__ == '__main__':
    test = 0
    
    englishDictionary = EnglishDictionary.restore(path="local/dictionaries/ed.pkl.gz")


    if test == 0:
        model = build_model()
        correct = (0, 0)
        total = 0

        print("model built")

        words = grab_words(1000)

        for word in words:

            sim_ms_non = simulate_ms(englishDictionary, word, False)
            if classify_ms(model, sim_ms_non) == 1:
                print("fail to classify simulated non " + word)
                print(sim_ms_non)
            else:
                correct = (correct[0]+1, correct[1])
            sim_ms_auto = simulate_ms(englishDictionary, word, True)
            if classify_ms(model, sim_ms_auto) == 0:
                print("fail to classify simulated auto " + word)
                print(sim_ms_auto)
            else:
                correct = (correct[0], correct[1] + 1)
            total += 2
        
        print("correct nons: " + str(correct[0]))
        print("correct autos: " + str(correct[1]))
        print("total: " + str(total))

    if test == 1:
        # test non words
        for key in ms_dict_non:
            sim_ms = simulate_ms(englishDictionary, key, False)
            if sim_ms != ms_dict_non[key]:
                print("failed sim on: " + key)
                print("gt: ", end = "")
                print(ms_dict_non[key])
                print("sim: ", end="")
                print(sim_ms)


        # test auto words
        for key in ms_dict_auto:
            if len(key) == 5:
                sim_ms = simulate_ms(englishDictionary, key, True)
                if sim_ms != ms_dict_auto[key]:
                    print("failed sim on: " + key)
                    print("gt: ", end = "")
                    print(ms_dict_auto[key])
                    print("sim: ", end="")
                    print(sim_ms)
                else:
                    print("success on " + key)