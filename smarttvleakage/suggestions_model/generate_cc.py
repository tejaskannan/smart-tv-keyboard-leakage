import random


from smarttvleakage.utils.file_utils import read_json, save_pickle_gz, read_pickle_gz
from smarttvleakage.dictionary import CharacterDictionary, UniformDictionary, EnglishDictionary, NumericDictionary, CreditCardDictionary, CreditCardDictionaryStrong, UNPRINTED_CHARACTERS, CHARACTER_TRANSLATION, SPACE, SELECT_SOUND_KEYS



def random_digits(n : int) -> list[int]:
    num = 0
    for i in range(n):
        newDigit = random.randint(0, 9)
        num = num * 10
        num += newDigit
    return num


def checksum(num : int) -> int:
    num_str = str(num)
    total = 0
    digits = list(map(int, num_str))
    for i in range(len(digits)):
        if i % 2 == 0:
            next = (2 * digits[i]) % 9
        else:
            next = digits[i]
        total += next
        
    return total % 10

def validate_cc(num : int) -> bool:
    return checksum(num) == 0



def finish_cc(num : int) -> int:
    cs = checksum(num)
    #print("cs: " + str(cs))
    
    if cs == 0:
        last = 0
    else:
        last = 10-cs
    

    return num*10 + last



def generate_bin(ty : str = "") -> int:
    if ty == "amex":
        if random.randint(0, 1) == 0:
            start = 34
        else:
            start = 37
    
    elif ty == "visa":
        start = 4
    
    elif ty == "master":
        start = 5

    elif ty == "discover":
        start = 6
    elif ty == "k":
        start = 417500

    else:
        r = random.randint(0, 99)
        if r < 20:
            return generate_bin("amex")
        elif r < 40:
            return generate_bin("visa")
        elif r < 60:
            return generate_bin("master")
        elif r < 80:
            return generate_bin("discover")

        elif r < 90:
            start = random.randint(0, 2)
        else:
            start = random.randint(7, 9)

    end = random_digits(6 - len(str(start)))
    return start * (pow(10, (6 - len(str(start))))) + end
    

def add_acc(bin : int) -> int:
    start = bin * (pow(10, 9))
    return start + random_digits(9)


def generate_cc(ty : str = "") -> int:
    bin = generate_bin(ty)
    cc = add_acc(bin)
    return finish_cc(cc)

def print_cc(cc : str):
    for i in range(4):
        for j in range(4):
            print(str(cc)[i+j], end="")
        print(" ", end="")
    print("\n", end="")




# work on this 
def save_zip_dictionary(path):
    zip_counts = {}

    zip_dicts = read_json(path)
    for zip_dict in zip_dicts:
        zip = str(zip_dict["zip_code"])
        zeroes = 5 - len(zip)
        zip_str = ""
        for i in range(zeroes):
            zip_str += "0"
        zip_str += zip

        if zip_str in zip_counts:
            zip_counts[zip_str] += 1
        else:
            zip_counts[zip_str] = 1
    
    save_path = "local/dictionaries/zip.pkl.gz"
    save_pickle_gz(zip_counts, save_path)
    
    return


# work on this 
def load_zip_dictionary(path):
    return read_pickle_gz(path)
    # turn zip dict into englishDictionary type




# read from zip pkl
# generate random one?
def generate_zip(zip_counts):
    return random.choice(list(zip_counts.keys()))

def validate_zip(zip_counts, zip : int):
    return zip in zip_counts











if __name__ == '__main__':

    test = 3

    if test == 0:
        for i in range(10):
            cc = random_digits(16)
            valid = validate_cc(cc)
            print_cc(cc)
            print(valid)

        print("\n\n\n")

        for i in range(10):
            cc = generate_cc("visa")
            valid = validate_cc(cc)
            print_cc(cc)
            print(valid)


    elif test == 1:
        #save_zip_dictionary("local/zips.json")
        print("zip saved already")

    elif test == 2:
        zip_counts = load_zip_dictionary("local/dictionaries/zip.pkl.gz")
        print(generate_zip(zip_counts))
    elif test == 3:
        zip_counts = load_zip_dictionary("local/dictionaries/zip.pkl.gz")
        for i in range(100):
            test_zip = str(random.randint(0, 99999))
            print(test_zip, end=": ")
            print(validate_zip(zip_counts, test_zip))