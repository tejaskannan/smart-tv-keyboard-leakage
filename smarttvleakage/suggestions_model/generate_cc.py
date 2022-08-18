import random

from os import walk

from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.dictionary import EnglishDictionary

def fill_zeroes(n : int, l : int) -> str:
    """Returns n as a string with l digits filled out"""
    if len(str(n)) >= l:
        return str(n)
    n_str = ""
    for i in range((l - len(str(n)))):
        n_str += "0"
    return n_str + str(n)

def random_digits(n : int) -> list[int]:
    """returns n random digits as a list"""
    num = 0
    for i in range(n):
        newDigit = random.randint(0, 9)
        num = num * 10
        num += newDigit
    return num

# CREDIT CARDS
# Could improve this
def checksum(num : int) -> int:
    """Calculates the mod 10 checksum of a number"""
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
    """Validates a cc via checksum"""
    return checksum(num) == 0


def finish_cc(num : int) -> int:
    """Finishes a cc with valid checksum digit"""
    cs = checksum(num)
    #print("cs: " + str(cs))
    return num*10 + ((10-cs) % 10)

# Ineffective
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


def build_valid_cc_list(base_path):
    path = "suggestions_model/ccs"
    valids = []

    for root, dirs, files in walk(path):
        for name in files:
            if name.startswith(base_path):
                full_path = path + "/" + name
                num_start = base_path.split("000", 1)[1]
                num_start = num_start[:3]

                for i in read_jsonl_gz(full_path):
                    num_end = i["cc"]
                    valids.append(num_start + num_end)
    
    #for valid in valids:
        #print(valid)
    return valids


def build_valid_cc_dict(save_path : str):
    valids = build_valid_cc_list("cc_bin_dict000")
    lines = []
    for valid in valids:
        lines.append(str(valid) + " 1000")

    with open(save_path, "w") as f:
        f.writelines(lines)

    cc_dictionary  = EnglishDictionary(50)
    cc_dictionary.build(save_path, 50, True)
    return cc_dictionary



# DATES
def generate_date() -> str:
    """Generates a random valid exp date"""
    mon = str(random.randint(1, 12))
    if len(mon) == 1:
        mon = "0" + mon
    yr = str(random.randint(22, 39))
    return mon + yr 

# Sec CODES
def generate_sec() -> str:
    """Generates a random valid Sec Code"""
    if random.randint(0, 3) == 3:
        sec = str(random.randint(0, 9999))
        return fill_zeroes(sec, 4)
    sec = str(random.randint(0, 999))
    return fill_zeroes(sec, 3)


if __name__ == '__main__':

    test = 0

    if test == 0:
        for i in range(10):
            cc = random_digits(16)
            valid = validate_cc(cc)
            print_cc(cc)
            print(valid)

        for i in range(10):
            cc = generate_cc("visa")
            valid = validate_cc(cc)
            print_cc(cc)
            print(valid)

