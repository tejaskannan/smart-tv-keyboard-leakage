import random
from smarttvleakage.utils.file_utils import read_json, save_pickle_gz, read_pickle_gz

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

# *double check
def finish_cc(num : int) -> int:
    """Finishes a cc with valid checksum digit"""
    cs = checksum(num)
    #print("cs: " + str(cs))
    return num*10 + ((10-cs) % 10)

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





# ZIPS
# work on this 
def save_zip_dictionary(path : str):
    zip_counts = {}

    zip_dicts = read_json(path)
    for zip_dict in zip_dicts:
        zip_str = fill_zeroes(zip_dict["zip_code"], 5)

        if zip_str in zip_counts:
            zip_counts[zip_str] += 1
        else:
            zip_counts[zip_str] = 1
    
    save_path = "local/dictionaries/zip.pkl.gz"
    save_pickle_gz(zip_counts, save_path)
    
    return


# work on this 
def load_zip_dictionary(path : str) -> dict[str, int]:
    return read_pickle_gz(path)
    # turn zip dict into englishDictionary type


# read from zip pkl
# generate random one?
def generate_zip(zip_counts : dict[str, int]) -> str:
    return random.choice(list(zip_counts.keys()))

def validate_zip(zip_counts, zip : int) -> bool:
    return zip in zip_counts




# DATES

def generate_date() -> str:
    mon = str(random.randint(1, 12))
    if len(mon) == 1:
        mon = "0" + mon
    yr = str(random.randint(20, 30))
    return mon + yr 

def validate_date(date : str) -> bool:
    if len(date) != 4:
        return False
    
    if int(date[:2]) > 12:
        return False

    return True





# Sec CODES


def generate_sec() -> str:
    if random.randint(0, 3) == 3:
        sec = str(random.randint(0, 9999))
        return fill_zeroes(sec, 4)
    sec = str(random.randint(0, 999))
    return fill_zeroes(sec, 3)

def validate_sec(date : str) -> bool:
    return len(date) in [3, 4]










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