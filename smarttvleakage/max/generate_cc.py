import random


def random_digits(n):
    num = 0
    for i in range(n):
        newDigit = random.randint(0, 9)
        num = num * 10
        num += newDigit
    return num


def checksum(num):
    num_str = str(num)
    total = 0
    for d in map(int, num_str):
        next = 0
        if d % 2 == 1:
            d = d * 2
            if d > 9:
                for i in map(int, str(d)):
                    next += i
            else:
                next = d
        else:
            next = d
        total += next
    print("total: " + str(total))
    return total % 10

def validate_cc(num):
    return checksum(num) == 0



# doesn't quite work!!
def finish_cc(num):
    cs = checksum(num)
    print("cs: " + str(cs))
    
    if cs == 0:
        last = 0
    
    elif cs % 2 == 0:
        last = 10-cs
    
    elif cs == 1:
        last = 9
    elif cs == 5:
        last = 7
    elif cs == 9:
        last = 5

    elif cs == 7:
        if num % 10 == 0:
            return finish_cc(num+1)
        else:
            return finish_cc(num-1)
    elif cs == 3:
        if num % 10 == 0:
            return finish_cc(num+1)
        else:
            return finish_cc(num-1)
    

    return num*10 + last



def generate_bin(ty):
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

    end = random_digits(6 - len(str(start)))
    return start * (pow(10, (6 - len(str(start))))) + end
    

def add_acc(bin):
    start = bin * (pow(10, 9))
    return start + random_digits(9)


def generate_cc(ty):
    bin = generate_bin(ty)
    cc = add_acc(bin)
    return finish_cc(cc)

def print_cc(cc):
    for i in range(4):
        for j in range(4):
            print(str(cc)[i+j], end="")
        print(" ", end="")
    print("\n", end="")


if __name__ == '__main__':

    for i in range(10):
        cc = random_digits(16)
        valid = validate_cc(cc)
        print(cc)
        print(valid)

    print("\n\n\n")

    for i in range(10):
        cc = generate_cc("visa")
        valid = validate_cc(cc)
        print_cc(cc)
        print(valid)
