from argparse import ArgumentParser
from io import TextIOWrapper
from typing import Set

WORD_LIST = ['function84', 'naarf666', 'p5ych0#7', 'chevy_1954', 'pva81-ph', '.sagara.', '8b7ce7df', 'tutuphpbb', 'bubba?51879', 'williame', 'turbofrogs', 'I.stalky', 'deanna69!', 'W00d!ot2', 'lindafred', 'red.st4r', 'never.di', 'b6j1opz8', 'sol21SOL', '23273500', 'erik4125', 'pretender_now', 'WqkIBI60', 'leninternet', 'mzalpomzaps', '*maiden*', 'dbix2000', 'bruno=never', 'cocoa&almond', '$3vereWx', '20340437', 'qwertz41', 'Paulina-111', 'martin65', 'lad12si.', 'ar#$ma76', 'karina18-', 'aaa*a123', 'everyday', 'torquemada', 'gingerstaind', '9337clns', 'patrick_burdon', 'immortal-phpbb', 'LegoLand', 'ilove$$$', 'U_7Qjgbj', '999999999', 'goblins2', 'davey_becks']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-file', type=str, required=True)
    args = parser.parse_args()

    found: Set[str] = set()

    with open(args.dictionary_file, 'rb') as fin:
        io_wrapper = TextIOWrapper(fin, encoding='utf-8', errors='ignore')
        for line_idx, line in enumerate(io_wrapper):

            # Remove leading and trailing whitespace
            word = line.strip()
            if len(word) <= 0:
                continue

            if word in WORD_LIST:
                found.add(word)

    print('Num Found: {} / {}'.format(len(found), len(WORD_LIST)))
    print(found)
