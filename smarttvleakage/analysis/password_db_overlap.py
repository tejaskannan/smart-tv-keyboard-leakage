import sqlite3
from argparse import ArgumentParser
from typing import Set

from smarttvleakage.utils.ngrams import create_ngrams, split_ngram


WORD_LIST = ['function84', 'naarf666', 'p5ych0#7', 'chevy_1954', 'pva81-ph', '.sagara.', '8b7ce7df', 'tutuphpbb', 'bubba?51879', 'williame', 'turbofrogs', 'I.stalky', 'deanna69!', 'W00d!ot2', 'lindafred', 'red.st4r', 'never.di', 'b6j1opz8', 'sol21SOL', '23273500', 'erik4125', 'pretender_now', 'WqkIBI60', 'leninternet', 'mzalpomzaps', '*maiden*', 'dbix2000', 'bruno=never', 'cocoa&almond', '$3vereWx', '20340437', 'qwertz41', 'Paulina-111', 'martin65', 'lad12si.', 'ar#$ma76', 'karina18-', 'aaa*a123', 'everyday', 'torquemada', 'gingerstaind', '9337clns', 'patrick_burdon', 'immortal-phpbb', 'LegoLand', 'ilove$$$', 'U_7Qjgbj', '999999999', 'goblins2', 'davey_becks']


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--db-file', type=str, required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()

    found: Set[str] = set()

    for word in WORD_LIST:
        is_present = True

        for ngram in create_ngrams(word, 5):
            prefix, suffix = split_ngram(ngram)
            result = cursor.execute('SELECT next FROM ngrams WHERE prefix = ?;', (prefix, ))
            results = result.fetchall()
            
            did_find = False
            for record in results:
                if (suffix == record[0]):
                    did_find = True
                    break
    
            is_present = is_present and did_find
            if (not is_present):
                break

        if is_present:
            found.add(word)
    
    print('Found {} / {} words via ngrams.'.format(len(found), len(WORD_LIST)))
    print(found)
