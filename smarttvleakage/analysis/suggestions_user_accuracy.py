import os.path
from argparse import ArgumentParser

from smarttvleakage.utils.file_utils import read_json, iterate_dir


PASSWORDS_PATH = 'samsung_passwords.json'
SEARCHES_PATH = 'web_searches.json'


if __name__ == '__main__':
    parser = ArgumentParser('Script to find the accuracy of keyboard type classification on Samsung devices.')
    parser.add_argument('--user-folder', type=str, required=True, help='Path to the folder containing the user results.')
    args = parser.parse_args()

    password_correct, password_total = 0, 0
    searches_correct, searches_total = 0, 0

    for subject_folder in iterate_dir(args.user_folder):
        password_move_sequences = read_json(os.path.join(subject_folder, PASSWORDS_PATH))
        password_suggestions_types = password_move_sequences['suggestions_types']
        
        password_correct += sum(int(t == 'standard') for t in password_suggestions_types)
        password_total += len(password_suggestions_types)

        searches_path = os.path.join(subject_folder, SEARCHES_PATH)
        if os.path.exists(searches_path):
            searches_move_sequences = read_json(searches_path)
            searches_suggestions_types = searches_move_sequences['suggestions_types']

            searches_correct += sum(int(t == 'suggestions') for t in searches_suggestions_types)
            searches_total += len(searches_suggestions_types)

    print('Password Accuracy: {:.4f}% ({} / {})'.format(100.0 * (password_correct / password_total), password_correct, password_total))

    if searches_total > 0:
        print('Searches Accuracy: {:.4f}% ({} / {})'.format(100.0 * (searches_correct / searches_total), searches_correct, searches_total))
