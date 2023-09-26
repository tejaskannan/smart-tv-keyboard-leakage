import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from smarttvleakage.dictionary.english_dictionary import SQLEnglishDictionary
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.suggestions_model.determine_autocomplete import make_dataset
from smarttvleakage.suggestions_model.utils import read_passwords, read_english_words_random
from smarttvleakage.utils.constants import SUGGESTIONS_CUTOFF
from smarttvleakage.utils.file_utils import read_pickle_gz, read_json


if __name__ == '__main__':
    parser = ArgumentParser('Script to test the suggestions model.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model pkl.gz file')
    parser.add_argument('--english-words-file', type=str, required=True, help='Path to a list of English words to use.')
    parser.add_argument('--passwords-file', type=str, required=True, help='Path to a list of passwords to use.')
    parser.add_argument('--dictionary-file', type=str, required=True, help='Path to the English dictionary used to simulate suggestions.')
    parser.add_argument('--word-count', type=int, default=5000, help='The number of words from each list to use. Defaults to 5,000')
    args = parser.parse_args()

    # Restore the saved model
    model = read_pickle_gz(args.model_path)

    # Read in the English words and Passwords
    rand = np.random.RandomState(seed=589)
    english_words = read_english_words_random(args.english_words_file, count=args.word_count, rand=rand)
    passwords = read_passwords(args.passwords_file, count=args.word_count)

    print(len(english_words))
    print(len(passwords))

    print(english_words)
    print(passwords)

    # Read in the initial suggestions (hard-coded)
    single_suggestions = read_json('single_suggestions.json')

    # Load the English dictionary
    english_dictionary = SQLEnglishDictionary(args.dictionary_file)

    # Make the dataset and train the model
    X_test, y_test = make_dataset(english_words=english_words,
                                  passwords=passwords,
                                  english_dictionary=english_dictionary,
                                  single_suggestions=single_suggestions)

    probs = model.predict_proba(X_test)[:, 1]  # Probability of suggestions for each instance
    preds = (probs > SUGGESTIONS_CUTOFF).astype(int)

    accuracy = accuracy_score(y_pred=preds, y_true=y_test)

    print('Test Accuracy: {:.4f}%'.format(accuracy * 100.0))
