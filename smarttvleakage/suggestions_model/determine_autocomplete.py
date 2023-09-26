import numpy as np
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Tuple, Optional

from smarttvleakage.audio.move_extractor import Move
from smarttvleakage.dictionary.english_dictionary import SQLEnglishDictionary
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.suggestions_model.simulate_ms import simulate_move_sequence, add_mistakes
from smarttvleakage.suggestions_model.utils import read_passwords, read_english_words
from smarttvleakage.utils.file_utils import read_json, save_pickle_gz
from smarttvleakage.utils.constants import KeyboardType


def moves_to_histogram(move_counts: List[int], bins: List[int]) -> List[int]:
    """
    Turns move list into histogram dictionary given bins and weighting
    """
    histogram = [0 for _ in bins]

    for move_idx, move in enumerate(move_counts):
        if move_idx == 0:
            continue  # Skip the first move, as it never contains suggestions

        if move in bins:
            histogram[move] += move_idx
        else:
            histogram[bins[len(bins) - 1]] += move_idx

    return histogram


def remap_histogram(histogram: List[int], mapping: List[int]) -> List[int]:
    """
    Remaps the histogram using the binning strategy (e.g., grouping 0 and 1 moves together
    will aggregate their individual weights)
    """
    assert len(histogram) == len(mapping), 'Must provide the same number of histogram bins and map bins'

    new_num_bins = max(mapping) + 1
    result = [0 for _ in range(new_num_bins)]

    for bin_idx, count in enumerate(histogram):
        result[mapping[bin_idx]] += count

    return result


def make_features(move_counts: List[int]) -> List[int]:
    """
    Creates a feature vector for the given list of move counts
    """
    bins = list(range(10))  # [0, 1, ..., 8, 9+]
    remapping = [0, 1, 2, 3, 4, 5, 6, 6, 6, 6]

    histogram = moves_to_histogram(move_counts, bins=bins)
    features = remap_histogram(histogram=histogram, mapping=remapping)

    return features


def make_dataset(english_words: List[str],
                 passwords: List[str],
                 english_dictionary: SQLEnglishDictionary,
                 single_suggestions: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a dataset for suggestions classification using the given set of english words and passwords
    """
    keyboard = MultiKeyboardGraph(KeyboardType.SAMSUNG)

    input_list: List[np.ndarray] = []
    label_list: List[int] = []

    for word in english_words:
        # Simulate moves with and without autocomplete
        for label in range(2):
            move_counts = simulate_move_sequence(word=word,
                                                 single_suggestions=single_suggestions,
                                                 use_suggestions=(label == 1),
                                                 english_dictionary=english_dictionary,
                                                 keyboard=keyboard,
                                                 num_mistakes=0)
            
            input_list.append(np.expand_dims(make_features(move_counts), axis=0))
            label_list.append(label)  # 1 for using suggestions

    for word in passwords:
         move_counts = simulate_move_sequence(word=word,
                                              single_suggestions=single_suggestions,
                                              use_suggestions=False,
                                              english_dictionary=english_dictionary,
                                              keyboard=keyboard,
                                              num_mistakes=0)

         input_list.append(np.expand_dims(make_features(move_counts), axis=0))
         label_list.append(0)  # 1 for using suggestions

    inputs = np.vstack(input_list)
    labels = np.vstack(label_list).reshape(-1)

    return inputs, labels


def train(X_train: np.ndarray, y_train: np.ndarray, max_depth: Optional[int]) -> RandomForestClassifier:
    """
    Trains a random forest on the given dataset
    """
    # Make the classifier model
    model = RandomForestClassifier(max_depth=max_depth)

    # Train the model
    model.fit(X_train, y_train)
    return model


def save_model(path: str, model: RandomForestClassifier):
    """Saves a model"""
    save_pickle_gz(model, path)


#################### CLASSIFY #########################
def classify_moves(model: RandomForestClassifier, moves: List[Move], cutoff: float):
    """
    Determines whether the given move count sequence uses a keyboard with suggestions
    """
    # Get only the move counts
    move_counts = [move.num_moves for move in moves]
  
    # Make the feature vector
    features = make_features(move_counts)
    features = np.expand_dims(features, axis=0)  # [1, K]

    # Predict from the dataframe
    pred_probas = model.predict_proba(features)[0]
    suggestions_prob = pred_probas[1]  # predicted probability this is a suggestions keyboard

    return int(suggestions_prob >= cutoff)  # 1 if suggestions, 0 if passwords


if __name__ == "__main__":
    parser = ArgumentParser('Script to train the keyboard suggestions model.')
    parser.add_argument('--english-words-file', type=str, required=True, help='Path to the list of english words to train on, sorted by frequency (descending).')
    parser.add_argument('--passwords-file', type=str, required=True, help='Path to the list of passwords to train on, sorted by frequency (descending).')
    parser.add_argument('--dictionary-file', type=str, required=True, help='Path to the English dictionary from which to estimate letter suggestions.')
    parser.add_argument('--output-file', type=str, required=True, help='Path at which to save the output model (pkl.gz).')
    parser.add_argument('--word-count', type=int, default=5000, help='The number of words from each list to use during training. Defaults to 5,000.')
    args = parser.parse_args()

    assert args.dictionary_file.endswith('.db'), 'Must provide a local SQL database containing the dictionary'
    assert args.word_count >= 1, 'Must provide a positive number of words to consider.'

    # Read in the English words and Passwords
    english_words = read_english_words(args.english_words_file, count=args.word_count)
    passwords = read_passwords(args.passwords_file, count=args.word_count)

    # Read in the initial suggestions (hard-coded)
    single_suggestions = read_json('single_suggestions.json')

    # Load the English dictionary
    english_dictionary = SQLEnglishDictionary(args.dictionary_file)

    # Make the dataset and train the model
    X_train, y_train = make_dataset(english_words=english_words,
                                    passwords=passwords,
                                    english_dictionary=english_dictionary,
                                    single_suggestions=single_suggestions)

    model = train(X_train=X_train, y_train=y_train, max_depth=None)

    train_accuracy = model.score(X_train, y_train)
    print('Train Accuracy: {:.4f}%'.format(train_accuracy * 100.0))

    save_model(model=model, path=args.output_file)
