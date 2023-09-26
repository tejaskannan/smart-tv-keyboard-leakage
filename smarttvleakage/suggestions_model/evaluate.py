import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from typing import Dict, List

from smarttvleakage.dictionary.english_dictionary import SQLEnglishDictionary
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.suggestions_model.determine_autocomplete import make_features
from smarttvleakage.suggestions_model.utils import read_passwords, read_english_words_random
from smarttvleakage.utils.constants import SUGGESTIONS_CUTOFF
from smarttvleakage.utils.file_utils import read_pickle_gz, read_json


def make_test_dataset(move_sequences: Dict[str, List[int]]) -> np.ndarray:
    input_list: List[np.ndarray] = []

    for move_counts in move_sequences.values():
        features = make_features(move_counts)
        input_list.append(np.expand_dims(features, axis=0))

    return np.vstack(input_list)


if __name__ == '__main__':
    parser = ArgumentParser('Script to test the suggestions model.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model pkl.gz file')
    args = parser.parse_args()

    # Restore the saved model
    model = read_pickle_gz(args.model_path)

    # Read in the training and testing sets
    suggestions_move_sequences = read_pickle_gz('move_sequences_suggestions.pkl.gz')
    suggestions_features = make_test_dataset(suggestions_move_sequences)
    suggestions_labels = np.ones(shape=(suggestions_features.shape[0], ))

    standard_move_sequences = read_pickle_gz('move_sequences_standard.pkl.gz')
    standard_features = make_test_dataset(standard_move_sequences)
    standard_labels = np.zeros(shape=(standard_features.shape[0], ))

    inputs = np.concatenate([suggestions_features, standard_features], axis=0)
    labels = np.concatenate([suggestions_labels, standard_labels], axis=0)

    probs = model.predict_proba(inputs)[:, 1]  # Probability of suggestions for each instance
    preds = (probs > SUGGESTIONS_CUTOFF).astype(int)

    accuracy = accuracy_score(y_pred=preds, y_true=labels)

    print('Test Accuracy: {:.4f}%'.format(accuracy * 100.0))
