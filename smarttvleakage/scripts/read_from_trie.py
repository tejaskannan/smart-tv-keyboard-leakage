from argparse import ArgumentParser
from smarttvleakage.dictionary.dictionaries import restore_dictionary
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.transformations import filter_and_normalize_scores
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--length', type=int)
    args = parser.parse_args()

    dictionary = restore_dictionary(path=args.dictionary_path)
    
    graph = MultiKeyboardGraph(keyboard_type=KeyboardType.SAMSUNG)
    dictionary.set_characters(graph.get_characters())

    counts = dictionary.get_letter_counts(prefix=args.prefix, length=args.length)
    print(counts)

    #filtered_probs = filter_and_normalize_scores(key_counts=counts,
    #                                             candidate_keys=['*', '-', '.', '6', '7', '<COM>', '<DELETEALL>', '?', '@', 'i', 'k', 'l', 'u'],
    #                                             current_string=args.prefix,
    #                                             dictionary=dictionary)
    #print(filtered_probs)

    #print(list(dictionary.get_words_for(prefixes=[args.prefix], max_num_results=10, min_length=None, max_count_per_prefix=None)))

    


