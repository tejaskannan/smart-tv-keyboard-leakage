import time
from argparse import ArgumentParser

from smarttvleakage.dictionary.dictionaries import restore_dictionary
from smarttvleakage.utils.constants import KeyboardType
from smarttvleakage.utils.transformations import filter_and_normalize_scores
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--length', type=int, required=True)
    args = parser.parse_args()

    dictionary = restore_dictionary(path=args.dictionary_path)
    
    graph = MultiKeyboardGraph(keyboard_type=KeyboardType.APPLE_TV_PASSWORD)
    dictionary.set_characters(graph.get_characters())

    #start = time.time()
    #projected = dictionary.projected_remaining_log_prob(args.prefix, length=args.length)
    #end = time.time()

    #print('Projected: {:.5f}, Elapsed: {:.5f}sec'.format(projected, end - start))

    score = dictionary.get_score_for_string(args.prefix, length=args.length)
    print('String Score: {:.5f}'.format(score))

    counts = dictionary.get_letter_counts(prefix=args.prefix, length=args.length)
    print(counts)
