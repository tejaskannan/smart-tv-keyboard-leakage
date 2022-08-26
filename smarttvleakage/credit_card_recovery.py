import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from argparse import ArgumentParser
from typing import Tuple, List, Dict, Optional

from smarttvleakage.audio import MoveExtractor, make_move_extractor, SmartTVTypeClassifier, Move
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import restore_dictionary, CharacterDictionary
from smarttvleakage.search_numeric import get_digits_from_moves
from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence
from smarttvleakage.utils.file_utils import read_pickle_gz, iterate_dir, read_json


def get_correct_rank(move_seq: List[Move], graph: MultiKeyboardGraph, dictionary: CharacterDictionary, tv_type: SmartTVType, max_num_guesses: int, target: str) -> Optional[int]:
    assert isinstance(target, str), 'Must provide the target as a string.'

    # Get the list of candidates
    ranked_candidates = get_digits_from_moves(move_sequence=move_seq,
                                              graph=graph,
                                              dictionary=dictionary,
                                              tv_type=tv_type,
                                              max_num_results=max_num_guesses)

    for rank, (guess, _, _) in enumerate(ranked_candidates):
        if guess == target:
            return rank + 1

    return None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--zip-code-dict-path', type=str, required=True)
    parser.add_argument('--max-num-results', type=int)
    parser.add_argument('--max-num-videos', type=int)
    args = parser.parse_args()

    # Load the video file paths and labels
    if os.path.isdir(args.video_path):
        video_paths = list(filter(lambda path: not path.endswith('json'), iterate_dir(args.video_path)))
    else:
        video_paths = [args.video_path]

    labels = read_json(args.labels_path)

    # Make the TV Type classifier
    tv_type_clf = SmartTVTypeClassifier()

    # Create the dictionaries
    ccn_dictionary = restore_dictionary('credit_card')
    numeric_dictionary = restore_dictionary('numeric')
    exp_year_dictionary = restore_dictionary('exp_year')
    zip_code_dictionary = restore_dictionary(args.zip_code_dict_path)

    rank_list: List[int] = []
    num_candidates_list: List[int] = []
    rank_dict: Dict[str, int] = dict()
    candidates_dict: Dict[str, int] = dict()
    not_found_list: List[str] = []

    top10_correct = 0
    total_count = 0
    num_not_found = 0

    print('Number of video files: {}'.format(len(video_paths)))

    for idx, video_path in enumerate(video_paths):
        if (args.max_num_videos is not None) and (idx >= args.max_num_videos):
            break

        video_clip = mp.VideoFileClip(video_path)
        audio = video_clip.audio
        signal = audio.to_soundarray()

        file_name = os.path.basename(video_path)
        file_name = file_name.replace('.MOV', '').replace('.mov', '').replace('.mp4', '')

        # Classify the TV type based on the sound profile
        tv_type = tv_type_clf.get_tv_type(audio=signal)

        # Extract the move sequence
        move_extractor = make_move_extractor(tv_type=tv_type)
        move_sequence, did_use_autocomplete, keyboard_type = move_extractor.extract_move_sequence(audio=signal, include_moves_to_done=True)

        # Split the move sequence into the credit card fields (automatically)
        credit_card_seq = extract_credit_card_sequence(move_sequence)
        assert credit_card_seq is not None, 'Could not extract the credit card sequence'

        # Make the graph based on the keyboard type
        graph = MultiKeyboardGraph(keyboard_type=keyboard_type)

        # Set the dictionary characters
        numeric_dictionary.set_characters(graph.get_characters())
        ccn_dictionary.set_characters(graph.get_characters())
        zip_code_dictionary.set_characters(graph.get_characters())

        # Obtain the top guesses for each field
        ccn_rank = get_correct_rank(move_seq=credit_card_seq.credit_card,
                                    graph=graph,
                                    dictionary=ccn_dictionary,
                                    tv_type=tv_type,
                                    max_num_guesses=args.max_num_results,
                                    target=labels[file_name]['card'])

        month_rank = get_correct_rank(move_seq=credit_card_seq.expiration_month,
                                      graph=graph,
                                      dictionary=numeric_dictionary,
                                      tv_type=tv_type,
                                      max_num_guesses=args.max_num_results,
                                      target=labels[file_name]['month'])

        year_rank = get_correct_rank(move_seq=credit_card_seq.expiration_year,
                                     graph=graph,
                                     dictionary=exp_year_dictionary,
                                     tv_type=tv_type,
                                     max_num_guesses=args.max_num_results,
                                     target=labels[file_name]['year'])

        cvv_rank = get_correct_rank(move_seq=credit_card_seq.security_code,
                                    graph=graph,
                                    dictionary=numeric_dictionary,
                                    tv_type=tv_type,
                                    max_num_guesses=args.max_num_results,
                                    target=labels[file_name]['cvv'])

        zip_code_rank = get_correct_rank(move_seq=credit_card_seq.zip_code,
                                         graph=graph,
                                         dictionary=zip_code_dictionary,
                                         tv_type=tv_type,
                                         max_num_guesses=args.max_num_results,
                                         target=labels[file_name]['zip_code'])

        ranks = [ccn_rank, month_rank, year_rank, cvv_rank, zip_code_rank]
        print('{}: {}'.format(file_name, ranks))
        
        if any([r is None for r in ranks]):
            rank = -1
            did_find = False
        else:
            rank = np.prod(ranks)
            did_find = True

        if did_find:
            rank_list.append(rank)
        else:
            num_not_found += 1

        total_count += 1

    avg_rank = np.average(rank_list)
    med_rank = np.median(rank_list)

    print('Avg Rank: {:.4f}, Median Rank: {:.4f}'.format(avg_rank, med_rank))
    #print('Top 10 Accuracy: {:.4f}'.format(top10_correct / total_count))
    print('Num Not Found: {} ({:.4f})'.format(num_not_found, num_not_found / total_count))
