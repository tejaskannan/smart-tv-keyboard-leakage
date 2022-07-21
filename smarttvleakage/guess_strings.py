import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from argparse import ArgumentParser
from typing import Tuple, List, Dict

from smarttvleakage.audio import MoveExtractor
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import EnglishDictionary, UniformDictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_autocomplete
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_pickle_gz, iterate_dir

from smarttvleakage.audio.determine_autocomplete import build_model, classify_ms


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--max-num-results', type=int)
    parser.add_argument('--max-num-videos', type=int)
    args = parser.parse_args()

    if os.path.isdir(args.video_path):
        video_paths = list(iterate_dir(args.video_path))
        should_plot = False
    else:
        video_paths = [args.video_path]
        should_plot = True

    graph = MultiKeyboardGraph()

    print('Starting to load the dictionary...')
    
    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary()
    else:
        dictionary = EnglishDictionary.restore(path=args.dictionary_path)

    print('Finished loading dictionary.')

    # Create the move extractor
    move_extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)

    rank_list: List[int] = []
    num_candidates_list: List[int] = []
    rank_dict: Dict[str, int] = dict()
    candidates_dict: Dict[str, int] = dict()

    top10_correct = 0
    total_count = 0
    num_not_found = 0

    print('Number of video files: {}'.format(len(video_paths)))

    for idx, video_path in enumerate(video_paths):
        if (args.max_num_videos is not None) and (idx >= args.max_num_videos):
            break

        video_clip = mp.VideoFileClip(video_path)
        audio = video_clip.audio

        file_name = os.path.basename(video_path)
        true_word = file_name.replace('.mp4', '').replace('.MOV', '').replace('.mov', '')

        signal = audio.to_soundarray()
        #move_sequence, did_use_autocomplete = move_extractor.extract_move_sequence(audio=signal)
        move_sequence, _ = move_extractor.extract_move_sequence(audio=signal)
        
        autocomplete_model = build_model()
        if classify_ms(autocomplete_model, move_sequence) == 1:
            use_autocomplete = True
        else:
            use_autocomplete = False

        if use_autocomplete:
            ranked_candidates = get_words_from_moves_autocomplete(move_sequence=move_sequence,
                                                                  graph=graph,
                                                                  dictionary=dictionary,
                                                                  did_use_autocomplete=True,
                                                                  max_num_results=args.max_num_results)
        else:
            ranked_candidates = get_words_from_moves(move_sequence=move_sequence,
                                                     graph=graph,
                                                     dictionary=dictionary,
                                                     max_num_results=args.max_num_results)

        did_find_word = False

        for rank, (guess, score, num_candidates) in enumerate(ranked_candidates):
            if guess == true_word:
                rank_list.append(rank + 1)
                rank_dict[true_word] = rank + 1
                num_candidates_list.append(num_candidates)
                candidates_dict[true_word] = num_candidates

                did_find_word = True
                break

        top10_correct += int(did_find_word and (rank < 10))
        total_count += 1

        if (not did_find_word) and (args.max_num_results is not None):
            rank = args.max_num_results

        print('==========')
        print('Word: {}'.format(true_word))
        print('Rank: {} ({})'.format(rank + 1, did_find_word))
        print('Move Sequence: {} ({})'.format(move_sequence, use_autocomplete))

        if not did_find_word:
            rank_list.append(rank + 1)
            rank_dict[true_word] = rank + 1
            candidates_dict[true_word] = num_candidates
            num_not_found += 1

    avg_rank = np.average(rank_list)
    med_rank = np.median(rank_list)

    avg_num_candidates = np.average(num_candidates_list)
    med_num_candidates = np.median(num_candidates_list)

    print('Ranking Dict: {}'.format(rank_dict))
    print('Candidates Dict: {}'.format(candidates_dict))
    print('Avg Rank: {:.4f}, Median Rank: {:.4f}'.format(avg_rank, med_rank))
    print('Avg # Candidates: {:.4f}, Median # Candidates: {:.4f}'.format(avg_num_candidates, med_num_candidates))
    print('Top 10 Accuracy: {:.4f}'.format(top10_correct / total_count))
    print('Num Not Found: {} ({:.4f})'.format(num_not_found, num_not_found / total_count))
