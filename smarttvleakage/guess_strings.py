import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from argparse import ArgumentParser
from typing import Tuple, List, Dict

from smarttvleakage.audio import MoveExtractor, make_move_extractor
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import EnglishDictionary, UniformDictionary
from smarttvleakage.search_without_autocomplete import get_words_from_moves
from smarttvleakage.search_with_autocomplete import get_words_from_moves_autocomplete
from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from smarttvleakage.utils.file_utils import read_pickle_gz, iterate_dir

#from smarttvleakage.audio.determine_autocomplete import build_model, classify_ms


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--tv-type', choices=[SmartTVType.SAMSUNG.name.lower(), SmartTVType.APPLE_TV.name.lower()], type=str, required=True)
    parser.add_argument('--max-num-results', type=int)
    parser.add_argument('--max-num-videos', type=int)
    args = parser.parse_args()

    if os.path.isdir(args.video_path):
        video_paths = list(iterate_dir(args.video_path))
        should_plot = False
    else:
        video_paths = [args.video_path]
        should_plot = True

    tv_type = SmartTVType[args.tv_type.upper()]
    keyboard_type = KeyboardType.SAMSUNG if tv_type == SmartTVType.SAMSUNG else KeyboardType.APPLE_TV_SEARCH
    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)
    characters = graph.get_characters()

    print('Starting to load the dictionary...')
    
    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary(characters=characters)
    else:
        dictionary = EnglishDictionary.restore(characters=characters, path=args.dictionary_path)

    print('Finished loading dictionary.')

    # Create the move extractor
    move_extractor = make_move_extractor(tv_type=tv_type)

    rank_list: List[int] = []
    num_candidates_list: List[int] = []
    rank_dict: Dict[str, int] = dict()
    candidates_dict: Dict[str, int] = dict()
    not_found_list: List[str] = []

    top10_correct = 0
    total_count = 0
    num_not_found = 0

    print('Number of video files: {}'.format(len(video_paths)))
    use_suggestions = False

    for idx, video_path in enumerate(video_paths):
        if (args.max_num_videos is not None) and (idx >= args.max_num_videos):
            break

        video_clip = mp.VideoFileClip(video_path)
        audio = video_clip.audio

        file_name = os.path.basename(video_path)
        true_word = file_name.replace('.mp4', '').replace('.MOV', '').replace('.mov', '')
        true_word = true_word.replace('_', ' ')

        signal = audio.to_soundarray()
        #move_sequence, did_use_autocomplete = move_extractor.extract_move_sequence(audio=signal)
        move_sequence, did_use_autocomplete, keyboard_type = move_extractor.extract_move_sequence(audio=signal)

        #suggestions_model = build_model()
        #if (classify_ms(suggestions_model, move_sequence) == 1) and (tv_type == SmartTVType.SAMSUNG):
        #    use_suggestions = True
        #else:
        #    use_suggestions = False

        # Make the graph based on the keyboard type
        graph = MultiKeyboardGraph(keyboard_type=keyboard_type)

        if use_suggestions:
            ranked_candidates = get_words_from_moves_autocomplete(move_sequence=move_sequence,
                                                                  graph=graph,
                                                                  dictionary=dictionary,
                                                                  did_use_autocomplete=did_use_autocomplete,
                                                                  max_num_results=args.max_num_results)
        else:
            ranked_candidates = get_words_from_moves(move_sequence=move_sequence,
                                                     graph=graph,
                                                     dictionary=dictionary,
                                                     tv_type=tv_type,
                                                     max_num_results=args.max_num_results)

        did_find_word = False

        for rank, (guess, score, num_candidates) in enumerate(ranked_candidates):
            print('Guess: {}, Score: {}'.format(guess, score))

            if guess == true_word:
                rank_list.append(rank + 1)
                rank_dict[true_word] = rank + 1
                num_candidates_list.append(num_candidates)
                candidates_dict[true_word] = num_candidates

                did_find_word = True
                break

        top10_correct += int(did_find_word and (rank <= 10))
        total_count += 1

        if (not did_find_word) and (args.max_num_results is not None):
            rank = args.max_num_results

        if (not did_find_word):
            not_found_list.append(true_word)

        print('==========')
        print('Word: {}'.format(true_word))
        print('Rank: {} (Did Find: {})'.format(rank + 1, did_find_word))
        print('Move Sequence: {} (Did Use Autocomplete: {})'.format(list(map(lambda move: move.num_moves, move_sequence)), did_use_autocomplete))

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
    print('Words not found: {}'.format(not_found_list))
