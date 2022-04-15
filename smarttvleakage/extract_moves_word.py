import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from argparse import ArgumentParser
from scipy.signal import correlate, convolve, find_peaks
from typing import Tuple, List, Dict

from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.dictionary import EnglishDictionary, UniformDictionary
from smarttvleakage.graph_search import get_words_from_moves
from smarttvleakage.utils.file_utils import read_pickle_gz, iterate_dir


MOVE_THRESHOLD = 10.0
MOVE_CUTOFF = 0.9
MOVE_TIME = 2500

SELECT_THRESHOLD = 0.05
SELECT_MIN = 0.05
SELECT_MAX = 0.1

WINDOW_SIZE = 256
MAX_NUM_RESULTS = 10


def filter_moves(move_peaks: np.ndarray, move_peak_heights: np.ndarray) -> Tuple[List[float], List[float]]:
    if len(move_peaks) == 0:
        return [], []

    result_peaks: List[float] = []
    result_heights: List[float] = []

    result_peaks.append(move_peaks[0])
    result_heights.append(move_peak_heights[0])

    for idx in range(1, len(move_peaks)):
        prev_peak = result_peaks[-1]
        prev_height = result_heights[-1]

        curr_peak = move_peaks[idx]
        curr_height = move_peak_heights[idx]

        if abs(prev_peak - curr_peak) > MOVE_TIME * 2:
            result_peaks.append(curr_peak)
            result_heights.append(curr_height)
        else:
            if prev_height > curr_height:
                if prev_height * MOVE_CUTOFF <= curr_height:
                    result_peaks.append(curr_peak)
                    result_heights.append(curr_height)
            else:
                if curr_height * MOVE_CUTOFF <= prev_height:
                    result_peaks.append(curr_peak)
                    result_heights.append(curr_height)
                else:
                    result_peaks[-1] = curr_peak
                    result_heights[-1] = curr_height

    return result_peaks, result_heights


def filter_select(select_peaks: np.ndarray, select_peak_heights: np.ndarray) -> Tuple[List[float], List[float]]:
    result_peaks: List[float] = []
    result_heights: List[float] = []

    for peak, height in zip(select_peaks, select_peak_heights):
        if (height >= SELECT_MIN) and (height <= SELECT_MAX):
            result_peaks.append(peak)
            result_heights.append(height)

    return result_peaks, result_heights


def extract_moves_for_word(audio_signal: np.ndarray, sounds_folder: str, should_plot: bool) -> List[int]:
    channel0, channel1 = audio_signal[:, 0], audio_signal[:, 1]

    move_sound = read_pickle_gz(os.path.join(sounds_folder, 'move.pkl.gz'))
    select_sound = read_pickle_gz(os.path.join(sounds_folder, 'select.pkl.gz'))

    move_correlation = correlate(in1=audio_signal, in2=move_sound)
    move_correlation_norm = np.sum(np.abs(move_correlation), axis=-1)

    select_correlation = correlate(in1=audio_signal, in2=select_sound)
    select_correlation_norm = np.sum(np.abs(select_correlation), axis=-1)

    avg_filter = (1.0 / WINDOW_SIZE) * np.ones(shape=(WINDOW_SIZE, ))
    filtered_move_correlation = convolve(in1=move_correlation_norm, in2=avg_filter)
    filtered_select_correlation = convolve(in1=select_correlation_norm, in2=avg_filter)

    num_moves = 0
    num_selects = 0

    move_starts: List[int] = []
    move_ends: List[int] = []

    select_starts: List[int] = []
    select_ends: List[int] = []

    move_start_idx = None
    select_start_idx = None

    move_peaks, move_peak_properties = find_peaks(x=filtered_move_correlation, height=MOVE_THRESHOLD, distance=MOVE_TIME)
    select_peaks, select_peak_properties = find_peaks(x=filtered_select_correlation, height=SELECT_THRESHOLD, distance=5000)

    move_peaks, move_peak_heights = filter_moves(move_peaks=move_peaks,
                                                 move_peak_heights=move_peak_properties['peak_heights'])

    select_peaks, select_peak_heights = filter_select(select_peaks=select_peaks,
                                                      select_peak_heights=select_peak_properties['peak_heights'])

    # Split the moves into a sequences based on `select`.
    num_moves_list: List[int] = []

    for idx in range(1, len(select_peaks)):
        start_time = select_peaks[idx - 1]
        end_time  = select_peaks[idx]

        num_moves = len(list(filter(lambda t: (t >= start_time) and (t <= end_time), move_peaks)))
        num_moves_list.append(num_moves)

    num_moves = len(move_peaks)
    num_selects = len(select_peaks)

    if should_plot:
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

        ax0.plot(list(range(len(filtered_move_correlation))), filtered_move_correlation)
        ax1.plot(list(range(len(filtered_select_correlation))), filtered_select_correlation)

        ax0.scatter(move_peaks, move_peak_heights, marker='o', color='orange')
        ax1.scatter(select_peaks, select_peak_heights, marker='o', color='orange')

        ax0.set_title('Moving Avg Correlation with Move Sound')
        ax1.set_title('Moving Avg Correlation with Select Sound')

        ax0.set_ylabel('Cross Correlation L1 Norm')
        ax1.set_ylabel('Cross Correlation L1 Norm')
        ax1.set_xlabel('Time')

        plt.tight_layout()
        plt.show()

    return num_moves_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--dictionary-path', type=str, required=True)
    parser.add_argument('--sounds-folder', type=str, required=True)
    args = parser.parse_args()

    if os.path.isdir(args.video_path):
        video_paths = list(iterate_dir(args.video_path))
        should_plot = False
    else:
        video_paths = [args.video_path]
        should_plot = True

    graph = MultiKeyboardGraph()
    
    if args.dictionary_path == 'uniform':
        dictionary = UniformDictionary()
    else:
        dictionary = EnglishDictionary.restore(path=args.dictionary_path)

    rank_list: List[int] = []
    num_candidates_list: List[int] = []
    rank_dict: Dict[str, int] = dict()

    for video_path in video_paths:
        video_clip = mp.VideoFileClip(video_path)
        audio = video_clip.audio

        file_name = os.path.basename(video_path)
        true_word = file_name.replace('.mp4', '').replace('.MOV', '')

        signal = audio.to_soundarray()
        num_moves = extract_moves_for_word(audio_signal=signal, sounds_folder=args.sounds_folder, should_plot=should_plot)

        # TODO: Add the 'cancel' sound into this. For now, we assume the last 'select' is for completion
        num_moves = num_moves[0:-1]

        ranked_candidates = get_words_from_moves(num_moves=num_moves,
                                                 graph=graph,
                                                 dictionary=dictionary,
                                                 max_num_results=None)

        did_find_word = False

        for rank, (guess, score, num_candidates) in enumerate(ranked_candidates):
            if guess == true_word:
                rank_list.append(rank + 1)
                rank_dict[true_word] = rank + 1
                num_candidates_list.append(num_candidates)

                did_find_word = True
                break

        if not did_find_word:
            rank_list.append(rank + 1)
            rank_dict[true_word] = rank + 1

        if should_plot:
            print('Number of Moves: {}'.format(num_moves))

    avg_rank = np.average(rank_list)
    med_rank = np.median(rank_list)

    avg_num_candidates = np.average(num_candidates_list)
    med_num_candidates = np.median(num_candidates_list)

    print('Ranking Dict: {}'.format(rank_dict))
    print('Avg Rank: {:.4f}, Median Rank: {:.4f}'.format(avg_rank, med_rank))
    print('Avg # Candidates: {:.4f}, Median # Candidates: {:.4f}'.format(avg_num_candidates, med_num_candidates))
