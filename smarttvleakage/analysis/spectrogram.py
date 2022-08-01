import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from collections import deque, namedtuple
from matplotlib import image
from scipy.ndimage import maximum_filter
from scipy.signal import convolve
from typing import List, Tuple

from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations
from smarttvleakage.audio.move_extractor import moving_window_similarity, create_spectrogram
from smarttvleakage.utils.file_utils import read_pickle_gz

# Key Select Parameters: (20, 40), threshold -75, constellation deltas (10, 10), TOL (2, 2)
# Select Parameters: (0, 20), threshold -67, constellation deltas (freq 3, time 5), TOL (1, 2)
# Move -> THRESHOLD -67, DELTAS: (t = 5, f = 3)  RANGE (0, 20), TOL (f 2, t 1), window size 25 -> set cutoff at 0.45
# DELETE -> THRESHOLD -67, DELTAS: (t = 5, f = 3), TOL: (t = 1, freq = 2) -> set cutoff at 0.95 (if move score below 315 and backspace -> pick backspace)

ThresholdRange = namedtuple('ThresholdRange', ['min_threshold', 'max_threshold', 'min_freq', 'max_freq'])


# Samsung
KEY_SELECT_RANGES = [ThresholdRange(-60, -40, 0, 20), ThresholdRange(-75, -65, 20, 40)]
MOVE_RANGES = [ThresholdRange(-55, -45, 0, 20), ThresholdRange(-65, -55, 20, 30)]
SELECT_RANGES = [ThresholdRange(-60, -55, 0, 10), ThresholdRange(-60, -45, 10, 30)]

# Apple
SYSTEM_MOVE_RANGES = [ThresholdRange(-65, -55, 0, 12), ThresholdRange(-60, -40, 12, 40)]
KEYBOARD_MOVE_RANGES = [ThresholdRange(-50, -38, 0, 15), ThresholdRange(-60, -50, 15, 40)]
KEYBOARD_SELECT_RANGES = [ThresholdRange(-60, -40, 0, 30)]
DELETE_RANGES = [ThresholdRange(-55, -35, 0, 20), ThresholdRange(-65, -55, 20, 40)]  # Set Threshold at 1.1


#FREQ_DELTA = 3
#TIME_DELTA = 5
#MIN_THRESHOLD = -60
#MAX_THRESHOLD = -40
#FREQ_RANGE = (0, 20)
#FREQ_TOL = 2
#TIME_TOL = 2
WINDOW_SIZE = 8

PLOT_DISTANCES = True
PLOT_TARGET = False


def compute_masked_spectrogram(spectrogram: float, threshold_range: ThresholdRange) -> np.ndarray:
    clipped_spectrogram = spectrogram[threshold_range.min_freq:threshold_range.max_freq, :]
    return np.logical_and(clipped_spectrogram >= threshold_range.min_threshold, clipped_spectrogram <= threshold_range.max_threshold).astype(int)


def moving_window_similarity(target: np.ndarray, known: np.ndarray) -> List[float]:
    target = target.T
    known = known.T

    segment_size = known.shape[0]
    similarity: List[float] = []

    for start in range(target.shape[0]):
        end = start + segment_size
        target_segment = target[start:end]

        if len(target_segment) < segment_size:
            target_segment = np.pad(target_segment, pad_width=[(0, segment_size - len(target_segment)), (0, 0)], constant_values=0, mode='constant')

        #sim_score = 1.0 / np.linalg.norm(target_segment - known, ord=1)
        if np.all(target_segment == known):
            sim_score = 1.0
        else:
            sim_score = 2 * np.sum(target_segment * known) / (np.sum(target_segment) + np.sum(known))

        similarity.append(sim_score)

    smooth_filter = np.ones(shape=(WINDOW_SIZE, )) / WINDOW_SIZE
    similarity = convolve(similarity, smooth_filter).astype(float).tolist()

    return similarity


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--sound-file', type=str, required=True)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    channel0, channel1 = audio_signal[:, 0], audio_signal[:, 1]

    known_sound = read_pickle_gz(args.sound_file)
    known_channel0 = known_sound[:, 0]

    target = create_spectrogram(channel0)
    known = create_spectrogram(known_channel0)
    threshold_ranges = SELECT_RANGES

#    target_times, target_freq = compute_constellation_map(target, freq_delta=FREQ_DELTA, time_delta=TIME_DELTA, threshold=THRESHOLD, freq_range=FREQ_RANGE)
#
#    known_times, known_freq = compute_constellation_map(known, freq_delta=FREQ_DELTA, time_delta=TIME_DELTA, threshold=THRESHOLD, freq_range=FREQ_RANGE)
#
#    match_times, match_fracs = match_constellations(target_times=target_times,
#                                                    target_freq=target_freq,
#                                                    ref_times=known_times,
#                                                    ref_freq=known_freq,
#                                                    freq_tol=FREQ_TOL,
#                                                    time_tol=TIME_TOL,
#                                                    window_size=25,
#                                                    time_steps=target.shape[1])
#

    similarity_list: List[List[float]] = []
    for threshold_range in threshold_ranges:
        target_masked = compute_masked_spectrogram(target, threshold_range)
        known_masked = compute_masked_spectrogram(known, threshold_range)
        similarity_list.append(moving_window_similarity(target=target_masked, known=known_masked))

    similarity = np.sum(similarity_list, axis=0)

    if PLOT_DISTANCES:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.plot(list(range(len(channel0))), channel0)
        #ax2.plot(match_times, match_fracs)
        ax2.plot(list(range(len(similarity))), similarity)
    else:
        fig, ax = plt.subplots()

        if PLOT_TARGET:
            ax.imshow(target, cmap='gray_r')
            #ax.scatter(target_times, target_freq, color='red', marker='o')
        else:
            ax.imshow(known, cmap='gray_r')
            #ax.scatter(known_times, known_freq, color='red', marker='o')

    plt.tight_layout()
    plt.show()


    #distances = moving_window_distances(target=target, known=known, should_smooth=True)

    #peaks, peak_properties = find_peaks(distances, height=0.01, distance=3500)
    #peak_heights = peak_properties['peak_heights']

    #plt.pcolormesh(times, freq[0:100], Pxx[0:100])
    #plt.show()

    #fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    #ax0.plot(list(range(len(channel0))), channel0)
    #ax1.plot(list(range(len(distances))), distances)
    #ax1.scatter(peaks, peak_heights, color='orange')
    #plt.show()
