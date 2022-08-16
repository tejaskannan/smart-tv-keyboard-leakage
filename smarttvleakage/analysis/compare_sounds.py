import numpy as np
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from collections import deque
from matplotlib import image
from scipy.ndimage import maximum_filter
from typing import List, Tuple

from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations
from smarttvleakage.audio.move_extractor import moving_window_similarity, create_spectrogram, compute_masked_spectrogram
from smarttvleakage.utils.file_utils import read_pickle_gz

# Key Select Parameters: (20, 40), threshold -75, constellation deltas (10, 10), TOL (2, 2)
# Select Parameters: (0, 20), threshold -67, constellation deltas (freq 3, time 5), TOL (1, 2)
# Move -> THRESHOLD -67, DELTAS: (t = 5, f = 3)  RANGE (0, 20), TOL (f 2, t 1), window size 25 -> set cutoff at 0.45
# DELETE -> THRESHOLD -67, DELTAS: (t = 5, f = 3), TOL: (t = 1, freq = 2) -> set cutoff at 0.95 (if move score below 315 and backspace -> pick backspace)


FREQ_DELTA = 3
TIME_DELTA = 5
MIN_THRESHOLD = -55
MAX_THRESHOLD = -40
FREQ_RANGE = (0, 40)
FREQ_TOL = 2
TIME_TOL = 2

PLOT_DISTANCES = True
PLOT_TARGET = False


def compute_masked_spectrogram(spectrogram: float, min_threshold: float, max_threshold: float, min_freq: int, max_freq: int) -> np.ndarray:
    clipped_spectrogram = spectrogram[min_freq:max_freq, :]
    return np.logical_and(clipped_spectrogram >= min_threshold, clipped_spectrogram <= max_threshold).astype(int)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sound-folder', type=str, required=True)
    parser.add_argument('--ref-sound', type=str, required=True)
    parser.add_argument('--sounds', type=str, nargs='+', required=True)
    args = parser.parse_args()

    with plt.style.context('seaborn-ticks'):
        fig, axes = plt.subplots(nrows=len(args.sounds))

        ref_sound = read_pickle_gz(os.path.join(args.sound_folder, '{}.pkl.gz'.format(args.ref_sound)))
        ref_channel0 = ref_sound[:, 0]
        ref_spectrogram = create_spectrogram(ref_channel0)
        ref_masked = compute_masked_spectrogram(ref_spectrogram, min_threshold=MIN_THRESHOLD, max_threshold=MAX_THRESHOLD, min_freq=FREQ_RANGE[0], max_freq=FREQ_RANGE[1])

        for idx, sound_name in enumerate(args.sounds):
            audio_signal = read_pickle_gz(os.path.join(args.sound_folder, '{}.pkl.gz'.format(sound_name)))
            channel0 = audio_signal[:, 0]

            target_spectrogram = create_spectrogram(channel0)
            target_masked = compute_masked_spectrogram(target_spectrogram, min_threshold=MIN_THRESHOLD, max_threshold=MAX_THRESHOLD, min_freq=FREQ_RANGE[0], max_freq=FREQ_RANGE[1])

            similarity = moving_window_similarity(target=target_masked, known=ref_masked, should_smooth=True, should_match_binary=True)
            axes[idx].plot(list(range(len(similarity))), similarity)
            axes[idx].set_title('Target: {}, Ref: {}'.format(sound_name, args.ref_sound))

        plt.tight_layout()
        plt.show()
