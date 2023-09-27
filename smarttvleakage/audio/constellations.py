import numpy as np
from collections import namedtuple
from scipy.ndimage import maximum_filter
from typing import Tuple


ConstellationMatch = namedtuple('ConstellationMatch', ['step', 'match_frac', 'match_dist'])


def compute_constellation_map(spectrogram: np.ndarray, freq_delta: int, time_delta: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    assert freq_delta > 0, 'Must provide a positive frequency delta'
    assert time_delta > 0, 'Must provide a positive time delta'

    # Find the peaks
    filtered_spectrogram = maximum_filter(spectrogram, size=(freq_delta, time_delta), mode='nearest')
    peak_matrix = np.logical_and(np.isclose(filtered_spectrogram, spectrogram), (spectrogram > threshold)).astype(int)
    freq_peaks, time_peaks = np.argwhere(peak_matrix == 1).T

    return time_peaks, freq_peaks
