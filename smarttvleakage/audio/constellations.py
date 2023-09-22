import numpy as np
import matplotlib.pyplot as plt
import math
from collections import namedtuple
from scipy.ndimage import maximum_filter
from typing import List, Tuple

from smarttvleakage.utils.constants import BIG_NUMBER, SMALL_NUMBER


ConstellationMatch = namedtuple('ConstellationMatch', ['step', 'match_frac', 'match_dist'])


def compute_constellation_map(spectrogram: np.ndarray, freq_delta: int, time_delta: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    assert freq_delta > 0, 'Must provide a positive frequency delta'
    assert time_delta > 0, 'Must provide a positive time delta'

    # Find the peaks
    filtered_spectrogram = maximum_filter(spectrogram, size=(freq_delta, time_delta), mode='nearest')
    peak_matrix = np.logical_and(np.isclose(filtered_spectrogram, spectrogram), (spectrogram > threshold)).astype(int)
    freq_peaks, time_peaks = np.argwhere(peak_matrix == 1).T

    return time_peaks, freq_peaks


def filter_and_shift_by_time(times: np.ndarray, freqs: np.ndarray, start_time: int, end_time: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters the given 2D points by times in the given range. Shifts times by the start time.
    """
    result_times: List[int] = []
    result_freqs: List[int] = []

    for t, freq in zip(times, freqs):
        if (t >= start_time) and (t <= end_time):
            result_times.append(t - start_time)
            result_freqs.append(freq)

    return np.array(result_times), np.array(result_freqs)


def match_constellations(target_spectrogram: np.ndarray,
                         ref_spectrogram: np.ndarray,
                         time_delta: int,
                         freq_delta: int,
                         threshold: float,
                         time_tol: int,
                         freq_tol: int,
                         window_buffer: int) -> List[float]:
    assert target_spectrogram.shape[0] == ref_spectrogram.shape[0], 'Spectrograms must have the same number of freq components'
    assert len(target_spectrogram.shape) == 2, 'Must provide a 2d target spectrogram'
    assert len(ref_spectrogram.shape) == 2, 'Must provide a 2d reference spectrogram'

    # Compute the constellation maps
    target_times, target_freq = compute_constellation_map(target_spectrogram, freq_delta=freq_delta, time_delta=time_delta, threshold=threshold)
    ref_times, ref_freq = compute_constellation_map(ref_spectrogram, freq_delta=freq_delta, time_delta=time_delta, threshold=threshold)

    # Shift the reference times to clip out unecessary space
    min_time, max_time = min(ref_times), max(ref_times)

    ref_times -= min_time
    window_size = (max_time - min_time) + window_buffer

    result: List[float] = []
    for start_time in range(target_spectrogram.shape[1]):
        end_time = start_time + window_size
        window_times, window_freq = filter_and_shift_by_time(times=target_times, freqs=target_freq, start_time=start_time, end_time=end_time)

        match_score = match_constellation_window(target_times=window_times,
                                                 target_freq=window_freq,
                                                 ref_times=ref_times,
                                                 ref_freq=ref_freq,
                                                 time_tol=time_tol,
                                                 freq_tol=freq_tol,
                                                 start_time=start_time)
        result.append(match_score)

    return result


def match_constellation_window(target_times: np.ndarray,
                               target_freq: np.ndarray,
                               ref_times: np.ndarray,
                               ref_freq: np.ndarray,
                               time_tol: int,
                               freq_tol: int,
                               start_time: int) -> float:
    # Validate the arguments
    assert target_times.shape == target_freq.shape, 'Must provide the same number of target times and frequencies'
    assert ref_times.shape == ref_freq.shape, 'Must provide the same number of reference times and frequencies'
    assert len(target_times.shape) == 1, 'Must provide a 1d array of target times'
    assert len(ref_times.shape) == 1, 'Must provide a 1d array of reference times'
    assert freq_tol >= 0, 'Frequency tolerance must be non-negative'
    assert time_tol >= 0, 'Time tolerance must be non-negative'

    if (len(target_times) == 0) or (len(ref_times) == 0):
        return 0.0

    # Get the pairwise differences in time and frequency indices
    time_diffs = np.abs(np.expand_dims(target_times, axis=-1) - np.expand_dims(ref_times, axis=0))  # [W, K]
    freq_diffs = np.abs(np.expand_dims(target_freq, axis=-1) - np.expand_dims(ref_freq, axis=0))  # [W, K]
    point_distances = time_diffs + freq_diffs  # [W, K]

    # Compute the matches in a greedy fashion
    num_matches = 0

    for target_idx in range(len(target_times)):
        min_ref_idx = np.argmin(point_distances[target_idx])  # A scalar in [K]
        is_match = (time_diffs[target_idx, min_ref_idx] <= time_tol) and (freq_diffs[target_idx, min_ref_idx] <= freq_tol)

        num_matches += 2 * int(is_match)  # Count 1 for ref and 1 for target

        # Clear the reference scores to avoid matching the point twice
        if is_match:
            point_distances[:, min_ref_idx] = BIG_NUMBER
            time_diffs[:, min_ref_idx] = BIG_NUMBER
            freq_diffs[:, min_ref_idx] = BIG_NUMBER

    # Compute the final matching score
    denom = 2 * max(len(target_times), len(ref_times))
    return num_matches / denom
