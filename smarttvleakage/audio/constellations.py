import numpy as np
from scipy.ndimage import maximum_filter
from typing import List, Tuple


def compute_constellation_map(spectrogram: np.ndarray, freq_delta: int, time_delta: int, threshold: float, freq_range: Tuple[int, int]) -> Tuple[List[int], List[int]]:
    assert freq_delta > 0, 'Must provide a positive frequency delta'
    assert time_delta > 0, 'Must provide a positive time delta'

    filtered_spectrogram = maximum_filter(spectrogram, size=(freq_delta + 1, time_delta + 1), mode='constant', cval=0.0)
    peak_matrix = np.logical_and(np.isclose(filtered_spectrogram, spectrogram), (spectrogram > threshold)).astype(int)

    freq_peaks, time_peaks = np.argwhere(peak_matrix == 1).T

    filtered_time_peaks: List[int] = []
    filtered_freq_peaks: List[int] = []

    for time, freq in zip(time_peaks, freq_peaks):
        if (freq >= freq_range[0]) and (freq <= freq_range[1]):
            filtered_time_peaks.append(time)
            filtered_freq_peaks.append(freq)

    return filtered_time_peaks, filtered_freq_peaks

    #freq_dims, time_dims = spectrogram.shape

    #time_peaks: List[int] = []
    #freq_peaks: List[int] = []

    #for time_idx in range(0, time_dims, time_delta):
    #    for freq_idx in range(freq_range[0], freq_range[1], freq_delta):
    #        freq_limit = freq_idx + freq_delta
    #        time_limit = time_idx + time_delta

    #        spectrogram_window = spectrogram[freq_idx:freq_limit, time_idx:time_limit]
    #        (max_freq_idx, max_time_idx) = np.unravel_index(np.argmax(spectrogram_window, axis=None), spectrogram_window.shape)
    #        max_val = spectrogram_window[max_freq_idx, max_time_idx]

    #        if max_val > threshold:
    #            time_peaks.append(max_time_idx + time_idx)
    #            freq_peaks.append(max_freq_idx + freq_idx)

    #return time_peaks, freq_peaks


def filter_points_by_time(times: np.ndarray, freqs: np.ndarray, start_time: int, end_time: int) -> Tuple[np.ndarray, np.ndarray]:
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


def match_constellations(target_times: np.ndarray,
                         target_freq: np.ndarray,
                         ref_times: np.ndarray,
                         ref_freq: np.ndarray,
                         freq_tol: float,
                         time_tol: float,
                         time_steps: int) -> Tuple[List[int], Tuple[int]]:
    result_steps: List[int] = []
    result_matches: List[float] = []

    min_ref_time = min(ref_times) - time_tol
    max_ref_time = max(ref_times) + time_tol
    ref_times = [(t - min_ref_time) for t in ref_times]
    num_ref_points = len(ref_times)

    window_size = max_ref_time - min_ref_time

    for base_time in range(time_steps):
        window_times, window_freqs = filter_points_by_time(times=target_times,
                                                           freqs=target_freq,
                                                           start_time=base_time,
                                                           end_time=base_time + window_size)

        if len(window_times) == 0:
            result_steps.append(base_time)
            result_matches.append(0.0)
            continue

        freq_abs_diff = np.abs(np.expand_dims(window_freqs, axis=-1) - np.expand_dims(ref_freq, axis=0))  # [W, K]
        time_abs_diff = np.abs(np.expand_dims(window_times, axis=-1) - np.expand_dims(ref_times, axis=0))  # [W, K]

        comparison = np.logical_and(freq_abs_diff <= freq_tol, time_abs_diff <= time_tol).astype(int)

        freq_matches = np.sum(np.max(comparison, axis=-1))
        time_matches = np.sum(np.max(comparison, axis=0))
        num_matches = time_matches + freq_matches

        num_window_points = len(window_times)
        match_fraction = num_matches / (num_ref_points + num_window_points)

        #if num_matches > 3:
        #    print('Base Time: {}, Comparison: {}, Freq Matches: {}, Time Matches: {}'.format(base_time, comparison, freq_matches, time_matches))
        #    print('Freq Diff: {}, Time Diff: {}'.format(freq_abs_diff, time_abs_diff))
        #    print('Match Fraction: {:.4f}, Num: {}, Denom: {} + {}'.format(match_fraction, num_matches, num_ref_points, num_window_points))
        #    print('==========')

        result_steps.append(base_time)
        result_matches.append(match_fraction)

    return result_steps, result_matches



