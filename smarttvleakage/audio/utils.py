import numpy as np
from scipy.signal import spectrogram, find_peaks, convolve
from typing import List, Tuple, Set, Union

from smarttvleakage.utils.constants import BIG_NUMBER, SMALL_NUMBER, Direction


CHANGE_DIR_MAX_THRESHOLD = 150  # Very long delays may be a user just pausing, so we filter them out


def does_conflict(candidate_time: int, comparison_times: List[int], forward_distance: int, backward_distance: int) -> bool:
    for t in comparison_times:
        diff = abs(t - candidate_time)

        if (candidate_time > t) and (diff < forward_distance):
            return True
        elif (candidate_time <= t) and (diff < backward_distance):
            return True

    return False


def filter_conflicts(target_times: List[int], comparison_times: List[List[int]], forward_distance: int, backward_distance: int) -> List[int]:
    filtered_times: List[int] = []

    for target in target_times:
        is_a_conflict = any((does_conflict(target, comparison, forward_distance, backward_distance) for comparison in comparison_times))
        if not is_a_conflict:
            filtered_times.append(target)

    return filtered_times


def perform_match_constellations(target_times: np.ndarray, target_freq: np.ndarray, ref_times: np.ndarray, ref_freq: np.ndarray, time_tol: int, freq_tol: int) -> float:
    # Compute the distance in time, frequency, and ranking
    time_diffs = np.abs(np.expand_dims(ref_times, axis=-1) - np.expand_dims(target_times, axis=0))  # [N, M]
    freq_diffs = np.abs(np.expand_dims(ref_freq, axis=-1) - np.expand_dims(target_freq, axis=0))  # [N, M]
    distances = time_diffs + freq_diffs  # [N, M]

    num_matches = 0

    for ref_idx in range(len(ref_times)):
        closest_target_idx = np.argmin(distances[ref_idx])

        if (time_diffs[ref_idx, closest_target_idx] <= time_tol) and (freq_diffs[ref_idx, closest_target_idx] <= freq_tol):
            num_matches += 1
            distances[ref_idx, :] = BIG_NUMBER
            distances[:, closest_target_idx] = BIG_NUMBER

    return num_matches / max(len(target_times), len(ref_times))



def perform_match_constellations_geometry(target_times: np.ndarray, target_freq: np.ndarray, ref_times: np.ndarray, ref_freq: np.ndarray) -> float:
    time_diffs = np.abs(np.expand_dims(target_times, axis=-1) - np.expand_dims(ref_times, axis=0))  # [N, K]
    freq_diffs = np.abs(np.expand_dims(target_freq, axis=-1) - np.expand_dims(ref_freq, axis=0))  # [N, K]

    time_nearby = np.sum((time_diffs <= 1).astype(int), axis=-1)  # [N]
    freq_nearby = np.sum((freq_diffs <= 1).astype(int), axis=-1)  # [N]

    time_matches = (time_nearby >= 2).astype(int)
    freq_matches = (freq_nearby >= 2).astype(int)

    matches = np.average(np.clip(time_matches + freq_matches, a_min=0, a_max=1))
    return matches


def perform_match_spectrograms(first_spectrogram: np.ndarray, second_spectrogram: np.ndarray) -> float:
    if first_spectrogram.shape[1] < second_spectrogram.shape[1]:
        return perform_match_spectrograms(second_spectrogram, first_spectrogram)

    similarity = 0.0

    for start_time in range(0, first_spectrogram.shape[1] - second_spectrogram.shape[1] + 1):
        end_time = start_time + second_spectrogram.shape[1]
        first_segment = first_spectrogram[:, start_time:end_time]

        dist = max(np.linalg.norm(first_segment - second_spectrogram, ord=2), SMALL_NUMBER)
        similarity = max(similarity, 1.0 / dist)

    return similarity


def perform_match_binary(first_spectrogram: np.ndarray, second_spectrogram: np.ndarray) -> float:
    if first_spectrogram.shape[1] < second_spectrogram.shape[1]:
        return perform_match(second_spectrogram, first_spectrogram)  # Make sure the first spectrogram has the most time steps

    first_sum = np.sum(first_spectrogram)
    second_sum = np.sum(second_spectrogram)
    max_similarity = max(first_sum, second_sum)

    time_diff = first_spectrogram.shape[1] - second_spectrogram.shape[1]
    pad_after = time_diff
    best_similarity = -1.0

    while pad_after >= 0:
        pad_before = time_diff - pad_after
        padded_spectrogram = np.pad(second_spectrogram, pad_width=[(0, 0), (pad_before, pad_after)], mode='constant')  # [F, T]

        comparison = np.sum(padded_spectrogram * first_spectrogram)  # Scaler
        similarity = comparison / max_similarity

        best_similarity = max(similarity, best_similarity)
        pad_after -= 1

    return best_similarity


def is_in_peak_range(peak_time: int, peak_ranges: Set[Tuple[int, int]]) -> bool:
    for peak_range in peak_ranges:
        if (peak_time >= peak_range[0]) and (peak_time <= peak_range[1]):
            return True

    return False


def get_sound_instances(spect: np.ndarray, forward_factor: float, backward_factor: float, peak_height: float, peak_distance: int, peak_prominence: float, smooth_window_size: int) -> Tuple[List[int], List[int]]:
    # First, normalize the energy values for each frequency
    mean_energy = np.mean(spect, axis=-1, keepdims=True)
    std_energy = np.std(spect, axis=-1, keepdims=True)
    max_energy = np.max((spect - mean_energy) / std_energy, axis=0)  # [T]

    if smooth_window_size > 1:
        smooth_filter = np.ones(shape=(smooth_window_size, ), dtype=max_energy.dtype) / float(smooth_window_size)
        max_energy = convolve(max_energy, smooth_filter, mode='full')  # [T]

    peak_ranges: Set[Tuple[int, int]] = set()

    peak_times, peak_properties = find_peaks(max_energy, height=peak_height, distance=peak_distance, prominence=(peak_prominence, None))
    peak_heights = peak_properties['peak_heights']
    prev_end = 1

    for peak_time, peak_height in zip(peak_times, peak_heights):
        if is_in_peak_range(peak_time, peak_ranges):
            continue

        forward_peak_threshold = peak_height * forward_factor
        backward_peak_threshold = peak_height * backward_factor

        # Get the start and end point
        start_time = peak_time
        while (start_time > prev_end) and ((max_energy[start_time] > max_energy[start_time - 1]) or (max_energy[start_time] > backward_peak_threshold)):
            start_time -= 1

        start_time += 1  # Adjust for going beyond the threshold

        end_time = peak_time
        while (end_time < (len(max_energy) - 1)) and ((max_energy[end_time] > max_energy[end_time + 1]) or (max_energy[end_time] > forward_peak_threshold)):
            end_time += 1

        # Add the range to the result set
        peak_ranges.add((start_time, end_time))
        prev_end = end_time

    # Sort the ranges by start time
    peak_ranges_sorted = list(sorted(peak_ranges, key=lambda t: t[0]))
    start_times = [t[0] for t in peak_ranges_sorted]
    end_times = [t[1] for t in peak_ranges_sorted]

    return start_times, end_times


def extract_move_directions(move_times: List[int]) -> Union[List[Direction], Direction]:
    if (len(move_times) <= 4):
        return Direction.ANY

    # List of at least length 4 diffs
    time_diffs = [ahead - behind for ahead, behind in zip(move_times[1:], move_times[:-1])]

    baseline_avg = np.average(time_diffs[0:-1])
    baseline_std = np.std(time_diffs[0:-1])
    cutoff = baseline_avg + 3 * baseline_std

    if (time_diffs[-1] >= cutoff) and (time_diffs[-1] < CHANGE_DIR_MAX_THRESHOLD):
        directions = [Direction.HORIZONTAL for _  in range(len(move_times) - 1)]
        directions.append(Direction.VERTICAL)
        return directions

    if len(time_diffs) < 5:
        return Direction.ANY

    baseline_avg = np.average(time_diffs[0:-2])
    baseline_std = np.std(time_diffs[0:-2])
    cutoff = baseline_avg + 3 * baseline_std

    if (time_diffs[-1] < cutoff) and (time_diffs[-2] >= cutoff and time_diffs[-2] < CHANGE_DIR_MAX_THRESHOLD):
        directions = [Direction.HORIZONTAL for _ in range(len(move_times) - 2)]
        directions.extend([Direction.VERTICAL, Direction.ANY])
        return directions

    return Direction.ANY


def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time
