import numpy as np
import matplotlib.pyplot as plt
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


def perform_match_spectrograms(first_spectrogram: np.ndarray, second_spectrogram: np.ndarray, mask_threshold: float, should_plot: bool) -> float:
    first_time_steps = first_spectrogram.shape[1]
    second_time_steps = second_spectrogram.shape[1]
    time_diff = abs(first_time_steps - second_time_steps)

    if (first_time_steps < second_time_steps):
        return perform_match_spectrograms(second_spectrogram, first_spectrogram, mask_threshold, should_plot)

    time_diff = first_time_steps - second_time_steps

    # Get the mask for the smaller (second) spectrogram
    second_mask = (second_spectrogram >= mask_threshold).astype(second_spectrogram.dtype)
    #masked_second = mask * second_spectrogram

    #num_nonzero = np.sum(mask)

    # Track the spectrogram similarity
    similarity = 0.0

    #print('Target time steps: {}, Ref time steps: {}'.format(target_time_steps, ref_time_steps))

    for offset in range(time_diff + 1):
        clipped_first = first_spectrogram[:, offset:(offset + second_time_steps)]

        first_mask = (clipped_first >= mask_threshold).astype(clipped_first.dtype)
        mask = np.clip(first_mask + second_mask, a_min=0.0, a_max=1.0)
        num_nonzero = np.sum(mask)

        masked_first = clipped_first * mask
        masked_second = second_spectrogram * mask
        diff = np.abs(masked_first - masked_second)
        dist = max(np.sum(diff / num_nonzero), SMALL_NUMBER)
        similarity = max(similarity, 1.0 / dist)

        if should_plot:
            print('Offset: {}, Similarity: {:.4f}'.format(offset, similarity))
            fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
            ax0.imshow(masked_first, cmap='gray_r')
            ax1.imshow(masked_second, cmap='gray_r')
            ax2.imshow(np.abs(masked_first - masked_second), cmap='gray_r')

            ax0.set_title('First')
            ax1.set_title('Second')
            ax2.set_title('Difference')
        
            plt.show()

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


def get_sound_instances(spect: np.ndarray, forward_factor: float, backward_factor: float, peak_height: float, peak_distance: int, peak_prominence: float, smooth_window_size: int, tolerance: float, merge_peak_factor: float, max_merge_height: float) -> Tuple[List[int], List[int], np.ndarray]:
    # First, normalize the energy values for each frequency
    mean_energy = np.mean(spect, axis=-1, keepdims=True)
    std_energy = np.std(spect, axis=-1, keepdims=True)
    max_energy = np.max((spect - mean_energy) / std_energy, axis=0)  # [T]

    if smooth_window_size > 1:
        smooth_filter = np.ones(shape=(smooth_window_size, ), dtype=max_energy.dtype) / float(smooth_window_size)
        max_energy = convolve(max_energy, smooth_filter, mode='full')  # [T]

    peak_ranges: Set[Tuple[int, int, int]] = set()

    peak_times, peak_properties = find_peaks(max_energy, height=peak_height, distance=peak_distance, prominence=(peak_prominence, None))
    peak_heights = peak_properties['peak_heights']
    prev_end = 1

    for peak_time, peak_height in zip(peak_times, peak_heights):
        if is_in_peak_range(peak_time, peak_ranges):
            continue

        forward_peak_threshold = peak_height * forward_factor
        backward_peak_threshold = peak_height * backward_factor

        if (peak_time >= 16710) and (peak_time <= 16750):
            print('Peak Time: {}, Forward Threshold: {:.5f}, Backward Threshold: {:.5f}'.format(peak_time, forward_peak_threshold, backward_peak_threshold))

        # Get the start and end point
        start_time = peak_time
        while (start_time > prev_end) and (((max_energy[start_time] + tolerance) > max_energy[start_time - 1]) or (max_energy[start_time] > backward_peak_threshold)):
            start_time -= 1

        start_time += 1  # Adjust for going beyond the threshold

        end_time = peak_time
        while (end_time < (len(max_energy) - 1)) and (((max_energy[end_time] + tolerance) > max_energy[end_time + 1]) or (max_energy[end_time] > forward_peak_threshold)):
            end_time += 1

        # Add the range to the result set
        if (end_time - start_time) >= 3:
            peak_ranges.add((start_time, end_time, peak_height))
            prev_end = end_time

    # Sort the ranges by start time
    peak_ranges_sorted = list(sorted(peak_ranges, key=lambda t: t[0]))

    if len(peak_ranges_sorted) == 0:
        return [], []

    peak_ranges_dedup: List[Tuple[int, int]] = [peak_ranges_sorted[0]]

    for idx in range(1, len(peak_ranges_sorted)):
        curr_start, curr_end, curr_height = peak_ranges_sorted[idx]
        prev_start, prev_end, prev_height = peak_ranges_dedup[-1]

        #if (prev_start >= 16700) and (prev_start < 16750):
        #    print('Prev End: {}, Curr Start: {}, Prev Height: {}, Curr Height: {}, Factor: {}, {}'.format(prev_end, curr_start, prev_height, curr_height, curr_height / prev_height, prev_height / curr_height))

        if ((prev_height < max_merge_height) or (curr_height < max_merge_height)) and (((curr_height / prev_height) >= merge_peak_factor) or ((prev_height / curr_height) >= merge_peak_factor)) and ((curr_start - prev_end) <= 20):
            peak_ranges_dedup.pop(-1)
            peak_ranges_dedup.append((prev_start, curr_end, max(prev_height, curr_height)))
        else:
            peak_ranges_dedup.append((curr_start, curr_end, curr_height))

    start_times = [t[0] for t in peak_ranges_dedup]
    end_times = [t[1] for t in peak_ranges_dedup]

    fig, ax = plt.subplots()
    ax.plot(list(range(len(max_energy))), max_energy)
    ax.scatter(peak_times, peak_heights, marker='o', color='green')
    
    for t in start_times:
        ax.axvline(t, color='orange')

    for t in end_times:
        ax.axvline(t, color='red')

    plt.show()

    return start_times, end_times, max_energy


def extract_move_directions(move_times: List[int]) -> Union[List[Direction], Direction]:
    # If we have a run of 4+ times of <= 30 apart, then we consider these to be all HORIZONTAL
    # and the remainder are ANY. This design comes from the keyboard layout.
    min_run_length = 4
    diff_threshold = 30

    if (len(move_times) < min_run_length):
        return Direction.ANY

    # Start with everything being direction 'ANY'
    move_directions: List[Direction] = [Direction.ANY for _ in range(len(move_times))]

    for start_idx in range(len(move_times) - min_run_length + 1):
        # Get the moves for this sequence
        move_slice = move_times[start_idx:(start_idx + min_run_length)]
        diffs = [move_slice[i] - move_slice[i - 1] for i in range(1, len(move_slice))]
        assert len(diffs) == (min_run_length - 1), 'Expected {} diffs, found {}'.format(min_run_length - 1, len(diffs))

        if all([(diff < diff_threshold) for diff in diffs]):
            for idx in range(start_idx, start_idx + min_run_length):
                move_directions[idx] = Direction.HORIZONTAL

    assert len(move_times) == len(move_directions), 'Found a different # of direction ({}) than moves ({})'.format(len(move_directions), len(move_times))

    if all([direction == Direction.ANY for direction in move_directions]):
        return Direction.ANY

    return move_directions


def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time
