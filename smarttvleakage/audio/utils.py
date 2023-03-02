import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks, convolve
from scipy.ndimage import maximum_filter1d
from typing import List, Tuple, Set, Union

from smarttvleakage.utils.constants import BIG_NUMBER, SMALL_NUMBER, Direction
from smarttvleakage.audio.constellations import compute_constellation_map


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


def get_bounding_times(mask: np.ndarray) -> Tuple[int, int]:
    assert len(mask.shape) == 2, 'Must provide a 2d binary matrix'

    max_values = np.max(mask, axis=0)  # [T]

    # Get the time of the first `1`
    start_time = 0
    for t in range(len(max_values)):
        if abs(max_values[t] - 1.0) < SMALL_NUMBER:
            start_time = t
            break

    # Get the time of the last `1`
    end_time = len(max_values)
    for t in reversed(range(len(max_values))):
        if (abs(max_values[t] - 1.0)) < SMALL_NUMBER:
            end_time = t + 1
            break

    return start_time, end_time

def perform_match_spectrograms(first_spectrogram: np.ndarray, second_spectrogram: np.ndarray, mask_threshold: float, should_plot: bool) -> float:
    """
    Measures the similarity between the two (normalized) spectrograms using an L1 distance where all
    values below the given threshold are set to zero (to avoid matching `random` noise).

    Args:
        first_spectrogram: A [F, T0] spectrogram
        second_spectrogram: A [F, T1] spectrogram
        mask_threshold: The value used to mask out noise
        should_plot: Whether to plot debugging information
    Returns:
        A similarity score >= 0.0
    """
    assert first_spectrogram.shape[0] == second_spectrogram.shape[0], 'Must provide spectrograms with the same number of frequence elements'

    # First, compute the masks for both spectrograms
    first_mask = (first_spectrogram >= mask_threshold).astype(first_spectrogram.dtype)
    second_mask = (second_spectrogram >= mask_threshold).astype(second_spectrogram.dtype)

    # Get the first and last time elements with unmasked values. We will match these `clipped` spectrograms to account
    # for time shifting.
    first_start, first_end = get_bounding_times(first_mask)
    second_start, second_end = get_bounding_times(second_mask)

    first_spectrogram_clipped = first_spectrogram[:, first_start:first_end]
    second_spectrogram_clipped = second_spectrogram[:, second_start:second_end]

    first_time_steps = first_spectrogram_clipped.shape[1]
    second_time_steps = second_spectrogram_clipped.shape[1]
    time_diff = abs(first_time_steps - second_time_steps)

    # Make it so the first spectrogram is always the longer of the two
    if (first_time_steps < second_time_steps):
        return perform_match_spectrograms(second_spectrogram, first_spectrogram, mask_threshold, should_plot)

    time_diff = first_time_steps - second_time_steps

    # Get the mask for the smaller (second) spectrogram
    second_mask = (second_spectrogram_clipped >= mask_threshold).astype(second_spectrogram.dtype)

    if should_plot:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(first_spectrogram_clipped, cmap='gray_r')
        ax1.imshow(second_spectrogram_clipped, cmap='gray_r')
        plt.show()

    # Track the spectrogram similarity
    similarity = 0.0

    for offset in range(time_diff + 1):
        first_segment = first_spectrogram_clipped[:, offset:(offset + second_time_steps)]

        # Compute the mask using an aggregate of the two spectrograms
        first_mask = (first_segment >= mask_threshold).astype(first_segment.dtype)
        mask = np.clip(first_mask + second_mask, a_min=0.0, a_max=1.0)
        num_nonzero = np.sum(mask)

        masked_first = first_segment * mask
        masked_second = second_spectrogram_clipped * mask
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


def dedup_samsung_move_delete(normalized_spectrogram: np.ndarray, freq_delta: int, time_delta: int, peak_threshold: float, mask_threshold: float, should_plot: bool) -> bool:
    # Compute the constellation map of the given spectrogram
    time_peaks, freq_peaks = compute_constellation_map(normalized_spectrogram, freq_delta=freq_delta, time_delta=time_delta, threshold=peak_threshold)

    # Get the highest peak time
    peak_heights = [normalized_spectrogram[freq, time] for freq, time in zip(freq_peaks, time_peaks)]
    highest_idx = np.argmax(peak_heights)
    highest_time, highest_freq = time_peaks[highest_idx], freq_peaks[highest_idx]

    # Mask the spectrogram based on the threshold
    binary_spectrogram = (normalized_spectrogram > mask_threshold).astype(normalized_spectrogram.dtype)

    # Get the number of `high` elements on either side of the highest peak time
    min_freq, max_freq = max(highest_freq - freq_delta, 0), (highest_freq + freq_delta + 1)
    num_before = np.sum(binary_spectrogram[min_freq:max_freq, 0:highest_time])
    num_after = np.sum(binary_spectrogram[min_freq:max_freq, (highest_time + 1):])

    if should_plot:
        print('Before: {}, After: {}'.format(num_before, num_after))

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(normalized_spectrogram, cmap='gray_r')
        ax0.scatter(time_peaks, freq_peaks, marker='o', color='red')

        ax1.imshow(binary_spectrogram[min_freq:max_freq, :], cmap='gray_r')

        plt.show()

    return num_after > num_before


def get_move_time_length(move_spectrogram: np.ndarray) -> int:
    freq_amplitude = move_spectrogram[4, :]  # Moves have a maximal amplitude at index 4
    cutoff = 0.85

    start_time = 0
    end_time = 0
    total_length = 0
    time_cutoff = 10

    while (start_time < freq_amplitude.shape[0]):
        while (start_time < freq_amplitude.shape[0]) and (freq_amplitude[start_time] < cutoff):
            start_time += 1

        end_time = start_time + 1
        while (end_time < freq_amplitude.shape[0]) and (freq_amplitude[end_time] > cutoff):
            end_time += 1

        if (end_time - start_time) >= time_cutoff:
            total_length += (end_time - start_time)

        start_time = end_time + 1

    return total_length


def count_cluster_size(segment_spectrogram: np.ndarray, start_time: int) -> int:
    raw_time_peaks, raw_freq_peaks = compute_constellation_map(segment_spectrogram, freq_delta=1, time_delta=15, threshold=-57)
    time_peaks = [t for t, freq in zip(raw_time_peaks, raw_freq_peaks) if (freq == 4)]
    freq_peaks = [freq for t, freq in zip(raw_time_peaks, raw_freq_peaks) if (freq == 4)]  # For debugging alone

    time_distance = 12

    peak_idx = 0
    num_clusters = 0
    clusters: List[List[int]] = []

    while (peak_idx < len(time_peaks)):
        current_peak_time = time_peaks[peak_idx]

        if (peak_idx < (len(time_peaks) - 1)) and ((time_peaks[peak_idx + 1] - current_peak_time) < time_distance):
            clusters.append([current_peak_time, time_peaks[peak_idx + 1]])
            peak_idx += 1
        else:
            clusters.append([current_peak_time])

        peak_idx += 1
        num_clusters += 1

    #if (start_time >= 77750):
    #    colors = ['red', 'green', 'orange']

    #    print('Time: {}, # Clusters: {}'.format(start_time, num_clusters))
    #    fig, ax = plt.subplots()
    #    #ax0.plot(list(range(len(max_filtered))), max_filtered)
    #    ax.imshow(segment_spectrogram, cmap='gray_r')
    #    ax.scatter(time_peaks, freq_peaks, color='red', marker='o')

    #    for cluster_idx, cluster in enumerate(clusters):
    #        for cluster_time in cluster:
    #            ax.axvline(cluster_time, color=colors[cluster_idx % len(colors)])

    #    plt.show()

    return max(num_clusters, 1)


def get_directions_appletv(move_times: List[int]) -> List[Direction]:
    directions: List[Direction] = []

    for idx in range(len(move_times)):
        prev_run_length = 0
        prev_idx = idx - 1
        while (prev_idx >= 0) and (move_times[prev_idx] == move_times[idx]):
            prev_run_length += 1
            prev_idx -= 1

        next_run_length = 0
        next_idx = idx + 1
        while (next_idx < len(move_times)) and (move_times[next_idx] == move_times[idx]):
            next_run_length += 1
            next_idx += 1

        run_length = prev_run_length + next_run_length + 1

        if (run_length >= 4) and (prev_run_length > 0):
            directions.append(Direction.HORIZONTAL)
        else:
            directions.append(Direction.ANY)

    return directions


def get_sound_instances_appletv(spect: np.ndarray, smooth_window_size: int) -> Tuple[List[int], List[int], np.ndarray, List[int]]:
     # First, normalize the energy values for each frequency
    mean_energy = np.mean(spect, axis=-1, keepdims=True)
    std_energy = np.std(spect, axis=-1, keepdims=True)
    max_energy = np.max((spect - mean_energy) / std_energy, axis=0)  # [T]

    if smooth_window_size > 1:
        smooth_filter = np.ones(shape=(smooth_window_size, ), dtype=max_energy.dtype) / float(smooth_window_size)
        max_energy = convolve(max_energy, smooth_filter, mode='full')  # [T]

    # Get the cutoff points for which to detect possible sounds
    median_normalized_energy = np.median(max_energy)
    iqr = np.percentile(max_energy, 75) - np.percentile(max_energy, 25)
    peak_cutoff = median_normalized_energy + 2.5 * iqr
    sound_cutoff = median_normalized_energy + 0.15 * iqr

    # Create the time ranges for each sound based on the normalized energy values
    start_times: List[int] = []
    end_times: List[int] = []
    cluster_sizes: List[int] = []
    min_time_length = 10  # Filter out spurious small peaks
    buffer_time = 1
    valley_factor = 0.5
    window_size = 10

    start_time = None
    max_in_between = 0

    for time, energy in enumerate(max_energy):
        if (start_time is None) and (energy >= sound_cutoff):
            start_time = time
            max_in_between = energy
        elif (start_time is not None) and ((energy <= sound_cutoff) or (is_local_min(time, max_energy, window_size) and (max_in_between * valley_factor >= energy))):
            cluster_size = 1

            if max_in_between > peak_cutoff:
                segment_spectrogram = spect[:, (start_time - buffer_time):(time + buffer_time)]
                cluster_size = count_cluster_size(segment_spectrogram, start_time=start_time)

            if ((time - start_time) >= min_time_length) and (max_in_between > peak_cutoff):
                start_times.append(start_time)
                end_times.append(time)
                cluster_sizes.append(cluster_size)

            max_in_between = 0
            start_time = None
        else:
            max_in_between = max(max_in_between, energy)

    # Plot the results for debugging    
    #fig, ax = plt.subplots()
    #ax.plot(list(range(len(max_energy))), max_energy)
    #ax.axhline(peak_cutoff, color='black')

    ##ax.scatter(peak_times, peak_heights, marker='o', color='green')
    #
    #for t in start_times:
    #    ax.axvline(t, color='orange')

    #for t in end_times:
    #    ax.axvline(t, color='red')

    #plt.show()

    return start_times, end_times, max_energy, cluster_sizes


def is_local_min(time: int, energy: np.ndarray, window_size: int) -> bool:
    if (time == 0) or (time == (len(energy) - 1)):
        return False

    start_time = max(time - window_size - 1, 0)
    end_time = min(time + window_size + 1, len(energy))
    min_value = float(np.min(energy[start_time:end_time]))

    return abs(min_value - energy[time]) < SMALL_NUMBER


def get_sound_instances_samsung(spect: np.ndarray, smooth_window_size: int) -> Tuple[List[int], List[int], np.ndarray, List[int]]:
     # First, normalize the energy values for each frequency
    mean_energy = np.mean(spect, axis=-1, keepdims=True)
    std_energy = np.std(spect, axis=-1, keepdims=True)
    max_energy = np.max((spect - mean_energy) / std_energy, axis=0)  # [T]

    if smooth_window_size > 1:
        smooth_filter = np.ones(shape=(smooth_window_size, ), dtype=max_energy.dtype) / float(smooth_window_size)
        max_energy = convolve(max_energy, smooth_filter, mode='full')  # [T]

    # Get the cutoff points for which to detect possible sounds
    median_normalized_energy = np.median(max_energy)
    iqr = np.percentile(max_energy, 75) - np.percentile(max_energy, 25)
    peak_cutoff = median_normalized_energy + 0.9 * iqr
    sound_cutoff = median_normalized_energy + 0.15 * iqr

    print('Peak Cutoff: {:.5f}, Sound Cutoff: {:.5f}'.format(peak_cutoff, sound_cutoff))

    # Create the time ranges for each sound based on the normalized energy values
    start_times: List[int] = []
    end_times: List[int] = []
    cluster_sizes: List[int] = []
    min_time_length = 10  # Filter out spurious small peaks
    buffer_time = 1
    valley_factor = 0.7
    window_size = 5

    start_time = None
    max_in_between = 0

    for time, energy in enumerate(max_energy):
        if (start_time is None) and (energy >= sound_cutoff):
            start_time = time
            max_in_between = energy
        elif (start_time is not None) and ((energy <= sound_cutoff) or (is_local_min(time, max_energy, window_size) and (max_in_between * valley_factor >= energy))):
            cluster_size = 1

            if max_in_between > peak_cutoff:
                segment_spectrogram = spect[:, (start_time - buffer_time):(time + buffer_time)]
                cluster_size = count_cluster_size(segment_spectrogram, start_time=start_time)

            if ((time - start_time) >= min_time_length) and (max_in_between > peak_cutoff):
                start_times.append(start_time)
                end_times.append(time)
                cluster_sizes.append(cluster_size)

            max_in_between = 0
            start_time = None
        else:
            max_in_between = max(max_in_between, energy)

    # Plot the results for debugging    
    #fig, ax = plt.subplots()
    #ax.plot(list(range(len(max_energy))), max_energy)
    #ax.axhline(peak_cutoff, color='black')

    ##ax.scatter(peak_times, peak_heights, marker='o', color='green')
    #
    #for t in start_times:
    #    ax.axvline(t, color='orange')

    #for t in end_times:
    #    ax.axvline(t, color='red')

    #plt.show()

    return start_times, end_times, max_energy, cluster_sizes


def get_num_scrolls(move_times: List[int], cutoff_size: int) -> int:
    num_scrolls = 0
    cluster_size = 0

    for idx in range(len(move_times) - 1):
        curr_time, next_time = move_times[idx], move_times[idx + 1]

        if (next_time == curr_time):
            cluster_size += 1
        else:
            num_scrolls += int(cluster_size >= cutoff_size)
            cluster_size = 0
    
    return num_scrolls


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
