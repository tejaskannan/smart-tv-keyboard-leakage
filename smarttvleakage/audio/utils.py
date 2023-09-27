import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, find_peaks, convolve
from scipy.ndimage import maximum_filter1d
from typing import List, Tuple, Set, Union

from smarttvleakage.utils.constants import BIG_NUMBER, SMALL_NUMBER, Direction
from smarttvleakage.audio.constellations import compute_constellation_map


def get_bounding_times(mask: np.ndarray) -> Tuple[int, int]:
    """
    Get the start and end times of values in the given
    binary matrix with values close to 1
    """
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
            fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
            ax0.imshow(masked_first, cmap='gray_r')
            ax1.imshow(masked_second, cmap='gray_r')
            ax2.imshow(np.abs(masked_first - masked_second), cmap='gray_r')

            ax0.set_title('First')
            ax1.set_title('Second')
            ax2.set_title('Difference')

            plt.show()

    return similarity


def count_cluster_size(segment_spectrogram: np.ndarray, start_time: int) -> int:
    """
    Counts the number of spectrogram peaks at a pre-set frequency. Useful
    for deduplicating scroll moves.
    """
    peak_threshold = -57
    target_freq = 4
    window_size = 1
    time_delta = 10
    time_distance = 13

    raw_time_peaks, raw_freq_peaks = compute_constellation_map(segment_spectrogram, freq_delta=1, time_delta=time_delta, threshold=peak_threshold)
    time_peaks = [t for t, freq in zip(raw_time_peaks, raw_freq_peaks) if (freq == target_freq) and (np.all(segment_spectrogram[target_freq, (t-window_size):(t+window_size+1)] >= peak_threshold))]
    freq_peaks = [freq for t, freq in zip(raw_time_peaks, raw_freq_peaks) if (freq == target_freq) and (np.all(segment_spectrogram[target_freq, (t-window_size):(t+window_size+1)] >= peak_threshold))]  # For debugging alone

    peak_idx = 0
    clusters: List[List[int]] = []

    while (peak_idx < len(time_peaks)):
        current_peak_time = time_peaks[peak_idx]
        cluster: List[int] = [current_peak_time]

        peak_idx += 1
        while (peak_idx < len(time_peaks)) and ((time_peaks[peak_idx] - current_peak_time) <= time_distance):
            cluster.append(time_peaks[peak_idx])
            peak_idx += 1

        clusters.append(cluster)

    num_clusters = len(clusters)

    return max(num_clusters, 1)


def get_directions_appletv(move_times: List[int]) -> List[Direction]:
    """
    Infers directions on Apple TV systems using long scrolls (which signal horizonal
    movements).
    """
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
    """
    Splits the given spectrogram into instances of candidate sounds
    on an Apple TV.
    """
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

    return start_times, end_times, max_energy, cluster_sizes


def is_local_min(time: int, energy: np.ndarray, window_size: int) -> bool:
    """
    Whether the energy value is a local minimum in the given window.
    """
    if (time == 0) or (time == (len(energy) - 1)):
        return False

    start_time = max(time - window_size - 1, 0)
    end_time = min(time + window_size + 1, len(energy))
    min_value = float(np.min(energy[start_time:end_time]))

    return abs(min_value - energy[time]) < SMALL_NUMBER


def get_sound_instances_samsung(spect: np.ndarray, smooth_window_size: int) -> Tuple[List[int], List[int], np.ndarray, List[int]]:
    """
    Splits a spectrogram from a Samsung TV into instances of candidate sounds.
    """
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

    return start_times, end_times, max_energy, cluster_sizes


def get_num_scrolls(move_times: List[int], cutoff_size: int) -> int:
    """
    Returns the number of rapid scrolls using timing information.
    """
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
    """
    Extracts move directions on a Samsung TV using movement timing.
    """
    # If we have a run of 4+ times of <= 30 apart, then we consider these to be all HORIZONTAL
    # and the remainder are ANY. This design comes from the keyboard layout.
    min_run_length = 4

    if (len(move_times) < min_run_length):
        return Direction.ANY

    move_diffs = [(move_times[idx] - move_times[idx - 1]) for idx in range(1, len(move_times))]
    diff_threshold = np.percentile(move_diffs, 50)

    # Start with everything being direction 'ANY'
    move_directions: List[Direction] = [Direction.ANY for _ in range(len(move_times))]

    for start_idx in range(len(move_times) - min_run_length + 1):
        # Get the moves for this sequence
        move_slice = move_times[start_idx:(start_idx + min_run_length)]
        diffs = [move_slice[i] - move_slice[i - 1] for i in range(1, len(move_slice))]
        assert len(diffs) == (min_run_length - 1), 'Expected {} diffs, found {}'.format(min_run_length - 1, len(diffs))

        if all([(diff <= diff_threshold) for diff in diffs]):
            for idx in range(start_idx, start_idx + min_run_length):
                move_directions[idx] = Direction.HORIZONTAL

    assert len(move_times) == len(move_directions), 'Found a different # of direction ({}) than moves ({})'.format(len(move_directions), len(move_times))

    if all([direction == Direction.ANY for direction in move_directions]):
        return Direction.ANY

    return move_directions


def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    """
    Creats a spectrogram from a given audio signal
    """
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time
