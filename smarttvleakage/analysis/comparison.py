import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import os.path
import math
from argparse import ArgumentParser
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks, convolve
from typing import List, Tuple, Set

#from smarttvleakage.audio import MatchConfig
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations
from smarttvleakage.audio.utils import create_spectrogram, perform_match_spectrograms
from smarttvleakage.utils.file_utils import read_pickle_gz
from smarttvleakage.utils.constants import BIG_NUMBER


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    #max_value, min_value = np.max(spectrogram), np.min(spectrogram)
    #return (spectrogram - min_value) / (max_value - min_value)
    avg_value, std_value = np.average(spectrogram), np.std(spectrogram)
    return (spectrogram - avg_value) / std_value


def similarity_scores(target_spectrogram: np.ndarray, known_spectrogram: np.ndarray) -> List[float]:
    window_size = known.shape[1]

    known_filtered = normalize_spectrogram(maximum_filter(known, size=(3, 3)))
    target_filtered = maximum_filter(target, size=(3, 3))

    similarity: List[float] = []

    for idx in range(0, target_filtered.shape[1] - window_size + 1):
        target_segment = target_filtered[:, idx:idx+window_size]
        normalized_segment = normalize_spectrogram(target_segment)

        dist = np.average(np.abs(normalized_segment - known_filtered))
        sim = 1.0 / dist

        similarity.append(sim)

    return similarity


def is_in_peak_range(peak_time: int, peak_ranges: Set[Tuple[int, int]]) -> bool:
    for peak_range in peak_ranges:
        if (peak_time >= peak_range[0]) and (peak_time <= peak_range[1]):
            return True

    return False


def get_sound_instances(max_energy: np.ndarray, peak_height: float, forward_threshold: float, backward_threshold: float, peak_prominence: float, min_time_gap: int) -> Tuple[List[int], List[int]]:
    peak_ranges: Set[Tuple[int, int]] = set()

    peak_times, peak_properties = find_peaks(max_energy, height=peak_height, distance=min_time_gap, prominence=(peak_prominence, None))
    peak_heights = peak_properties['peak_heights']
    prev_end = 0
    buffer_amt = 0.02

    for peak_time, peak_height in zip(peak_times, peak_heights):
        if is_in_peak_range(peak_time, peak_ranges):
            continue

        forward_peak_threshold = peak_height * forward_threshold
        backward_peak_threshold = peak_height * backward_threshold

        # Get the start and end point
        start_time = peak_time
        while (start_time > prev_end) and (((max_energy[start_time] + buffer_amt) > max_energy[start_time - 1]) or (max_energy[start_time] > backward_peak_threshold)):
            start_time -= 1

        start_time += 1  # Adjust for going beyond the threshold

        end_time = peak_time
        while (end_time < (len(max_energy) - 1)) and (((max_energy[end_time] + buffer_amt) > max_energy[end_time + 1]) or (max_energy[end_time] > forward_peak_threshold)):
            end_time += 1

            #if (end_time >= 450) and (end_time <= 460):
            #    print('Time: {}, Max Energy: {}, Next Max Energy: {}'.format(end_time, max_energy[end_time] + buffer_amt, max_energy[end_time + 1]))

        # Add the range to the result set
        peak_ranges.add((start_time, end_time, peak_height))
        prev_end = end_time

    # Sort the ranges by start time
    peak_ranges_sorted = list(sorted(peak_ranges, key=lambda t: t[0]))

    if len(peak_ranges_sorted) == 0:
        return [], [], [], []

    peak_ranges_dedup: List[Tuple[int, int]] = [(peak_ranges_sorted[0][0], peak_ranges_sorted[0][1])]

    for idx in range(1, len(peak_ranges_sorted)):
        curr_start, curr_end, curr_height = peak_ranges_sorted[idx]
        prev_start, prev_end, prev_height = peak_ranges_sorted[idx - 1]

        if (((curr_height / prev_height) >= 1.5) or ((prev_height / curr_height) >= 1.5)) and ((curr_start - prev_end) <= 20):
            peak_ranges_dedup.pop(-1)
            peak_ranges_dedup.append((prev_start, curr_end))
        else:
            peak_ranges_dedup.append((curr_start, curr_end))

    start_times = [t[0] for t in peak_ranges_dedup]
    end_times = [t[1] for t in peak_ranges_dedup]
    return start_times, end_times, peak_times, peak_heights


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--sound-file', type=str, required=True)
    parser.add_argument('--ranges', nargs='+', type=int)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--plot-similarity', action='store_true')
    parser.add_argument('--plot-max-energy', action='store_true')
    parser.add_argument('--plot-constellations', action='store_true')
    args = parser.parse_args()

    assert (args.ranges is None) or (len(args.ranges) == 2), 'When provided, ranges should have 2 numbers'

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()[:, 0]

    known_sound = read_pickle_gz(args.sound_file)
    known_signal = known_sound[:, 0]

    range_min = 5
    range_max = 50
    energy_start_freq = 0
    window_size = 8

    peak_height = 1.75
    forward_threshold = 0.9
    backward_threshold = 0.6

    target = create_spectrogram(audio_signal)[range_min:range_max]  # [F, T0]
    known = create_spectrogram(known_signal)[range_min:range_max]  # [F, T1]

    if args.plot_similarity:
        # Normalize the energy values for each freq component
        clipped_target = target[energy_start_freq:, :]
        clipped_target[clipped_target < -BIG_NUMBER] = 0.0
        avg_energy = np.mean(clipped_target, axis=-1, keepdims=True)  # [F, 1]
        std_energy = np.std(clipped_target, axis=-1, keepdims=True)  # [F, 1]

        normalized_energy = (clipped_target - avg_energy) / std_energy  # [F, T]
        max_energy = np.max(normalized_energy, axis=0)  # [T]

        #max_energy = np.max(target[energy_start_freq:, :], axis=0)  # [T0]
        smooth_filter = np.ones(shape=(window_size, ), dtype=max_energy.dtype) / window_size
        max_energy = convolve(max_energy, smooth_filter, mode='full')

        start_times, end_times, peak_times, peak_heights = get_sound_instances(max_energy,
                                                                               peak_height=peak_height,
                                                                               forward_threshold=forward_threshold,
                                                                               backward_threshold=backward_threshold,
                                                                               peak_prominence=0.1,
                                                                               min_time_gap=5)

        idx = 3
        threshold = 0.7
        start_time, end_time = start_times[idx], end_times[idx]

        clipped_target = target[:, start_time:end_time]
        max_target, min_target = np.max(clipped_target), np.min(clipped_target)
        normalized_target = (clipped_target - min_target) / (max_target - min_target)

        max_ref, min_ref = np.max(known), np.min(known)
        normalized_ref = (known - min_ref) / (max_ref - min_ref)

        similarity = perform_match_spectrograms(first_spectrogram=normalized_target,
                                                second_spectrogram=normalized_ref,
                                                mask_threshold=threshold)

        print('Final Similarity: {:.5f}'.format(similarity))

        #sim_scores = similarity_scores(target_spectrogram=target, known_spectrogram=known)

        #fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        #
        #ax0.plot(list(range(len(audio_signal))), audio_signal)
        #ax1.plot(list(range(len(sim_scores))), sim_scores)

        #plt.show()
    elif args.plot_constellations:
        # Normalize the energy values for each freq component
        clipped_known = known[:, :]
        clipped_known[clipped_known < -BIG_NUMBER] = 0.0
        avg_energy = np.mean(clipped_known, axis=-1, keepdims=True)  # [F, 1]
        std_energy = np.std(clipped_known, axis=-1, keepdims=True)  # [F, 1]

        normalized_energy = (clipped_known - avg_energy) / std_energy  # [F, T]
        max_energy = np.max(normalized_energy, axis=0)  # [T]

        #max_energy = np.max(target[energy_start_freq:, :], axis=0)  # [T0]
        smooth_filter = np.ones(shape=(window_size, ), dtype=max_energy.dtype) / window_size
        max_energy = convolve(max_energy, smooth_filter, mode='full')

        known_starts, known_ends, peak_times, peak_heights = get_sound_instances(max_energy,
                                                                                 peak_height=peak_height,
                                                                                 forward_threshold=forward_threshold,
                                                                                 backward_threshold=backward_threshold,
                                                                                 peak_prominence=0.1,
                                                                                 min_time_gap=5)

        max_val, min_val = np.max(known), np.min(known)
        normalized = (known - min_val) / (max_val - min_val)
        normalized = (normalized > 0.7).astype(normalized.dtype) * normalized

        if len(known_starts) > 0:
            normalized = normalized[:, known_starts[0]:known_ends[0]]

        times, freq = compute_constellation_map(spectrogram=normalized,
                                                freq_delta=5,
                                                time_delta=5,
                                                start_threshold=0.99,
                                                end_threshold=0.1)

        print('Times: {}, Freq: {}'.format(times, freq))

        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(normalized, cmap='gray_r')
        ax0.scatter(times, freq, color='red')

        ax1.plot(list(range(len(max_energy))), max_energy)
        ax1.scatter(peak_times, peak_heights, marker='o', color='red')
        #ax1.axvline(known_starts[0], color='orange')
        #ax1.axvline(known_ends[0], color='green')

        plt.show()
    elif args.plot_max_energy:
        # Normalize the energy values for each freq component
        clipped_target = target[energy_start_freq:, :]
        clipped_target[clipped_target < -BIG_NUMBER] = 0.0
        avg_energy = np.mean(clipped_target, axis=-1, keepdims=True)  # [F, 1]
        std_energy = np.std(clipped_target, axis=-1, keepdims=True)  # [F, 1]

        normalized_energy = (clipped_target - avg_energy) / std_energy  # [F, T]
        max_energy = np.max(normalized_energy, axis=0)  # [T]

        #max_energy = np.max(target[energy_start_freq:, :], axis=0)  # [T0]
        smooth_filter = np.ones(shape=(window_size, ), dtype=max_energy.dtype) / window_size
        max_energy = convolve(max_energy, smooth_filter, mode='full')

        start_times, end_times, peak_times, peak_heights = get_sound_instances(max_energy,
                                                                               peak_height=peak_height,
                                                                               forward_threshold=forward_threshold,
                                                                               backward_threshold=backward_threshold,
                                                                               peak_prominence=0.1,
                                                                               min_time_gap=5)


        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        
        ax0.plot(list(range(len(audio_signal))), audio_signal)
        ax0.imshow(target, cmap='gray_r')
        ax1.plot(list(range(max_energy.shape[0])), max_energy)
        ax1.scatter(peak_times, peak_heights, color='green', marker='o')

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Maximum Normalized Amplitude')
        ax1.set_title('Maximum Normalized Amplitude for Audio Recordings')

        for t in start_times:
            ax1.axvline(t, color='orange')

        for t in end_times:
            ax1.axvline(t, color='red')

        if args.output_path is not None:
            plt.savefig(args.output_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()
    elif args.ranges is None:
        fig, ax = plt.subplots()
        ax.imshow(target, cmap='gray_r')
        plt.show()
    else:
        start, end = args.ranges
        window_size = known.shape[1]

        known_filtered = normalize_spectrogram(maximum_filter(known, size=(3, 3)))
        target_filtered = maximum_filter(target, size=(3, 3))

        for idx in range(start, end + 1):
            target_segment = target_filtered[:, idx:idx+window_size]
            normalized_segment = normalize_spectrogram(target_segment)

            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
            ax0.imshow(normalized_segment, cmap='gray_r')
            ax1.imshow(known_filtered, cmap='gray_r')

            dist = np.average(np.abs(normalized_segment - known_filtered))
            print('Distance: {}'.format(dist))
            print('Avg Value: {}'.format(np.average(target_segment)))

            plt.show()
            plt.close()
