import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks
from typing import List, Tuple, Set

from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations
from smarttvleakage.audio.move_extractor import moving_window_similarity, create_spectrogram, clip_spectrogram, binarize_spectrogram
from smarttvleakage.utils.file_utils import read_pickle_gz


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    max_value, min_value = np.max(spectrogram), np.min(spectrogram)
    return (spectrogram - min_value) / (max_value - min_value)


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



def get_sound_instances(max_energy: np.ndarray, peak_height: float, threshold: float) -> Tuple[List[int], List[int]]:
    result_points: Set[Tuple[int, int]] = set()

    peak_times, _ = find_peaks(max_energy, height=peak_height, distance=2)

    for peak_time in peak_times:
        # Get the start and end point
        start_time = peak_time
        while (start_time > 0) and (max_energy[start_time] > threshold):
            start_time -= 1

        end_time = peak_time
        while (end_time < len(max_energy)) and (max_energy[end_time] > threshold):
            end_time += 1

        # Add the range to the result set
        result_points.add((start_time, end_time))

    # Sort the results by start time
    results_sorted = list(sorted(result_points, key=lambda t: t[0]))

    start_times = [t[0] for t in results_sorted]
    end_times = [t[1] for t in results_sorted]

    return start_times, end_times


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
    range_max = 30

    target = create_spectrogram(audio_signal)[range_min:range_max]  # [F, T0]
    known = create_spectrogram(known_signal)[range_min:range_max]  # [F, T1]

    if args.plot_similarity:
        sim_scores = similarity_scores(target_spectrogram=target, known_spectrogram=known)

        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        
        ax0.plot(list(range(len(audio_signal))), audio_signal)
        ax1.plot(list(range(len(sim_scores))), sim_scores)

        plt.show()
    elif args.plot_constellations:
        energy = np.max(known, axis=0)
        known_starts, known_ends = get_sound_instances(energy, peak_height=-50, threshold=-55)
        known = known[:, known_starts[0]:known_ends[0]]

        normalized = normalize_spectrogram(known)
        times, freq = compute_constellation_map(spectrogram=known,
                                                freq_delta=5,
                                                time_delta=3,
                                                threshold=-60)

        print('Times: {}, Freq: {}'.format(times, freq))

        fig, ax = plt.subplots()
        ax.imshow(known, cmap='gray_r')
        ax.scatter(times, freq)
        plt.show()
    elif args.plot_max_energy:
        max_energy = np.max(target, axis=0)  # [T0]

        start_times, end_times = get_sound_instances(max_energy, peak_height=-50, threshold=-55)

        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

        #ax0.plot(list(range(len(audio_signal))), audio_signal)
        ax0.imshow(target, cmap='gray_r')
        ax1.plot(list(range(max_energy.shape[0])), max_energy)

        for t in start_times:
            ax1.axvline(t, color='orange')

        for t in end_times:
            ax1.axvline(t, color='red')
        
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
