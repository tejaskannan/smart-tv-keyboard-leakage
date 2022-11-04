import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os.path
import math
from argparse import ArgumentParser
from collections import namedtuple
from itertools import permutations
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks
from typing import List, Tuple, Set

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio import MatchConfig
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations, match_constellation_window
from smarttvleakage.audio.move_extractor import moving_window_similarity, create_spectrogram, clip_spectrogram, binarize_spectrogram
from smarttvleakage.utils.constants import BIG_NUMBER
from smarttvleakage.utils.file_utils import read_pickle_gz, read_json


SoundMatch = namedtuple('SoundMatch', ['start_time', 'end_time', 'sound'])
MatchProperties = namedtuple('MatchProperties', ['time_tol', 'freq_tol', 'time_delta', 'freq_delta', 'threshold'])

ALTER_FACTOR = 1.25


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    avg_value, std_value = np.average(spectrogram), np.std(spectrogram)
    return (spectrogram - avg_value) / std_value


def binarize_and_clip(spectrogram: np.ndarray, match_configs: List[MatchConfig]) -> np.ndarray:
    # Create the binarized spectro-gram
    binarized = np.zeros_like(spectrogram, dtype=int)

    for config in match_configs:
        min_freq, max_freq = config.min_freq, config.max_freq
        clipped = spectrogram[min_freq:max_freq, :]
        binarized[min_freq:max_freq, :] += np.logical_and(clipped >= config.min_threshold, clipped <= config.max_threshold).astype(int)

    binarized = np.clip(binarized, a_min=0, a_max=1)

    # Get the clip ranges
    min_freq = min(map(lambda config: config.min_freq, match_configs))
    max_freq = max(map(lambda config: config.max_freq, match_configs))

    return binarized[min_freq:max_freq, :]


def moving_window_distances(target_spectrogram: np.ndarray, known_spectrogram: np.ndarray) -> List[float]:
    if target_spectrogram.shape[1] < known_spectrogram.shape[1]:
        return moving_window_distances(known_spectrogram, target_spectrogram)

    window_size = known_spectrogram.shape[1]
    distances: List[float] = []

    for idx in range(0, target_spectrogram.shape[1] - window_size + 1):
        target_segment = target_spectrogram[:, idx:idx+window_size]
        dist = np.average(np.abs(target_segment - known_spectrogram))
        distances.append(dist)

    return distances


def perform_match_constellations(target_times: np.ndarray, target_freq: np.ndarray, target_peaks: np.ndarray, ref_times: np.ndarray, ref_freq: np.ndarray, ref_peaks: np.ndarray):
    time_diffs = np.expand_dims(ref_times, axis=-1) - np.expand_dims(target_times, axis=0)  # [N, M]
    freq_diffs = np.expand_dims(ref_freq, axis=-1) - np.expand_dims(target_freq, axis=0)  # [N, M]
    distances = np.abs(time_diffs) + np.abs(freq_diffs)  # [N, M]

    # Match from the larger to the smaller set, ensure that each in the smaller set
    # has AT LEAST one match. To do this, we first run a bipartite matching from the smaller to the
    # larger set. Then, we use a greedy approach on the remaining entries to get the full match.
    # The first portion of this algorithm will ensure that we have a full matching set, while
    # the second will ensure we use all of the points on both sides
    bipartite_matrix = sp.csr_matrix(distances + 1)
    ref_indices, target_indices = sp.csgraph.min_weight_full_bipartite_matching(bipartite_matrix, maximize=False)

    # Start with the total weight of the bipartite matching
    total_dist = np.sum(distances[ref_indices, target_indices])

    # Account for point imbalance by greedily matching the remaining points (these can be duplicates)
    if len(target_times) < len(ref_times):
        for ref_idx in filter(lambda i: (i not in ref_indices), range(len(ref_times))):
            total_dist += np.min(distances[ref_idx])
    elif len(ref_times) < len(target_times):
        for target_idx in filter(lambda i: (i not in target_indices), range(len(target_times))):
            total_dist += np.min(distances[:, target_idx])

    # Return the average matching distances
    return total_dist / max(len(ref_times), len(target_times))


def perform_match(first_spectrogram: np.ndarray, second_spectrogram: np.ndarray) -> float:
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
        #distance = np.average(np.abs(padded_spectrogram - first_spectrogram))
        #similarity = 1.0 / distance

        best_similarity = max(similarity, best_similarity)
        pad_after -= 1

    return best_similarity


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


class AppleTVMatcher:

    def __init__(self, sound_folder: str):
        self._range_min = 5
        self._range_max = 50

        self._sound_threshold = -53.5
        self._peak_threshold = -50
        self._similarity_threshold = 0.15

        self._spectrograms: Dict[str, np.ndarray] = dict()

        self._match_configs = read_json(os.path.join(sound_folder, 'config.json'))

        for sound in sounds.APPLETV_SOUNDS:
            # Read in the saved sound
            path = os.path.join(sound_folder, '{}.pkl.gz'.format(sound))
            audio = read_pickle_gz(path)[:, 0]  # Extract channel 0

            # Compute the spectrogram, clip the sound, and normalize the result
            spectrogram = create_spectrogram(audio)[self._range_min:self._range_max]

            sound_starts, sound_ends = get_sound_instances(max_energy=np.max(spectrogram, axis=0),
                                                           peak_height=self._peak_threshold,
                                                           threshold=self._sound_threshold)
            sound_start, sound_end = sound_starts[0], sound_ends[0]

            #normalized_spectrogram = normalize_spectrogram(spectrogram[:, sound_start:sound_end])
            #binarized_spectrogram = binarize_and_clip(spectrogram[:, sound_start:sound_end], sound_configs)

            self._spectrograms[sound] = spectrogram[:, sound_start:sound_end]

    @property
    def range_min(self) -> int:
        return self._range_min

    @property
    def range_max(self) -> int:
        return self._range_max

    def get_candidate_sounds(self, segment: np.ndarray) -> List[str]:
        candidates = [sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_SYSTEM_MOVE, sounds.APPLETV_KEYBOARD_DELETE, sounds.APPLETV_TOOLBAR_MOVE]

        best_move_dist = BIG_NUMBER
        best_move_sound = None
        move_sounds = [sounds.APPLETV_KEYBOARD_MOVE, sounds.APPLETV_KEYBOARD_DOUBLE_MOVE, sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE, sounds.APPLETV_KEYBOARD_SCROLL_SIX]

        for move_sound in move_sounds:
            diff = abs(segment.shape[1] - self._spectrograms[move_sound].shape[1])

            if diff < best_move_dist:
                best_move_dist = diff
                best_move_sound = move_sound

        if best_move_sound is not None:
            candidates.append(best_move_sound)

        return candidates

    def find_sounds(self, target_audio: np.ndarray) -> List[SoundMatch]:
        assert len(target_audio.shape) == 1, 'Must provide a 1d array of audio values'

        # Compute the spectrogram of the target audio signal
        target_spectrogram = create_spectrogram(target_audio)[self._range_min:self._range_max]  # [F, T]

        # Find instances of any sounds (for later matching)
        max_energy = np.max(target_spectrogram, axis=0)
        start_times, end_times = get_sound_instances(max_energy=max_energy,
                                                     peak_height=self._peak_threshold,
                                                     threshold=self._sound_threshold)

        results: List[SoundMatch] = []

        for start_time, end_time in zip(start_times, end_times):
            # Clip the start and end time
            target_segment = target_spectrogram[:, start_time:end_time]
            #normalized_target_segment = normalize_spectrogram(target_segment)
            #normalized_target_segment = maximum_filter(normalized_target_segment, size=(3, 3))

            # Get a list of possible sounds to match against the identified noise
            #candidate_sounds = self.get_candidate_sounds(segment=target_segment)

            # Find the closest known sound to the observed noise
            best_sim = 0.0
            best_sound = None

            for sound in sounds.APPLETV_SOUNDS:
                # Match the sound on the spectrograms
                ref_spectrogram = self._spectrograms[sound]  # [F, T0]
                #binarized_target = binarize_and_clip(target_segment, self._match_configs[sound])

                properties = self._match_configs[sound]
                #properties = MatchProperties(time_delta=3, freq_delta=5, time_tol=2, freq_tol=2, threshold=-60)

                # Binarize both spectrograms
                #ref_spectrogram_binary = (ref_spectrogram > properties.threshold).astype(int)
                #target_spectrogram_binary = (normalized_target_segment > properties.threshold).astype(int)

                # Compute the match
                #similarity = perform_match(first_spectrogram=binarized_target,
                #                           second_spectrogram=ref_spectrogram)

                # Compute the constellation maps
                target_times, target_freq = compute_constellation_map(spectrogram=target_segment,
                                                                      freq_delta=properties['freq_delta'],
                                                                      time_delta=properties['time_delta'],
                                                                      threshold=properties['threshold'])

                target_peaks = [target_segment[freq, time] for time, freq in zip(target_times, target_freq)]

                ref_times, ref_freq = compute_constellation_map(spectrogram=ref_spectrogram,
                                                                 freq_delta=properties['freq_delta'],
                                                                 time_delta=properties['time_delta'],
                                                                 threshold=properties['threshold'])

                ref_peaks = [self._spectrograms[sound][freq, time] for time, freq in zip(ref_times, ref_freq)]

                #if sound in (sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE, sounds.APPLETV_KEYBOARD_MOVE, sounds.APPLETV_KEYBOARD_SCROLL_SIX):
                #    print('Sound: {}'.format(sound))

                #    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
                #    ax0.imshow(ref_spectrogram, cmap='gray_r')
                #    ax0.scatter(ref_times, ref_freq, marker='o', color='red')

                #    ax1.imshow(target_segment, cmap='gray_r')
                #    ax1.scatter(target_times, target_freq, marker='o', color='red')

                #    ax0.set_title('Reference')
                #    ax1.set_title('Target')

                #    plt.show()
                #    plt.close()

                shifted_target_times = target_times - np.min(target_times)
                shifted_ref_times = ref_times - np.min(ref_times)

                avg_dist = perform_match_constellations(target_times=shifted_target_times,
                                         target_freq=target_freq,
                                         target_peaks=np.array(target_peaks),
                                         ref_times=shifted_ref_times,
                                         ref_freq=ref_freq,
                                         ref_peaks=np.array(ref_peaks))
                similarity = (1.0 / avg_dist)

                print('Sound: {}, Similarity: {}'.format(sound, similarity))

                if similarity > best_sim:
                    best_sim = similarity
                    best_sound = sound

                #for match_start_time in range(self._spectrograms[sound].shape[1]):
                #    #print('Target Times: {}'.format(shifted_target_times))

                #    match_score = match_constellation_window(target_times=target_times,
                #                                             target_freq=target_freq,
                #                                             ref_times=ref_times,
                #                                             ref_freq=ref_freq,
                #                                             time_tol=properties.time_tol,
                #                                             freq_tol=properties.freq_tol,
                #                                             start_time=start_time)
                #    match_scores.append(match_score)
                #    shifted_target_times += 1

                #max_sim = max(match_scores)

                #print('Sound: {}, Sim Score: {:.4f}'.format(sound, best_sim))

                #if max_sim > best_sim:
                #    best_sim = max_sim
                #    best_sound = sound

            print('Best Sim: {:.5f}, Best sound: {}'.format(best_sim, best_sound))
            print('Sound Shape: {}'.format(target_segment.shape))
            print('==========')

            if (best_sound is not None) and (best_sim >= self._similarity_threshold):
                match = SoundMatch(start_time=start_time,
                                   end_time=end_time,
                                   sound=best_sound)
                results.append(match)

        return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--sound-folder', type=str, required=True)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()[:, 0]

    matcher = AppleTVMatcher(args.sound_folder)
    matches = matcher.find_sounds(audio_signal)

    spectrogram = create_spectrogram(audio_signal)

    for match in matches:
        print('{} - {}: {}'.format(match.start_time, match.end_time, match.sound))

    fig, ax = plt.subplots()
    ax.imshow(spectrogram, cmap='gray_r')
    plt.show()
