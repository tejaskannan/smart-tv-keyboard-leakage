import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import os.path
import math
from argparse import ArgumentParser
from collections import namedtuple
from itertools import permutations
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks
from typing import List, Tuple, Set

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations, match_constellation_window
from smarttvleakage.audio.move_extractor import moving_window_similarity, create_spectrogram, clip_spectrogram, binarize_spectrogram
from smarttvleakage.utils.constants import BIG_NUMBER
from smarttvleakage.utils.file_utils import read_pickle_gz


SoundMatch = namedtuple('SoundMatch', ['start_time', 'end_time', 'sound'])
MatchProperties = namedtuple('MatchProperties', ['time_tol', 'freq_tol', 'time_delta', 'freq_delta', 'threshold'])

ALTER_FACTOR = 1.0


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    #max_value, min_value = np.max(spectrogram), np.min(spectrogram)
    #return (spectrogram - min_value) / (max_value - min_value)

    avg_value, std_value = np.average(spectrogram), np.std(spectrogram)
    return (spectrogram - avg_value) / std_value


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


def perform_match_constellations(first_times: np.ndarray, first_freq: np.ndarray, first_peaks: np.ndarray, second_times: np.ndarray, second_freq: np.ndarray, second_peaks: np.ndarray):
    diff = abs(len(first_times) - len(second_times))

    if len(first_times) < len(second_times):
        peak_indices = np.argsort(second_peaks)
        indices_to_keep = peak_indices[diff:]

        second_times = second_times[indices_to_keep]
        second_freq = second_freq[indices_to_keep]
    elif len(second_times) < len(first_times):
        peak_indices = np.argsort(first_peaks)
        indices_to_keep = peak_indices[diff:]

        first_times = first_times[indices_to_keep]
        first_freq = first_freq[indices_to_keep]

    time_diffs = np.expand_dims(first_times, axis=-1) - np.expand_dims(second_times, axis=0)  # [N, M]
    freq_diffs = np.expand_dims(first_freq, axis=-1) - np.expand_dims(second_freq, axis=0)  # [N, M]
    distances = np.sqrt(np.square(time_diffs) + np.square(freq_diffs)) + ALTER_FACTOR * diff

    # Get the total number of unique matchings
    first_indices = list(range(len(first_times)))
    second_indices = list(range(len(second_times)))

    unique_matchings: List[List[Tuple[int, int]]] = []
    perms = permutations(first_indices, len(second_indices))

    for perm in perms:
        unique_matchings.append(list(zip(perm, second_indices)))

    # Find the matching with the lowest distance by brute force
    best_dist = BIG_NUMBER
    for matching in unique_matchings:
        total_dist = 0.0
        total_points = 0.0

        for match in matching:
            first_idx, second_idx = match
            total_dist += distances[first_idx, second_idx]
            total_points += 1.0
            
        avg_dist = total_dist / total_points
        if avg_dist < best_dist:
            best_dist = avg_dist

    return best_dist


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
        self._range_max = 35

        self._sound_threshold = -55
        self._peak_threshold = -50
        self._similarity_threshold = 0.15

        self._spectrograms: Dict[str, np.ndarray] = dict()

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

            #spectrogram = maximum_filter(spectrogram, size=(3, 3))

            normalized_spectrogram = normalize_spectrogram(spectrogram[:, sound_start:sound_end])
            #print('{}: {}'.format(sound, normalized_spectrogram.shape))

            #fig, ax = plt.subplots()
            #ax.imshow(normalized_spectrogram, cmap='gray_r')
            #plt.show()
            #plt.close()

            self._spectrograms[sound] = normalized_spectrogram

        self._match_properties = {
            sounds.APPLETV_KEYBOARD_SELECT: MatchProperties(time_tol=2, freq_tol=1, time_delta=5, freq_delta=5, threshold=0.8),
            sounds.APPLETV_KEYBOARD_MOVE: MatchProperties(time_tol=2, freq_tol=2, time_delta=5, freq_delta=5, threshold=0.7),
            sounds.APPLETV_KEYBOARD_DELETE: MatchProperties(time_tol=2, freq_tol=2, time_delta=5, freq_delta=5, threshold=0.8),
            sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE: MatchProperties(time_tol=2, freq_tol=1, time_delta=5, freq_delta=5, threshold=0.7),
            sounds.APPLETV_KEYBOARD_SCROLL_SIX: MatchProperties(time_tol=2, freq_tol=2, time_delta=3, freq_delta=5, threshold=0.7),
            sounds.APPLETV_SYSTEM_MOVE: MatchProperties(time_tol=2, freq_tol=2, time_delta=5, freq_delta=5, threshold=0.7),
            sounds.APPLETV_TOOLBAR_MOVE: MatchProperties(time_tol=2, freq_tol=2, time_delta=5, freq_delta=5, threshold=0.7)
        }

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
            normalized_target_segment = normalize_spectrogram(target_segment)

            # Get a list of possible sounds to match against the identified noise
            #candidate_sounds = self.get_candidate_sounds(segment=target_segment)

            # Find the closest known sound to the observed noise
            best_sim = 0.0
            best_sound = None

            for sound in sounds.APPLETV_SOUNDS:
                # Match the sound on the spectrograms
                ref_spectrogram = self._spectrograms[sound]  # [F, T0]

                #properties = self._match_properties[sound]
                properties = MatchProperties(time_delta=3, freq_delta=5, time_tol=2, freq_tol=2, threshold=1.5)

                # Binarize both spectrograms
                ref_spectrogram_binary = (ref_spectrogram > properties.threshold).astype(int)
                target_spectrogram_binary = (normalized_target_segment > properties.threshold).astype(int)

                # Compute the match
                similarity = perform_match(first_spectrogram=target_spectrogram_binary,
                                           second_spectrogram=ref_spectrogram_binary)

                # Compute the constellation maps
                #target_times, target_freq = compute_constellation_map(spectrogram=normalized_target_segment,
                #                                                      freq_delta=properties.freq_delta,
                #                                                      time_delta=properties.time_delta,
                #                                                      threshold=properties.threshold)

                #target_peaks = [normalized_target_segment[freq, time] for time, freq in zip(target_times, target_freq)]

                #ref_times, ref_freq = compute_constellation_map(spectrogram=self._spectrograms[sound],
                #                                                 freq_delta=properties.freq_delta,
                #                                                 time_delta=properties.time_delta,
                #                                                 threshold=properties.threshold)

                #ref_peaks = [self._spectrograms[sound][freq, time] for time, freq in zip(ref_times, ref_freq)]

                #if sound in (sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE, sounds.APPLETV_KEYBOARD_SCROLL_SIX):
                #    print('Sound: {}'.format(sound))

                #    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
                #    ax0.imshow(ref_spectrogram_binary, cmap='gray_r')
                #    #ax0.scatter(ref_times, ref_freq, marker='o', color='red')

                #    ax1.imshow(target_spectrogram_binary, cmap='gray_r')
                #    #ax1.scatter(target_times, target_freq, marker='o', color='red')

                #    ax0.set_title('Reference')
                #    ax1.set_title('Target')
                #    
                #    plt.show()
                #    plt.close()

                #shifted_target_times = target_times - np.min(target_times)
                #shifted_ref_times = ref_times - np.min(ref_times)

                #avg_dist = perform_match(first_times=shifted_target_times,
                #                         first_freq=target_freq,
                #                         first_peaks=np.array(target_peaks),
                #                         second_times=shifted_ref_times,
                #                         second_freq=ref_freq,
                #                         second_peaks=np.array(ref_peaks))

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
