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
from typing import List, Tuple, Set, Any

import smarttvleakage.audio.sounds as sounds
#from smarttvleakage.audio import MatchConfig
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


def binarize_and_clip(spectrogram: np.ndarray, match_configs: List[Any]) -> np.ndarray:
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
    # Compute the distance in time, frequency, and ranking
    time_diffs = np.abs(np.expand_dims(ref_times, axis=-1) - np.expand_dims(target_times, axis=0))  # [N, M]
    freq_diffs = np.abs(np.expand_dims(ref_freq, axis=-1) - np.expand_dims(target_freq, axis=0))  # [N, M]
    distances = time_diffs + freq_diffs  # [N, M]

    num_matches = 0

    for ref_idx in range(len(ref_times)):
        closest_target_idx = np.argmin(distances[ref_idx])

        if (time_diffs[ref_idx, closest_target_idx] <= 1) and (freq_diffs[ref_idx, closest_target_idx] <= 1):
            num_matches += 1
            distances[ref_idx, :] = BIG_NUMBER
            distances[:, closest_target_idx] = BIG_NUMBER

    return num_matches / max(len(target_times), len(ref_times))


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
        #distance = np.average(np.abs(padded_spectrogram - first_spectrogram))
        #similarity = 1.0 / distance

        best_similarity = max(similarity, best_similarity)
        pad_after -= 1

    return best_similarity


def is_in_peak_range(peak_time: int, peak_ranges: Set[Tuple[int, int]]) -> bool:
    for peak_range in peak_ranges:
        if (peak_time >= peak_range[0]) and (peak_time <= peak_range[1]):
            return True

    return False


def get_sound_instances(max_energy: np.ndarray, peak_height: float, threshold_factor: float) -> Tuple[List[int], List[int]]:
    peak_ranges: Set[Tuple[int, int]] = set()

    peak_times, peak_properties = find_peaks(max_energy, height=peak_height, distance=2)
    peak_heights = peak_properties['peak_heights']
    prev_end = 0

    for peak_time, peak_height in zip(peak_times, peak_heights):
        if is_in_peak_range(peak_time, peak_ranges):
            continue

        peak_threshold = peak_height * threshold_factor

        # Get the start and end point
        start_time = peak_time
        while (start_time > prev_end) and (max_energy[start_time] > peak_threshold):
            start_time -= 1

        start_time += 1  # Adjust for going beyond the threshold

        end_time = peak_time
        while (end_time < len(max_energy)) and (max_energy[end_time] > peak_threshold):
            end_time += 1

        # Add the range to the result set
        peak_ranges.add((start_time, end_time))
        prev_end = end_time

    # Sort the ranges by start time
    peak_ranges_sorted = list(sorted(peak_ranges, key=lambda t: t[0]))
    start_times = [t[0] for t in peak_ranges_sorted]
    end_times = [t[1] for t in peak_ranges_sorted]
    return start_times, end_times



class AppleTVMatcher:

    def __init__(self, sound_folder: str):
        self._range_min = 5
        self._range_max = 150

        self._sound_threshold_factor = 1.3
        self._peak_threshold = -47
        self._min_time_gap = 3
        self._similarity_threshold = 0.15
        self._min_length = 3

        self._spectrograms: Dict[str, np.ndarray] = dict()

        self._match_configs = read_json(os.path.join(sound_folder, 'config.json'))

        for sound in sounds.APPLETV_SOUNDS:
            # Read in the saved sound
            path = os.path.join(sound_folder, '{}.pkl.gz'.format(sound))
            audio = read_pickle_gz(path)[:, 0]  # Extract channel 0

            # Compute the spectrogram, clip the sound, and normalize the result
            spectrogram = create_spectrogram(audio)[self._range_min:self._range_max]

            sound_threshold_factor = self._sound_threshold_factor if (sound != sounds.APPLETV_KEYBOARD_MOVE) else 1.15

            sound_starts, sound_ends = get_sound_instances(max_energy=np.max(spectrogram, axis=0),
                                                           peak_height=self._peak_threshold,
                                                           threshold_factor=sound_threshold_factor)
            sound_start, sound_end = sound_starts[0], sound_ends[0]

            self._spectrograms[sound] = spectrogram[:, sound_start:sound_end]

    @property
    def range_min(self) -> int:
        return self._range_min

    @property
    def range_max(self) -> int:
        return self._range_max

    def get_reference_constellation(self, sound: str) -> Tuple[List[float], List[float]]:
        if sound in sounds.APPLETV_SOUNDS:
            properties = self._match_configs[sound]
            return compute_constellation_map(spectrogram=self._spectrograms[sound],
                                             freq_delta=properties['freq_delta'],
                                             time_delta=properties['time_delta'],
                                             threshold=properties['threshold'])

        if sound == sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE:
            num_reps = 2
        elif sound == sounds.APPLETV_KEYBOARD_SCROLL_TRIPLE:
            num_reps = 3
        elif sound == sounds.APPLETV_KEYBOARD_SCROLL_FOUR:
            num_reps = 4
        elif sound == sounds.APPLETV_KEYBOARD_SCROLL_FIVE:
            num_reps = 5
        else:
            raise ValueError('Unknown sound: {}'.format(sound))

        # Stitch together multiple moves
        base_spectrogram = self._spectrograms[sounds.APPLETV_KEYBOARD_MOVE]
        spectrogram = np.concatenate([base_spectrogram for _ in range(num_reps)], axis=-1)

        properties = self._match_configs[sounds.APPLETV_KEYBOARD_MOVE]
        times, freqs = compute_constellation_map(spectrogram=spectrogram,
                                                 freq_delta=properties['freq_delta'],
                                                 time_delta=properties['time_delta'],
                                                 threshold=properties['threshold'])

        #fig, ax = plt.subplots()

        #ax.imshow(spectrogram, cmap='gray_r')
        #ax.scatter(times, freqs, marker='o', color='red')
        #ax.set_title('Reference {}'.format(sound))

        #plt.show()

        return times, freqs

    def find_sounds(self, target_audio: np.ndarray) -> List[SoundMatch]:
        assert len(target_audio.shape) == 1, 'Must provide a 1d array of audio values'

        # Compute the spectrogram of the target audio signal
        target_spectrogram = create_spectrogram(target_audio)[self._range_min:self._range_max]  # [F, T]

        # Find instances of any sounds (for later matching)
        max_energy = np.max(target_spectrogram, axis=0)
        start_times, end_times = get_sound_instances(max_energy=max_energy,
                                                     peak_height=self._peak_threshold,
                                                     threshold_factor=self._sound_threshold_factor)

        results: List[SoundMatch] = []

        for start_time, end_time in zip(start_times, end_times):
            # Clip the start and end time
            target_segment = target_spectrogram[:, start_time:end_time]
            #normalized_target_segment = normalize_spectrogram(target_segment)
            #normalized_target_segment = maximum_filter(normalized_target_segment, size=(3, 3))

            # Skip segments that are too short, as these are likely just noise
            if target_spectrogram.shape[1] < self._min_length:
                continue

            # Get a list of possible sounds to match against the identified noise
            #candidate_sounds = self.get_candidate_sounds(segment=target_segment)

            # Find the closest known sound to the observed noise
            best_sim, second_best_sim = 0.0, 0.0
            best_sound, second_best_sound = None, None

            for sound in sorted(sounds.APPLETV_SOUNDS_EXTENDED):
                # Match the sound on the spectrograms
                properties = self._match_configs[sound] if (sound not in sounds.APPLETV_MOVE_SOUNDS) else self._match_configs[sounds.APPLETV_KEYBOARD_MOVE]

                # Compute the constellation maps
                target_times, target_freq = compute_constellation_map(spectrogram=target_segment,
                                                                      freq_delta=properties['freq_delta'],
                                                                      time_delta=properties['time_delta'],
                                                                      threshold=properties['threshold'])

                target_peaks = [target_segment[freq, time] for time, freq in zip(target_times, target_freq)]

                ref_times, ref_freq = self.get_reference_constellation(sound=sound)
                ref_peaks = []

                #if sound in (sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE, sounds.APPLETV_KEYBOARD_MOVE, sounds.APPLETV_KEYBOARD_SCROLL_TRIPLE):
                #if (sound == sounds.APPLETV_KEYBOARD_SCROLL_TRIPLE) and (start_time > 1700):
                #    print('Sound: {}'.format(sound))

                #    #fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
                #    #ax0.imshow(ref_spectrogram, cmap='gray_r')
                #    #ax0.scatter(ref_times, ref_freq, marker='o', color='red')

                #    fig, ax = plt.subplots()

                #    ax.imshow(target_segment, cmap='gray_r')
                #    ax.scatter(target_times, target_freq, marker='o', color='red')
                #    ax.scatter(ref_times, ref_freq, marker='x', color='green')

                #    #ax0.set_title('Reference')
                #    ax.set_title('Target')

                #    print('Target: Times -> {}, Freq -> {}'.format(target_times, target_freq))
                #    print('Ref: Times -> {}, Freq -> {}'.format(ref_times, ref_freq))

                #    plt.show()
                #    plt.close()

                shifted_target_times = target_times - np.min(target_times)
                shifted_ref_times = ref_times - np.min(ref_times)

                similarity = perform_match_constellations(target_times=shifted_target_times,
                                         target_freq=target_freq,
                                         target_peaks=np.array(target_peaks),
                                         ref_times=shifted_ref_times,
                                         ref_freq=ref_freq,
                                         ref_peaks=np.array(ref_peaks))

                print('Sound: {}, Similarity: {}'.format(sound, similarity))

                if similarity > best_sim:
                    second_best_sim = best_sim
                    second_best_sound = best_sound

                    best_sim = similarity
                    best_sound = sound

            # The toolbar sound constellations can sometimes conflict with the keyboard selection and/or the single movement.
            # The order of the the peaks, however, are vastly difference between these sounds. Thus, we check matches on the toolbar
            # sound against the second best candidate using binary matching directly on the spectrograms.
            if (best_sound == sounds.APPLETV_TOOLBAR_MOVE) and (second_best_sound is not None):
                # Compute the binary spectrograms
                binary_threshold = -70

                best_ref_spectrogram_binary = (self._spectrograms[best_sound] > binary_threshold).astype(int)
                second_best_ref_spectrogram_binary = (self._spectrograms[second_best_sound] > binary_threshold).astype(int)
                target_spectrogram_binary = (target_segment > binary_threshold).astype(int)

                # Get the matching scores
                best_sim_binary = perform_match_binary(target_spectrogram_binary, best_ref_spectrogram_binary)
                second_best_sim_binary = perform_match_binary(target_spectrogram_binary, second_best_ref_spectrogram_binary)

                print('Toolbar Sim Binary: {}, {} Sim Binary: {}'.format(best_sim_binary, second_best_sound, second_best_sim_binary))

                # Determine the best sound based on this `tie-breaker`
                best_sound = best_sound if (best_sim_binary > second_best_sim_binary) else second_best_sound
                best_sim = best_sim if (best_sim_binary > second_best_sim_binary) else second_best_sim

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
