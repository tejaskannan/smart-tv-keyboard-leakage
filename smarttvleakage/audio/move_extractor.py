import os.path
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
from collections import namedtuple, defaultdict
from scipy.signal import spectrogram, find_peaks, convolve
from typing import List, Dict, Tuple, DefaultDict

from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, iterate_dir


SoundProfile = namedtuple('SoundProfile', ['channel0', 'channel1', 'start', 'end'])

SOUNDS = ['move', 'select', 'key_select', 'double_move']
MOVE_STEPS = 100
SELECT_FACTOR = 0.82
MIN_DISTANCE = 20
WINDOW_SIZE = 8

SOUND_THRESHOLDS = {
    'move': (0.005, 0.006),
    'select': (0.007, 0.01),
    'key_select': (0.065, 0.09),
    'double_move': (0.0045, 0.006)
}


SOUND_PROMINENCE = {
    'move': 1e-3,
    'select': 1e-3,
    'key_select': 1e-3,
    'double_move': 1e-3
}


#SOUND_THRESHOLDS = {
#    'move': 0.005,
#    'select': 0.003,
#    'key_select': 0.07,
#    'double_move': 0.0045
#}


SOUND_FACTORS = {
    'move': 0.4,
    'select': 2.0,
    'key_select': 2.0,
    'double_move': 1.0
}

#SOUND_FACTORS = {
#    'move': 0.1,
#    'select': 0.5,
#    'key_select': 1.1,
#    'double_move': 0.75
#}



def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time


def moving_window_distances(target: np.ndarray, known: np.ndarray, should_smooth: bool, should_match_binary: bool) -> List[float]:
    target = target.T
    known = known.T

    segment_size = known.shape[0]

    distances: List[float] = []
    for start in range(target.shape[0]):
        end = start + segment_size
        target_segment = target[start:end]

        if len(target_segment) < segment_size:
            target_segment = np.pad(target_segment, pad_width=[(0, segment_size - len(target_segment)), (0, 0)], constant_values=0, mode='constant')

        if not should_match_binary:
            dist = 1.0 / np.linalg.norm(target_segment - known, ord=1)
        else:
            dist = np.sum(target_segment * known)

        distances.append(dist)

    if should_smooth:
        smooth_filter = np.ones(shape=(WINDOW_SIZE, )) / WINDOW_SIZE
        distances = convolve(distances, smooth_filter).astype(float).tolist()

    return distances


class MoveExtractor:

    def __init__(self, tv_type: SmartTVType):
        directory = os.path.dirname(__file__)
        sound_directory = os.path.join(directory, '..', 'sounds', tv_type.name.lower())

        self._known_sounds: DefaultDict[str, List[SoundProfile]] = defaultdict(list)

        # Read in the start / end indices (known beforehand)
        freq_range_dict = read_json(os.path.join(sound_directory, 'freq_ranges.json'))

        for sound in SOUNDS:
            for path in iterate_dir(sound_directory):
                file_name = os.path.basename(path)
                if not file_name.startswith(sound):
                    continue
                
                audio = read_pickle_gz(path)

                start, end = freq_range_dict[sound]['start'], freq_range_dict[sound]['end']
                channel0 = create_spectrogram(signal=audio[:, 0])
                channel1 = create_spectrogram(signal=audio[:, 1])

                profile = SoundProfile(channel0=channel0, channel1=channel1, start=start, end=end)
                self._known_sounds[sound].append(profile)

    def compute_spectrogram_distances_for_sound(self, audio: np.ndarray, sound: str) -> List[float]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The name of the known sound to use
        Returns:
            An array of the moving-window distances
        """
        # Create the spectrogram from the known signal
        channel0 = create_spectrogram(signal=audio[:, 0])
        channel1 = create_spectrogram(signal=audio[:, 1])

        # For each sound type, compute the moving average distances
        distance_lists: List[List[float]] = []

        for sound_profile in self._known_sounds[sound]:
            start, end = sound_profile.start, sound_profile.end

            channel0_dist = moving_window_distances(target=channel0[start:end],
                                                    known=sound_profile.channel0[start:end],
                                                    should_smooth=True)

            channel1_dist = moving_window_distances(target=channel1[start:end],
                                                    known=sound_profile.channel1[start:end],
                                                    should_smooth=True)

            distances = [c0 + c1 for c0, c1 in zip(channel0_dist, channel1_dist)]
            distance_lists.append(distances)

        return np.max(distance_lists, axis=0)

    def find_instances_of_sound(self, audio: np.ndarray, sound: str) -> Tuple[List[int], List[float]]:
        """
        Finds instances of the given sound in the provided audio signal
        by finding peaks in the spectrogram distance chart.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The name of the sound. Must be in SOUNDS.
        Return:
            A tuple of 3 elements:
                (1) A list of the `times` in which the peaks occur in the distance graph
                (2) A list of the peak values in the distance graph
        """
        assert sound in SOUNDS, 'The provided sound must be in [{}]. Got: {}'.format(','.join(SOUNDS), sound)
        distances = self.compute_spectrogram_distances_for_sound(audio=audio, sound=sound)

        (min_threshold, max_threshold) = SOUND_THRESHOLDS[sound]
        threshold = np.mean(distances) + SOUND_FACTORS[sound] * np.std(distances)
        threshold = min(max(threshold, min_threshold), max_threshold)
        #threshold = SOUND_THRESHOLDS[sound]
        #unique_distances = list(set(np.round(distances, decimals=4)))
        #iqr = np.percentile(unique_distances, 75) - np.percentile(unique_distances, 25)
        #threshold = np.median(unique_distances) + SOUND_FACTORS[sound] * iqr

        #print('Sound: {}, Threshold: {}'.format(sound, threshold))

        peaks, peak_properties = find_peaks(x=distances, height=threshold, distance=MIN_DISTANCE, prominence=(SOUND_PROMINENCE[sound], None))
        peak_heights = peak_properties['peak_heights']

        #print(peak_properties['prominences'])

        return peaks, peak_heights

        # Filter out sounds that are not above 0.5 * stddev if not in a cluster of peaks
        #if sound == 'move':
        #    filtered_peaks: List[int] = []
        #    filtered_peak_heights: List[float] = []

        #    high_threshold = np.mean(distances) + 2 * SOUND_FACTORS[sound] * np.std(distances)
        #    
        #    # Get the first peak above the higher threshold
        #    for i in range(len(peaks)):
        #        if any([peak_heights[j] > high_threshold for j in range(i + 1) if ((peaks[i] - peaks[j]) < MOVE_STEPS)]):
        #            filtered_peaks.append(peaks[i])
        #            filtered_peak_heights.append(peak_heights[i])

        #    return filtered_peaks, filtered_peak_heights
        #elif sound == 'select':
        #    #cutoff = SELECT_FACTOR * max(peak_heights)
        #    cutoff = 0.003
        #    filtered_peaks = [peaks[i] for i in range(len(peaks)) if peak_heights[i] > cutoff]
        #    filtered_peak_heights = [peak_heights[i] for i in range(len(peaks)) if peak_heights[i] > cutoff]

        #    return filtered_peaks, filtered_peak_heights
        #else:
        #    return peaks, peak_heights

    def extract_move_sequence(self, audio: np.ndarray) -> Tuple[List[int], bool]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        key_select_idx, key_select_heights = self.find_instances_of_sound(audio=audio, sound='key_select')

        # Signals without any key selections do not interact with the keyboard
        if len(key_select_idx) == 0:
            return [], False

        # Get occurances of the other two sounds
        move_idx, move_heights = self.find_instances_of_sound(audio=audio, sound='move')
        select_idx, select_heights = self.find_instances_of_sound(audio=audio, sound='select')
        double_idx, double_heights = self.find_instances_of_sound(audio=audio, sound='double_move')

        # The first move starts before the first key select and after the nearest select
        first_key = key_select_idx[0]
        selects_before = list(filter(lambda i: i < first_key, select_idx))
        start_idx = (selects_before[-1] + 50) if len(selects_before) > 0 else 0

        # Extract the number of moves between selections
        # TODO: Handle sequences with multiple keyboard interactions
        clipped_move_idx = list(filter(lambda i: i > start_idx, move_idx))
        clipped_double_idx = list(filter(lambda i: i > start_idx, double_idx))

        key_idx = 0
        num_moves = 0
        last_num_moves = 0
        result: List[int] = []

        #print(clipped_double_idx)
        #print(clipped_move_idx)

        i = 0
        j = 0
        while i < len(clipped_move_idx):
            #print('Move Idx: {}, Num Moves: {}'.format(clipped_move_idx[i], num_moves))

            while (key_idx < len(key_select_idx)) and (clipped_move_idx[i] > key_select_idx[key_idx]):
                result.append(num_moves)
                key_idx += 1
                num_moves = 0

            if key_idx >= len(key_select_idx):
                # Get the remaining number of moves before the last done (or end of sequence)
                last_num_moves = (len(clipped_move_idx) - i)
                break

            is_double = 0
            if (j < len(clipped_double_idx)) and (abs(clipped_double_idx[j] - clipped_move_idx[i]) < 50):
                is_double = 1
                while (i < len(clipped_move_idx)) and (j < len(clipped_double_idx)) and (abs(clipped_double_idx[j] - clipped_move_idx[i]) < 50):
                    i += 1

                j += 1
                num_moves += 2
            else:
                num_moves += 1
                i += 1

        # If the last number of moves was 0 or 1, then we have the potential to have use the search complete feature
        did_use_autocomplete = (last_num_moves <= 1)

        return result, did_use_autocomplete


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/smart-tv-gettysburg/portion.MOV')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    sound = 'select'

    extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
    distances = extractor.compute_spectrogram_distances_for_sound(audio=audio_signal, sound=sound)
    instance_idx, instance_heights = extractor.find_instances_of_sound(audio=audio_signal, sound=sound)

    move_seq, did_use_autocomplete = extractor.extract_move_sequence(audio=audio_signal)
    print('Move Sequence: {}'.format(move_seq))
    print('Did use autocomplete: {}'.format(did_use_autocomplete))

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    ax0.plot(list(range(audio_signal.shape[0])), audio_signal[:, 0])
    ax1.plot(list(range(len(distances))), distances)

    ax1.scatter(instance_idx, instance_heights, marker='o', color='orange')

    plt.show()
