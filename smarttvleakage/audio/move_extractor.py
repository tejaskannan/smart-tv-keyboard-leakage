import os.path
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
from collections import namedtuple, defaultdict
from scipy.signal import spectrogram, find_peaks, convolve
from typing import List, Dict, Tuple, DefaultDict

from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, iterate_dir
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations


SoundProfile = namedtuple('SoundProfile', ['channel0', 'channel1', 'channel0_constellation', 'channel1_constellation', 'start', 'end'])
ConstellationParams = namedtuple('ConstellationParams', ['threshold', 'freq_delta', 'time_delta', 'freq_tol', 'time_tol'])

SOUNDS = ['move', 'select', 'key_select']
MIN_DISTANCE = 15
MOVE_BINARY_THRESHOLD = -70
WINDOW_SIZE = 8
SOUND_PROMINENCE = 0.0009

CONSTELLATION_PARAMS = {
    'key_select': ConstellationParams(threshold=-75, freq_delta=10, time_delta=10, freq_tol=2, time_tol=2),
    'select': ConstellationParams(threshold=-60, freq_delta=3, time_delta=3, freq_tol=2, time_tol=2),
    'move': ConstellationParams(threshold=-65, freq_delta=3, time_delta=5, freq_tol=2, time_tol=2)
}


SOUND_THRESHOLDS = {
    'move': (275, 600),
    #'select': (0.0017, 0.0003),
    #'key_select': (0.00275, 0.003)
    'select': (0.85, 0.85),
    'key_select': (0.85, 0.85)
}


def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time


def compute_masked_spectrogram(spectrogram: float, threshold: float, min_freq: int, max_freq: int) -> np.ndarray:
    clipped_spectrogram = spectrogram[min_freq:max_freq, :]
    return (clipped_spectrogram > threshold).astype(int)


def moving_window_similarity(target: np.ndarray, known: np.ndarray, should_smooth: bool, should_match_binary: bool) -> List[float]:
    target = target.T
    known = known.T

    segment_size = known.shape[0]
    similarity: List[float] = []

    for start in range(target.shape[0]):
        end = start + segment_size
        target_segment = target[start:end]

        if len(target_segment) < segment_size:
            target_segment = np.pad(target_segment, pad_width=[(0, segment_size - len(target_segment)), (0, 0)], constant_values=0, mode='constant')

        if not should_match_binary:
            sim_score = 1.0 / np.linalg.norm(target_segment - known, ord=1)
        else:
            sim_score = np.sum(target_segment * known)

        similarity.append(sim_score)

    if should_smooth:
        smooth_filter = np.ones(shape=(WINDOW_SIZE, )) / WINDOW_SIZE
        similarity = convolve(similarity, smooth_filter).astype(float).tolist()

    return similarity


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

                if sound == 'move':
                    channel0_clipped = compute_masked_spectrogram(channel0, threshold=MOVE_BINARY_THRESHOLD, min_freq=start, max_freq=end)
                    channel1_clipped = compute_masked_spectrogram(channel1, threshold=MOVE_BINARY_THRESHOLD, min_freq=start, max_freq=end)
                else:
                    channel0_clipped = channel0[start:end, :]
                    channel1_clipped = channel1[start:end, :]

                constellation_params = CONSTELLATION_PARAMS[sound]

                channel0_constellation = compute_constellation_map(spectrogram=channel0,
                                                                   freq_delta=constellation_params.freq_delta,
                                                                   time_delta=constellation_params.time_delta,
                                                                   threshold=constellation_params.threshold,
                                                                   freq_range=(start, end))

                channel1_constellation = compute_constellation_map(spectrogram=channel1,
                                                                   freq_delta=constellation_params.freq_delta,
                                                                   time_delta=constellation_params.time_delta,
                                                                   threshold=constellation_params.threshold,
                                                                   freq_range=(start, end))

                profile = SoundProfile(channel0=channel0_clipped,
                                       channel1=channel1_clipped,
                                       channel0_constellation=channel0_constellation,
                                       channel1_constellation=channel1_constellation,
                                       start=start,
                                       end=end)
                self._known_sounds[sound].append(profile)

    def compute_spectrogram_similarity_for_sound(self, audio: np.ndarray, sound: str) -> List[float]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The name of the known sound to use
        Returns:
            An array of the moving-window distances
        """
        assert sound in SOUNDS, 'Unknown sound: {}'.format(sound)

        # Create the spectrogram from the known signal
        channel0 = create_spectrogram(signal=audio[:, 0])
        channel1 = create_spectrogram(signal=audio[:, 1])

        # Create the constellations (if needed)
        if sound in ('key_select', 'select'):
            sound_profile = self._known_sounds[sound][0]
            start, end = sound_profile.start, sound_profile.end
            constellation_params = CONSTELLATION_PARAMS[sound]

            channel0_constellation = compute_constellation_map(spectrogram=channel0,
                                                               freq_delta=constellation_params.freq_delta,
                                                               time_delta=constellation_params.time_delta,
                                                               threshold=constellation_params.threshold,
                                                               freq_range=(start, end))

            channel1_constellation = compute_constellation_map(spectrogram=channel1,
                                                               freq_delta=constellation_params.freq_delta,
                                                               time_delta=constellation_params.time_delta,
                                                               threshold=constellation_params.threshold,
                                                               freq_range=(start, end))

        # For each sound type, compute the moving average distances
        similarity_lists: List[List[float]] = []

        for sound_profile in self._known_sounds[sound]:
            start, end = sound_profile.start, sound_profile.end

            if sound in ('key_select', 'select'):
                _, channel0_sim = match_constellations(target_times=channel0_constellation[0],
                                                       target_freq=channel0_constellation[1],
                                                       ref_times=sound_profile.channel0_constellation[0],
                                                       ref_freq=sound_profile.channel0_constellation[1],
                                                       freq_tol=constellation_params.freq_tol,
                                                       time_tol=constellation_params.time_tol,
                                                       time_steps=channel0.shape[1])

                _, channel1_sim = match_constellations(target_times=channel1_constellation[0],
                                                       target_freq=channel1_constellation[1],
                                                       ref_times=sound_profile.channel1_constellation[0],
                                                       ref_freq=sound_profile.channel1_constellation[1],
                                                       freq_tol=constellation_params.freq_tol,
                                                       time_tol=constellation_params.time_tol,
                                                       time_steps=channel1.shape[1])
            else:
                if sound == 'move':
                    channel0_clipped = compute_masked_spectrogram(channel0, threshold=MOVE_BINARY_THRESHOLD, min_freq=start, max_freq=end)
                    channel1_clipped = compute_masked_spectrogram(channel1, threshold=MOVE_BINARY_THRESHOLD, min_freq=start, max_freq=end)
                    should_match_binary = True
                else:
                    channel0_clipped = channel0[start:end, :]
                    channel1_clipped = channel1[start:end, :]
                    should_match_binary = False

                channel0_sim = moving_window_similarity(target=channel0_clipped,
                                                        known=sound_profile.channel0,
                                                        should_smooth=True,
                                                        should_match_binary=should_match_binary)

                channel1_sim = moving_window_similarity(target=channel1_clipped,
                                                        known=sound_profile.channel1,
                                                        should_smooth=True,
                                                        should_match_binary=should_match_binary)

            similarities = [max(c0, c1) for c0, c1 in zip(channel0_sim, channel1_sim)]
            similarity_lists.append(similarities)

        return np.max(similarity_lists, axis=0)

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
        similarity = self.compute_spectrogram_similarity_for_sound(audio=audio, sound=sound)

        (min_threshold, max_threshold) = SOUND_THRESHOLDS[sound]
        threshold = min_threshold

        #if sound == 'key_select':
        #    threshold = min(max(np.mean(similarity) + 4 * np.std(similarity), min_threshold), max_threshold)

        peaks, peak_properties = find_peaks(x=similarity, height=threshold, distance=MIN_DISTANCE, prominence=(SOUND_PROMINENCE, None))
        peak_heights = peak_properties['peak_heights']

        return peaks, peak_heights

    def extract_move_sequence(self, audio: np.ndarray) -> Tuple[List[int], bool]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        raw_key_select_idx, raw_key_select_heights = self.find_instances_of_sound(audio=audio, sound='key_select')

        # Signals without any key selections do not interact with the keyboard
        if len(raw_key_select_idx) == 0:
            return [], False

        # Get occurances of the other two sounds
        move_idx, move_heights = self.find_instances_of_sound(audio=audio, sound='move')
        raw_select_idx, raw_select_heights = self.find_instances_of_sound(audio=audio, sound='select')

        select_idx: List[int] = []
        select_heights: List[int] = []

        # Filter out selects using move detection
        for (idx, height) in zip(raw_select_idx, raw_select_heights):
            idx_diff = np.abs(np.subtract(move_idx, idx))
            if np.all(idx_diff > MIN_DISTANCE):
                select_idx.append(idx)
                select_heights.append(height)

        # Filter out any conflicting key and normal selects
        key_select_idx: List[int] = []
        key_select_heights: List[float] = []

        for (key_idx, peak_height) in zip(raw_key_select_idx, raw_key_select_heights):
            idx_diff = np.abs(np.subtract(select_idx, key_idx))
            if np.all(idx_diff > MIN_DISTANCE):
                key_select_idx.append(key_idx)
                key_select_heights.append(peak_height)

        # The first move starts before the first key select and after the nearest select
        first_key_idx = key_select_idx[0]
        selects_before = list(filter(lambda i: i < first_key_idx, select_idx))
        start_idx = (selects_before[-1] + 50) if len(selects_before) > 0 else 0

        # Extract the number of moves between selections
        # TODO: Handle sequences with multiple keyboard interactions
        clipped_move_idx = list(filter(lambda i: i > start_idx, move_idx))

        key_idx = 0
        num_moves = 0
        last_num_moves = 0
        result: List[int] = []

        i = 0
        while i < len(clipped_move_idx):
            while (key_idx < len(key_select_idx)) and (clipped_move_idx[i] > key_select_idx[key_idx]):
                result.append(num_moves)
                key_idx += 1
                num_moves = 0

            if key_idx >= len(key_select_idx):
                # Get the remaining number of moves before the last done (or end of sequence)
                last_num_moves = (len(clipped_move_idx) - i)
                break

            num_moves += 1
            i += 1

        # If the last number of moves was 0 or 1, then we have the potential to have use the search complete feature
        did_use_autocomplete = (last_num_moves <= 1)

        return result, did_use_autocomplete


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/smart-tv-gettysburg/altogether.MOV')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    sound = 'move'

    extractor = MoveExtractor(tv_type=SmartTVType.SAMSUNG)
    similarity = extractor.compute_spectrogram_similarity_for_sound(audio=audio_signal, sound=sound)
    instance_idx, instance_heights = extractor.find_instances_of_sound(audio=audio_signal, sound=sound)

    move_seq, did_use_autocomplete = extractor.extract_move_sequence(audio=audio_signal)
    print('Move Sequence: {}'.format(move_seq))
    print('Did use autocomplete: {}'.format(did_use_autocomplete))

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    ax0.plot(list(range(audio_signal.shape[0])), audio_signal[:, 0])
    ax1.plot(list(range(len(similarity))), similarity)

    ax1.scatter(instance_idx, instance_heights, marker='o', color='orange')

    plt.show()
