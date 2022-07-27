import os.path
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
from collections import namedtuple, defaultdict
from enum import Enum, auto
from scipy.signal import spectrogram, find_peaks, convolve
from typing import List, Dict, Tuple, DefaultDict

from smarttvleakage.utils.constants import SmartTVType, BIG_NUMBER
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, iterate_dir
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations


SoundProfile = namedtuple('SoundProfile', ['channel0', 'channel1', 'channel0_constellation', 'channel1_constellation', 'start', 'end'])
ConstellationParams = namedtuple('ConstellationParams', ['threshold', 'freq_delta', 'time_delta', 'freq_tol', 'time_tol'])
Move = namedtuple('Move', ['num_moves', 'end_sound'])


MIN_DISTANCE = 12
KEY_SELECT_DISTANCE = 30
MIN_DOUBLE_MOVE_DISTANCE = 30
MOVE_BINARY_THRESHOLD = -70
WINDOW_SIZE = 8
SOUND_PROMINENCE = 0.0009
KEY_SELECT_MOVE_DISTANCE = 20
SELECT_MOVE_DISTANCE = 30
MOVE_DELETE_THRESHOLD = 315


class Sound(Enum):
    MOVE = auto()
    DOUBLE_MOVE = auto()
    SELECT = auto()
    KEY_SELECT = auto()
    DELETE = auto()


CONSTELLATION_PARAMS = {
    #'key_select': ConstellationParams(threshold=-75, freq_delta=10, time_delta=10, freq_tol=2, time_tol=2),
    Sound.KEY_SELECT: ConstellationParams(threshold=-70, freq_delta=5, time_delta=5, freq_tol=3, time_tol=2),
    Sound.SELECT: ConstellationParams(threshold=-60, freq_delta=3, time_delta=5, freq_tol=3, time_tol=3),
    Sound.MOVE: ConstellationParams(threshold=-65, freq_delta=3, time_delta=5, freq_tol=2, time_tol=2),
    Sound.DOUBLE_MOVE: ConstellationParams(threshold=-65, freq_delta=3, time_delta=5, freq_tol=2, time_tol=2),
    Sound.DELETE: ConstellationParams(threshold=-65, freq_delta=3, time_delta=5, freq_tol=2, time_tol=2)
}


SOUND_THRESHOLDS = {
    Sound.MOVE: (275, 600),
    Sound.DOUBLE_MOVE: (450, 600),
    Sound.SELECT: (0.79, 0.85),
    Sound.KEY_SELECT: (0.79, 0.9),
    Sound.DELETE: (0.95, 0.95)
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

        self._known_sounds: DefaultDict[Sound, List[SoundProfile]] = defaultdict(list)

        # Read in the start / end indices (known beforehand)
        freq_range_dict = read_json(os.path.join(sound_directory, 'freq_ranges.json'))

        for sound in Sound:
            sound_name = sound.name.lower()

            for path in iterate_dir(sound_directory):
                file_name = os.path.basename(path)
                if not file_name.startswith(sound_name):
                    continue
                
                audio = read_pickle_gz(path)

                start, end = freq_range_dict[sound_name]['start'], freq_range_dict[sound_name]['end']
                channel0 = create_spectrogram(signal=audio[:, 0])
                channel1 = create_spectrogram(signal=audio[:, 1])

                if sound in (Sound.MOVE, Sound.DOUBLE_MOVE):
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

    def compute_spectrogram_similarity_for_sound(self, audio: np.ndarray, sound: Sound) -> List[float]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The (known) sound to use
        Returns:
            An array of the moving-window distances
        """
        # Create the spectrogram from the known signal
        channel0 = create_spectrogram(signal=audio[:, 0])
        channel1 = create_spectrogram(signal=audio[:, 1])

        # Create the constellations (if needed)
        if sound in (Sound.KEY_SELECT, Sound.SELECT, Sound.DELETE):
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

            if sound in (Sound.KEY_SELECT, Sound.SELECT, Sound.DELETE):
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
                channel0_clipped = compute_masked_spectrogram(channel0, threshold=MOVE_BINARY_THRESHOLD, min_freq=start, max_freq=end)
                channel1_clipped = compute_masked_spectrogram(channel1, threshold=MOVE_BINARY_THRESHOLD, min_freq=start, max_freq=end)

                channel0_sim = moving_window_similarity(target=channel0_clipped,
                                                        known=sound_profile.channel0,
                                                        should_smooth=True,
                                                        should_match_binary=True)

                channel1_sim = moving_window_similarity(target=channel1_clipped,
                                                        known=sound_profile.channel1,
                                                        should_smooth=True,
                                                        should_match_binary=True)

            similarities = [max(c0, c1) for c0, c1 in zip(channel0_sim, channel1_sim)]
            similarity_lists.append(similarities)

        return np.max(similarity_lists, axis=0)

    def find_instances_of_sound(self, audio: np.ndarray, sound: Sound) -> Tuple[List[int], List[float]]:
        """
        Finds instances of the given sound in the provided audio signal
        by finding peaks in the spectrogram distance chart.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The sound to find
        Return:
            A tuple of 3 elements:
                (1) A list of the `times` in which the peaks occur in the distance graph
                (2) A list of the peak values in the distance graph
        """
        similarity = self.compute_spectrogram_similarity_for_sound(audio=audio, sound=sound)

        (min_threshold, max_threshold) = SOUND_THRESHOLDS[sound]
        threshold = min_threshold

        distance = KEY_SELECT_DISTANCE if sound == Sound.KEY_SELECT else MIN_DISTANCE
        peaks, peak_properties = find_peaks(x=similarity, height=threshold, distance=distance, prominence=(SOUND_PROMINENCE, None))
        peak_heights = peak_properties['peak_heights']

        # Filter out duplicate double moves
        if (sound == Sound.DOUBLE_MOVE) and (len(peaks) > 0):
            filtered_peaks = [peaks[0]]
            filtered_peak_heights = [peak_heights[0]]

            for idx in range(len(peaks)):
                if (peaks[idx] - filtered_peaks[-1]) >= MIN_DOUBLE_MOVE_DISTANCE:
                    filtered_peaks.append(peaks[idx])
                    filtered_peak_heights.append(peak_heights[idx])

                idx += 1

            return filtered_peaks, filtered_peak_heights
        else:
            return peaks, peak_heights

    def extract_move_sequence(self, audio: np.ndarray) -> Tuple[List[Move], bool]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        raw_key_select_times, raw_key_select_heights = self.find_instances_of_sound(audio=audio, sound=Sound.KEY_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(raw_key_select_times) == 0:
            return [], False

        # Get occurances of the other two sounds
        move_times, move_heights = self.find_instances_of_sound(audio=audio, sound=Sound.MOVE)
        double_move_times, double_move_heights = self.find_instances_of_sound(audio=audio, sound=Sound.DOUBLE_MOVE)
        raw_select_times, raw_select_heights = self.find_instances_of_sound(audio=audio, sound=Sound.SELECT)

        select_times: List[int] = []
        select_heights: List[int] = []

        # Filter out selects using move detection
        for (t, height) in zip(raw_select_times, raw_select_heights):
            move_diff = np.abs(np.subtract(move_times, t))
            double_move_diff = np.abs(np.subtract(double_move_times, t))

            if np.all(move_diff > SELECT_MOVE_DISTANCE) and np.all(double_move_diff > MIN_DOUBLE_MOVE_DISTANCE):
                select_times.append(t)
                select_heights.append(height)

        # Filter out any conflicting key and normal selects
        key_select_times: List[int] = []
        key_select_heights: List[float] = []

        last_select = select_times[-1] if len(select_times) > 0 else BIG_NUMBER

        for (t, peak_height) in zip(raw_key_select_times, raw_key_select_heights):
            select_time_diff = np.abs(np.subtract(select_times, t))
            move_time_diff = np.abs(np.subtract(move_times, t))

            num_moves_between = len(list(filter(lambda move_time: (move_time > last_select) and (move_time < t), move_times)))

            if np.all(select_time_diff > KEY_SELECT_DISTANCE) and np.all(move_time_diff > KEY_SELECT_DISTANCE) and ((t < last_select) or (num_moves_between > 0)):
                key_select_times.append(t)
                key_select_heights.append(peak_height)

        if len(key_select_times) == 0:
            return [], False

        # The first move starts before the first key select and after the nearest select
        first_key_time = key_select_times[0]
        selects_before = list(filter(lambda t: t < first_key_time, select_times))
        start_time = (selects_before[-1] + MIN_DISTANCE) if len(selects_before) > 0 else 0

        # Extract the number of moves between selections
        # TODO: Handle sequences with multiple keyboard interactions
        clipped_move_times = list(filter(lambda t: t > start_time, move_times))
        clipped_double_move_times = list(filter(lambda t: t > start_time, double_move_times))
        clipped_select_times = list(filter(lambda t: (t > start_time) and any(True for s in key_select_times if s > t), select_times))

        move_idx = 0
        key_idx = 0
        select_idx = 0
        double_move_idx = 0
        num_moves = 0
        last_num_moves = 0
        result: List[int] = []

        while move_idx < len(clipped_move_times):
            while (key_idx < len(key_select_times)) and (clipped_move_times[move_idx] > key_select_times[key_idx]):
                result.append(Move(num_moves=num_moves, end_sound=Sound.KEY_SELECT))
                key_idx += 1
                num_moves = 0

            while (select_idx < len(clipped_select_times)) and (clipped_move_times[move_idx] > clipped_select_times[select_idx]):
                result.append(Move(num_moves=num_moves, end_sound=Sound.SELECT))
                select_idx += 1
                num_moves = 0

            if (double_move_idx < len(clipped_double_move_times)) and (abs(clipped_double_move_times[double_move_idx] - clipped_move_times[move_idx]) <= MIN_DOUBLE_MOVE_DISTANCE):
                move_idx += 1
                while (move_idx < len(clipped_move_times)) and (abs(clipped_double_move_times[double_move_idx] - clipped_move_times[move_idx]) <= MIN_DOUBLE_MOVE_DISTANCE):
                    move_idx += 1

                double_move_idx += 1
                num_moves += 2
            else:
                num_moves += 1
                move_idx += 1

        # Write out the last group if we haven't reached the last key
        if key_idx < len(key_select_times):
            result.append(Move(num_moves=num_moves, end_sound=Sound.KEY_SELECT))

        # If the last number of moves was 0 or 1, then the user leveraged the word autocomplete feature
        # NOTE: We can also validate this based on the number of possible moves (whether it was possible to get
        # to the top of the keyboard on this turn)
        last_key_select = key_select_times[-1]
        selects_after = list(filter(lambda t: t > last_key_select, select_times))
        next_select = selects_after[0] if len(selects_after) > 0 else BIG_NUMBER

        moves_between = len(list(filter(lambda t: (t <= next_select) and (t >= last_key_select), clipped_move_times)))
        did_use_autocomplete = (moves_between == 0) and (len(selects_after) > 0)

        if did_use_autocomplete and len(result) > 0:
            return result[0:-1], did_use_autocomplete

        return result, did_use_autocomplete


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/smart-tv-backspace/tet.MOV')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    sound = Sound.DELETE

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
