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
from smarttvleakage.audio.constants import SAMSUNG_DELETE, SAMSUNG_DOUBLE_MOVE, SAMSUNG_KEY_SELECT
from smarttvleakage.audio.constants import SAMSUNG_MOVE, SAMSUNG_SELECT, APPLETV_KEYBOARD_DELETE
from smarttvleakage.audio.constants import APPLETV_KEYBOARD_MOVE, APPLETV_KEYBOARD_SELECT, APPLETV_SYSTEM_MOVE
from smarttvleakage.audio.constants import SAMSUNG_SOUNDS, APPLETV_SOUNDS


SoundProfile = namedtuple('SoundProfile', ['channel0', 'channel1', 'start', 'end', 'match_threshold'])
Move = namedtuple('Move', ['num_moves', 'end_sound'])


APPLETV_MOVE_DISTANCE = 5
MIN_DISTANCE = 12
KEY_SELECT_DISTANCE = 30
MIN_DOUBLE_MOVE_DISTANCE = 30
MOVE_BINARY_THRESHOLD = -70
WINDOW_SIZE = 8
SOUND_PROMINENCE = 0.0009
KEY_SELECT_MOVE_DISTANCE = 20
SELECT_MOVE_DISTANCE = 30
MOVE_DELETE_THRESHOLD = 315


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
        self._tv_type = tv_type

        # Read in the start / end indices (known beforehand)
        config = read_json(os.path.join(sound_directory, 'config.json'))

        if tv_type == SmartTVType.SAMSUNG:
            self._sound_names = SAMSUNG_SOUNDS
        elif tv_type == SmartTVType.APPLE_TV:
            self._sound_names = APPLETV_SOUNDS
        else:
            raise ValueError('Unknown TV Type: {}'.format(tv_type.name))

        for sound in self._sound_names:
            for path in iterate_dir(sound_directory):
                file_name = os.path.basename(path)
                if not file_name.startswith(sound):
                    continue
                
                audio = read_pickle_gz(path)

                start, end = config[sound]['start'], config[sound]['end']
                channel0 = create_spectrogram(signal=audio[:, 0])
                channel1 = create_spectrogram(signal=audio[:, 1])

                threshold = config[sound]['mask_threshold']

                channel0_clipped = compute_masked_spectrogram(channel0, threshold=threshold, min_freq=start, max_freq=end)
                channel1_clipped = compute_masked_spectrogram(channel1, threshold=threshold, min_freq=start, max_freq=end)

                profile = SoundProfile(channel0=channel0_clipped,
                                       channel1=channel1_clipped,
                                       start=start,
                                       end=end,
                                       match_threshold=config[sound]['match_threshold'])
                self._known_sounds[sound].append(profile)

    def compute_spectrogram_similarity_for_sound(self, audio: np.ndarray, sound: str) -> List[float]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The name of the sound to use
        Returns:
            An array of the moving-window distances
        """
        assert sound in self._sound_names, 'Unknown sound {}. The sound must be one of {}'.format(sound, self._sound_names)

        # Create the spectrogram from the known signal
        channel0 = create_spectrogram(signal=audio[:, 0])
        channel1 = create_spectrogram(signal=audio[:, 1])

        # For each sound type, compute the moving average distances
        similarity_lists: List[List[float]] = []

        for sound_profile in self._known_sounds[sound]:
            start, end = sound_profile.start, sound_profile.end

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

    def find_instances_of_sound(self, audio: np.ndarray, sound: str) -> Tuple[List[int], List[float]]:
        """
        Finds instances of the given sound in the provided audio signal
        by finding peaks in the spectrogram distance chart.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
            sound: The name of the sound to find
        Return:
            A tuple of 3 elements:
                (1) A list of the `times` in which the peaks occur in the distance graph
                (2) A list of the peak values in the distance graph
        """
        assert sound in self._sound_names, 'Unknown sound {}. The sound must be one of {}'.format(sound, self._sound_names)

        similarity = self.compute_spectrogram_similarity_for_sound(audio=audio, sound=sound)

        sound_profile = self._known_sounds[sound][0]
        threshold = sound_profile.match_threshold

        if (self._tv_type == SmartTVType.SAMSUNG) and (sound == SAMSUNG_KEY_SELECT):
            distance = KEY_SELECT_DISTANCE
        elif (self._tv_type == SmartTVType.APPLE_TV) and (sound == APPLETV_KEYBOARD_MOVE):
            distance = APPLETV_MOVE_DISTANCE
        else:
            distance = MIN_DISTANCE

        peaks, peak_properties = find_peaks(x=similarity, height=threshold, distance=distance, prominence=(SOUND_PROMINENCE, None))
        peak_heights = peak_properties['peak_heights']

        # Filter out duplicate double moves
        if (sound == SAMSUNG_DOUBLE_MOVE) and (len(peaks) > 0):
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
        raise NotImplementedError()


class SamsungMoveExtractor(MoveExtractor):

    def __init__(self):
        super().__init__(SmartTVType.SAMSUNG)

    def extract_move_sequence(self, audio: np.ndarray) -> Tuple[List[Move], bool]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        raw_key_select_times, raw_key_select_heights = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_KEY_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(raw_key_select_times) == 0:
            return [], False

        # Get occurances of the other two sounds
        move_times, move_heights = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_MOVE)
        double_move_times, double_move_heights = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_DOUBLE_MOVE)
        raw_select_times, raw_select_heights = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_SELECT)

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
                result.append(Move(num_moves=num_moves, end_sound=SAMSUNG_KEY_SELECT))
                key_idx += 1
                num_moves = 0

            while (select_idx < len(clipped_select_times)) and (clipped_move_times[move_idx] > clipped_select_times[select_idx]):
                result.append(Move(num_moves=num_moves, end_sound=SAMSUNG_SELECT))
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
            result.append(Move(num_moves=num_moves, end_sound=SAMSUNG_KEY_SELECT))

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


class AppleTVMoveExtractor(MoveExtractor):

    def __init__(self):
        super().__init__(SmartTVType.APPLE_TV)

    def extract_move_sequence(self, audio: np.ndarray) -> Tuple[List[Move], bool]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        keyboard_select_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(keyboard_select_times) == 0:
            return [], False

        # Get occurances of the other sounds
        raw_keyboard_move_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_MOVE)
        raw_system_move_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_SYSTEM_MOVE)

        # Filter out conflicting sounds
        system_move_times: List[int] = []
        for move_time in raw_system_move_times:
            diff = np.abs(np.subtract(keyboard_select_times, move_time))
            if np.all(diff > MIN_DISTANCE):
                system_move_times.append(move_time)

        keyboard_move_times: List[int] = []
        for move_time in raw_keyboard_move_times:
            keyboard_diff = np.abs(np.subtract(keyboard_select_times, move_time))
            system_diff = np.abs(np.subtract(system_move_times, move_time))
            if np.all(keyboard_diff > MIN_DISTANCE) and np.all(system_diff > MIN_DISTANCE):
                keyboard_move_times.append(move_time)

        # Get the end time as the last system move
        end_time = system_move_times[-1] if len(system_move_times) > 0 else BIG_NUMBER

        # Extract the move sequence
        move_sequence: List[Move] = []

        num_moves = 0
        move_idx = 0
        key_select_idx = 0

        while (move_idx < len(keyboard_move_times)) and (keyboard_move_times[move_idx] < end_time):
            # Write move elements
            while (key_select_idx < len(keyboard_select_times)) and (keyboard_move_times[move_idx] > keyboard_select_times[key_select_idx]):
                # A quirk of Apple TV -> The entry to the keyboard makes the keyboard move sound. We remove this move
                if len(move_sequence) == 0:
                    num_moves -= 1

                move_sequence.append(Move(num_moves=num_moves, end_sound=APPLETV_KEYBOARD_SELECT))
                num_moves = 0
                key_select_idx += 1

            num_moves += 1
            move_idx += 1

        if key_select_idx < len(keyboard_select_times):
            move_sequence.append(Move(num_moves=num_moves, end_sound=APPLETV_KEYBOARD_SELECT))

        return move_sequence, False


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/apple-tv/ten/wecrashed.MOV')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    sound = APPLETV_KEYBOARD_SELECT

    extractor = AppleTVMoveExtractor()
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
