import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from enum import Enum, auto
from typing import List, Dict, Tuple, DefaultDict, Union

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.utils.constants import SmartTVType, BIG_NUMBER, KeyboardType, Direction, SMALL_NUMBER
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, iterate_dir
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations
from smarttvleakage.audio.data_types import Move
from smarttvleakage.audio.utils import filter_conflicts
from smarttvleakage.audio.audio_extractor import SmartTVAudio


MatchParams = namedtuple('MatchParams', ['threshold', 'min_freq', 'max_freq', 'time_tol', 'freq_tol', 'time_delta', 'freq_delta'])
SoundProfile = namedtuple('SoundProfile', ['spectrogram', 'match_params', 'match_threshold', 'match_buffer', 'min_threshold'])


APPLETV_PASSWORD_THRESHOLD = 800
APPLETV_MOVE_DISTANCE = 3
APPLETV_MOVE_CONFLICT_DISTANCE = 20
APPLETV_SCROLL_DOUBLE_CONFLICT_DISTANCE = 5
APPLETV_SCROLL_CONFLICT_DISTANCE_FORWARD = 50
APPLETV_SCROLL_CONFLICT_DISTANCE_BACKWARD = 15
APPLETV_TOOLBAR_MOVE_DISTANCE = 50
APPLETV_PROMINENCE = 0.01
SAMSUNG_PROMINENCE = 0.05
MIN_DISTANCE = 12
KEY_SELECT_DISTANCE = 25
MIN_DOUBLE_MOVE_DISTANCE = 30
MOVE_BINARY_THRESHOLD = -70
WINDOW_SIZE = 1
KEY_SELECT_MOVE_DISTANCE = 20
SELECT_MOVE_DISTANCE = 30
MOVE_DELETE_THRESHOLD = 315
CHANGE_DIR_MAX_THRESHOLD = 150  # Very long delays may be a user just pausing, so we filter them out


def extract_move_directions(move_times: List[int]) -> Union[List[Direction], Direction]:
    if (len(move_times) <= 4):
        return Direction.ANY

    # List of at least length 4 diffs
    time_diffs = [ahead - behind for ahead, behind in zip(move_times[1:], move_times[:-1])]

    baseline_avg = np.average(time_diffs[0:-1])
    baseline_std = np.std(time_diffs[0:-1])
    cutoff = baseline_avg + 3 * baseline_std

    if (time_diffs[-1] >= cutoff) and (time_diffs[-1] < CHANGE_DIR_MAX_THRESHOLD):
        directions = [Direction.HORIZONTAL for _  in range(len(move_times) - 1)]
        directions.append(Direction.VERTICAL)
        return directions

    if len(time_diffs) < 5:
        return Direction.ANY

    baseline_avg = np.average(time_diffs[0:-2])
    baseline_std = np.std(time_diffs[0:-2])
    cutoff = baseline_avg + 3 * baseline_std

    if (time_diffs[-1] < cutoff) and (time_diffs[-2] >= cutoff and time_diffs[-2] < CHANGE_DIR_MAX_THRESHOLD):
        directions = [Direction.HORIZONTAL for _ in range(len(move_times) - 2)]
        directions.extend([Direction.VERTICAL, Direction.ANY])
        return directions

    return Direction.ANY


def create_spectrogram(audio: np.ndarray) -> np.ndarray:
    assert len(audio.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = signal.spectrogram(audio, fs=44100, nfft=1024, nperseg=256)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time


def clip_spectrogram(spectrogram: np.ndarray, params: MatchParams) -> np.ndarray:
    clipped_spectrogram = spectrogram[params.min_freq:params.max_freq, :]
    return clipped_spectrogram


def binarize_spectrogram(spectrogram: np.ndarray, threshold: float) -> np.ndarray:
    #avg, std = np.average(spectrogram), np.std(spectrogram)
    #normalized = (spectrogram - avg) / max(std, SMALL_NUMBER)
    #return (normalized >= threshold).astype(int)
    max_val, min_val = np.max(spectrogram), np.min(spectrogram)
    normalized = (spectrogram - min_val) / (max_val - min_val)
    return (normalized >= threshold).astype(int)


def moving_window_similarity(target: np.ndarray, known: np.ndarray, threshold: float) -> List[float]:
    # Make the time the first dimension
    target = target.T
    known = known.T

    # Binarize the known sound
    #known_binary = binarize_spectrogram(known, threshold=threshold)
    #known_masked = (known > threshold).astype(float) * known
    #target_masked = (target > threshold).astype(float) * target

    segment_size = known.shape[0]
    similarity: List[float] = []

    for start in range(target.shape[0]):
        end = start + segment_size
        target_segment = target[start:end]

        if len(target_segment) < segment_size:
            target_segment = np.pad(target_segment, pad_width=[(0, segment_size - len(target_segment)), (0, 0)], constant_values=0, mode='constant')

        # Normalize the spectrogram
        #target_binary = binarize_spectrogram(target_segment, threshold=threshold)

        # Compute the similarity score
        #alignment = target_binary * known_binary
        #sim_score = 2 * np.sum(alignment) / (np.sum(target_binary) + np.sum(known_binary))
        #similarity.append(sim_score)

        dist = np.sum(np.abs(known - target_segment))
        sim_score = 1.0 / max(dist, 1e-7)
        similarity.append(sim_score)

    # Smooth the result using a moving average
    smooth_filter = np.ones(shape=(WINDOW_SIZE, )) / WINDOW_SIZE
    similarity = signal.convolve(similarity, smooth_filter).astype(float).tolist()
    
    return similarity


class MoveExtractor:

    def __init__(self, tv_type: SmartTVType):
        directory = os.path.dirname(__file__)
        sound_directory = os.path.join(directory, '..', 'sounds', tv_type.name.lower())

        self._known_sounds: Dict[str, SoundProfile] = dict()
        self._tv_type = tv_type

        # Read in the start / end indices (known beforehand)
        config = read_json(os.path.join(sound_directory, 'config.json'))

        if tv_type == SmartTVType.SAMSUNG:
            self._sound_names = sounds.SAMSUNG_SOUNDS
        elif tv_type == SmartTVType.APPLE_TV:
            self._sound_names = sounds.APPLETV_SOUNDS
        else:
            raise ValueError('Unknown TV Type: {}'.format(tv_type.name))

        for sound in self._sound_names:

            # Read in the audio
            path = os.path.join(sound_directory, '{}.pkl.gz'.format(sound))
            audio = read_pickle_gz(path)

            # Compute the spectrograms
            spectrogram = create_spectrogram(audio[:, 0])

            # Parse the sound configuration and pre-compute the clipped spectrograms
            match_params_dict = config[sound]['match_params']
            
            match_params = MatchParams(min_freq=match_params_dict['min_freq'],
                                       max_freq=match_params_dict['max_freq'],
                                       threshold=match_params_dict['threshold'],
                                       time_delta=match_params_dict['time_delta'],
                                       freq_delta=match_params_dict['freq_delta'],
                                       time_tol=match_params_dict['time_tol'],
                                       freq_tol=match_params_dict['freq_tol'])

            spectrogram_clipped = clip_spectrogram(spectrogram, params=match_params)

            profile = SoundProfile(spectrogram=spectrogram_clipped,
                                   match_params=match_params,
                                   match_threshold=config[sound]['match_threshold'],
                                   match_buffer=config[sound]['match_buffer'],
                                   min_threshold=config[sound].get('min_threshold', 0.0))
            self._known_sounds[sound] = profile

    def compute_spectrogram_similarity_for_sound(self, spectrogram: np.ndarray, sound: str) -> List[float]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            spectrogram: A 2d spectrogram for the target audio signal
            sound: The name of the sound to use
        Returns:
            An array of the moving-window distances
        """
        assert sound in self._sound_names, 'Unknown sound {}. The sound must be one of {}'.format(sound, self._sound_names)
        assert len(spectrogram.shape) == 2,  'Must provide a 2d spectrogram. Got: {}'.format(spectrogram.shape)

        sound_profile = self._known_sounds[sound]
        match_params = sound_profile.match_params

        spectrogram_clipped = clip_spectrogram(spectrogram, params=match_params)

        #spectrogram_sim = moving_window_similarity(target=spectrogram_clipped,
        #                                           known=sound_profile.spectrogram,
        #                                           threshold=match_params.threshold)

        spectrogram_sim = match_constellations(target_spectrogram=spectrogram_clipped,
                                               ref_spectrogram=sound_profile.spectrogram,
                                               time_delta=match_params.time_delta,
                                               freq_delta=match_params.freq_delta,
                                               threshold=match_params.threshold,
                                               time_tol=match_params.time_tol,
                                               freq_tol=match_params.freq_tol)
        return spectrogram_sim

    def find_instances_of_sound(self, spectrogram: np.ndarray, sound: str) -> Tuple[List[int], List[float]]:
        """
        Finds instances of the given sound in the provided audio signal
        by finding peaks in the spectrogram distance chart.

        Args:
            spectrogram: A 2d spectrogram of the target audio signal
            sound: The name of the sound to find
        Return:
            A tuple of 3 elements:
                (1) A list of the `times` in which the peaks occur in the distance graph
                (2) A list of the peak values in the distance graph
        """
        assert sound in self._sound_names, 'Unknown sound {}. The sound must be one of {}'.format(sound, self._sound_names)

        similarity = self.compute_spectrogram_similarity_for_sound(spectrogram=spectrogram, sound=sound)

        sound_profile = self._known_sounds[sound]
        threshold = sound_profile.match_threshold

        if (self._tv_type == SmartTVType.SAMSUNG) and (sound == sounds.SAMSUNG_KEY_SELECT):
            distance = KEY_SELECT_DISTANCE
        elif (self._tv_type == SmartTVType.APPLE_TV) and (sound == sounds.APPLETV_KEYBOARD_MOVE):
            distance = APPLETV_MOVE_DISTANCE
        elif (self._tv_type == SmartTVType.APPLE_TV) and (sound == sounds.APPLETV_TOOLBAR_MOVE):
            distance = APPLETV_TOOLBAR_MOVE_DISTANCE
        elif sound in (sounds.SAMSUNG_DOUBLE_MOVE, sounds.APPLETV_KEYBOARD_DOUBLE_MOVE):
            distance = MIN_DOUBLE_MOVE_DISTANCE
        else:
            distance = MIN_DISTANCE

        prominence = SAMSUNG_PROMINENCE if (self._tv_type == SmartTVType.SAMSUNG) else APPLETV_PROMINENCE
        peaks, peak_properties = signal.find_peaks(x=similarity, height=threshold, distance=distance, prominence=(prominence, None))
        peak_heights = peak_properties['peak_heights']

        return peaks, peak_heights

    def extract_move_sequence(self, audio: np.ndarray, include_moves_to_done: bool) -> Tuple[List[Move], bool, KeyboardType]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
            include_moves_to_done: Whether to include the number of moves needed to reach the `done` key
        Returns:
            A tuple with three elements:
                (1) A list of moves before selections. The length of this list is the number of selections.
                (2) Whether the system finished on an autocomplete
                (3) The keyboard type to use
        """
        raise NotImplementedError()


class SamsungMoveExtractor(MoveExtractor):

    def __init__(self):
        super().__init__(SmartTVType.SAMSUNG)

    def extract_move_sequence(self, audio: np.ndarray, include_moves_to_done: bool) -> Tuple[List[Move], bool, KeyboardType]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        # Compute the spectrogram for this audio signal
        spectrogram = create_spectrogram(audio)

        key_select_times, _ = self.find_instances_of_sound(spectrogram, sound=sounds.SAMSUNG_KEY_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(key_select_times) == 0:
            return [], False, KeyboardType.SAMSUNG

        # Get occurances of the other sounds
        #delete_times, _ = self.find_instances_of_sound(spectrogram, sound=sounds.SAMSUNG_DELETE)
        move_times, _ = self.find_instances_of_sound(spectrogram, sound=sounds.SAMSUNG_MOVE)
        #raw_double_move_times, _ = self.find_instances_of_sound(spectrogram, sound=sounds.SAMSUNG_DOUBLE_MOVE)
        raw_select_times, _ = self.find_instances_of_sound(spectrogram, sound=sounds.SAMSUNG_SELECT)

        # Filter out duplicate moves
        #double_move_times: List[int] = filter_conflicts(target_times=raw_double_move_times,
        #                                                comparison_times=[select_times, delete_times],
        #                                                forward_distance=SELECT_MOVE_DISTANCE,
        #                                                backward_distance=SELECT_MOVE_DISTANCE)
        double_move_times: List[int] = []
        delete_times: List[int] = []

        select_times: List[int] = filter_conflicts(target_times=raw_select_times,
                                                   comparison_times=[move_times],
                                                   forward_distance=SELECT_MOVE_DISTANCE,
                                                   backward_distance=SELECT_MOVE_DISTANCE)

        # Get the ending sound times
        key_select_pairs = list(map(lambda t: (t, sounds.SAMSUNG_KEY_SELECT), key_select_times))
        delete_pairs = list(map(lambda t: (t, sounds.SAMSUNG_DELETE), delete_times))
        system_select_pairs = list(map(lambda t: (t, sounds.SAMSUNG_SELECT), select_times))

        end_time_pairs = list(sorted(key_select_pairs + delete_pairs + system_select_pairs, key=lambda t: t[0]))

        move_idx = 0
        double_move_idx = 0
        result: List[Move] = []

        for (end_time, end_sound) in end_time_pairs:
            # Record the times of each move in this window
            window_move_times: List[int] = []

            while (move_idx < len(move_times)) and (move_times[move_idx] < end_time):
                window_move_times.append(move_times[move_idx])
                move_idx += 1

            while (double_move_idx < len(double_move_times)) and (double_move_times[double_move_idx] < end_time):
                window_move_times.append(double_move_times[double_move_idx])
                window_move_times.append(double_move_times[double_move_idx])  # Append twice to account for the double move
                double_move_idx += 1

            # Sort the window move times
            window_move_times = list(sorted(window_move_times))
            num_moves = len(window_move_times)
            start_time = min(window_move_times) if num_moves > 0 else 0

            move = Move(num_moves=num_moves,
                        end_sound=end_sound,
                        directions=extract_move_directions(window_move_times),
                        start_time=start_time,
                        end_time=end_time)

            result.append(move)

        # Remove any moves of length 0 an end sound 'select' at the start. These are due to
        # opening the keyboard
        start_idx = 0
        while (start_idx < len(result)) and (result[start_idx].num_moves == 0) and (result[start_idx].end_sound == sounds.SAMSUNG_SELECT):
            start_idx += 1

        if len(result) <= start_idx:
            return [], False, KeyboardType.SAMSUNG

        result = result[start_idx:]

        # Clip the final move to done (if needed)
        if (not include_moves_to_done) and (result[-1].end_sound == sounds.SAMSUNG_SELECT):
            return result[0:-1], False, KeyboardType.SAMSUNG

        return result, False, KeyboardType.SAMSUNG

        # TODO: Include the 'done' sound here and track the number of move until 'done' as a way to find the
        # last key -> could be a good way around the randomized start key 'defense' on Samsung (APPLE TV search not suceptible)
        #if include_moves_to_done and (len(selects_after) > 0):
        #    start_time = moves_between[0] if len(moves_between) > 0 else 0
        #    move_to_done = Move(num_moves=len(moves_between), end_sound=sounds.SAMSUNG_SELECT, directions=Direction.ANY, start_time=start_time, end_time=next_select)
        #    result.append(move_to_done)

        ## TODO: Include tests for the 'done' autocomplete. On passwords with >= 8 characters, 1 move can mean
        ## <Done> at the end (no special sound), so give the option to stop early (verify with recordings)

        #return result, did_use_autocomplete, KeyboardType.SAMSUNG


class AppleTVMoveExtractor(MoveExtractor):

    def __init__(self):
        super().__init__(SmartTVType.APPLE_TV)

    def extract_move_sequence(self, audio: np.ndarray, include_moves_to_done: bool) -> Tuple[List[Move], bool, KeyboardType]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        # Get the raw keyboard select times
        raw_keyboard_select_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(raw_keyboard_select_times) == 0:
            return [], False, KeyboardType.APPLE_TV_SEARCH

        # Get occurances of the other sounds
        raw_keyboard_move_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_MOVE)
        raw_system_move_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_SYSTEM_MOVE)
        raw_keyboard_double_move_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_DOUBLE_MOVE)
        raw_keyboard_scroll_double_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE)
        raw_keyboard_scroll_triple_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_SCROLL_TRIPLE)
        keyboard_scroll_six_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_SCROLL_SIX)
        keyboard_delete_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_KEYBOARD_DELETE)
        raw_toolbar_move_times, _ = self.find_instances_of_sound(audio=audio, sound=sounds.APPLETV_TOOLBAR_MOVE)

        # Filter out conflicting sounds
        keyboard_select_times: List[int] = filter_conflicts(target_times=raw_keyboard_select_times,
                                                            comparison_times=[keyboard_delete_times],
                                                            forward_distance=MIN_DISTANCE,
                                                            backward_distance=MIN_DISTANCE)

        toolbar_move_times: List[int] = filter_conflicts(target_times=raw_toolbar_move_times,
                                                         comparison_times=[keyboard_select_times],
                                                         forward_distance=APPLETV_TOOLBAR_MOVE_DISTANCE,
                                                         backward_distance=APPLETV_TOOLBAR_MOVE_DISTANCE)

        system_move_times: List[int] = filter_conflicts(target_times=raw_system_move_times,
                                                        comparison_times=[keyboard_select_times, keyboard_delete_times],
                                                        forward_distance=MIN_DISTANCE,
                                                        backward_distance=MIN_DISTANCE)

        keyboard_double_move_times: List[int] = filter_conflicts(target_times=raw_keyboard_double_move_times,
                                                                 comparison_times=[system_move_times],
                                                                 forward_distance=MIN_DISTANCE,
                                                                 backward_distance=MIN_DISTANCE)

        keyboard_scroll_triple_times: List[int] = filter_conflicts(target_times=raw_keyboard_scroll_triple_times,
                                                                   comparison_times=[keyboard_scroll_six_times],
                                                                   forward_distance=MIN_DISTANCE,
                                                                   backward_distance=MIN_DISTANCE)

        keyboard_scroll_double_times: List[int] = filter_conflicts(target_times=raw_keyboard_scroll_double_times,
                                                                   comparison_times=[keyboard_scroll_six_times, keyboard_scroll_triple_times],
                                                                   forward_distance=MIN_DISTANCE,
                                                                   backward_distance=MIN_DISTANCE)

        keyboard_move_times: List[int] = filter_conflicts(target_times=raw_keyboard_move_times,
                                                          comparison_times=[keyboard_select_times, system_move_times, keyboard_delete_times, keyboard_double_move_times],
                                                          forward_distance=APPLETV_MOVE_CONFLICT_DISTANCE,
                                                          backward_distance=APPLETV_MOVE_CONFLICT_DISTANCE)

        keyboard_move_times = filter_conflicts(target_times=keyboard_move_times,
                                               comparison_times=[keyboard_scroll_double_times],
                                               forward_distance=APPLETV_SCROLL_DOUBLE_CONFLICT_DISTANCE,
                                               backward_distance=APPLETV_SCROLL_DOUBLE_CONFLICT_DISTANCE)

        keyboard_move_times = filter_conflicts(target_times=keyboard_move_times,
                                               comparison_times=[keyboard_scroll_six_times, keyboard_scroll_triple_times],
                                               forward_distance=APPLETV_SCROLL_CONFLICT_DISTANCE_FORWARD,
                                               backward_distance=APPLETV_SCROLL_CONFLICT_DISTANCE_BACKWARD)

        # Create the move sequence by (1) marking the end sounds and (2) calculating the number of moves in each block
        key_select_pairs = list(map(lambda t: (t, sounds.APPLETV_KEYBOARD_SELECT), keyboard_select_times))
        delete_pairs = list(map(lambda t: (t, sounds.APPLETV_KEYBOARD_DELETE), keyboard_delete_times))
        toolbar_move_pairs = list(map(lambda t: (t, sounds.APPLETV_TOOLBAR_MOVE), toolbar_move_times))

        end_time_pairs = list(sorted(key_select_pairs + delete_pairs + toolbar_move_pairs, key=lambda pair: pair[0]))

        #print('Keyboard Moves: {}'.format(keyboard_move_times))
        #print('Scroll Double: {}'.format(keyboard_scroll_double_times))
        #print('Scroll Six: {}'.format(keyboard_scroll_six_times))

        start_time = 0
        move_sequence: List[Move] = []

        for end_time_pair in end_time_pairs:
            end_time, end_sound = end_time_pair
            
            keyboard_moves = [t for t in keyboard_move_times if (t >= start_time) and (t <= end_time)]
            keyboard_double_moves = [t for t in keyboard_double_move_times if (t >= start_time) and (t <= end_time)]
            scroll_double_moves = [t for t in keyboard_scroll_double_times if (t >= start_time) and (t <= end_time)]
            scroll_triple_moves = [t for t in keyboard_scroll_triple_times if (t >= start_time) and (t <= end_time)]
            scroll_full_moves = [t for t in keyboard_scroll_six_times if (t >= start_time) and (t <= end_time)]

            num_moves = len(keyboard_moves) + 2 * len(keyboard_double_moves) + 2 * len(scroll_double_moves) + 3 * len(scroll_triple_moves) + 6 * len(scroll_full_moves)

            window_move_times = sorted(keyboard_moves + keyboard_double_moves + scroll_double_moves + scroll_triple_moves + scroll_full_moves)
            move_start_time = window_move_times[0] if len(window_move_times) > 0 else (move_sequence[-1].end_time if len(move_sequence) > 0 else 0)

            if (num_moves > 0) or (end_sound == sounds.APPLETV_KEYBOARD_SELECT):
                move = Move(num_moves=num_moves, end_sound=end_sound, directions=Direction.ANY, start_time=move_start_time, end_time=end_time)
                move_sequence.append(move)

            start_time = end_time

        ## Get the end time as the last system move / toolbar move
        last_keyboard_move = keyboard_move_times[-1]
        next_toolbar_moves = list(filter(lambda t: t > last_keyboard_move, toolbar_move_times))
        next_system_moves = list(filter(lambda t: t > last_keyboard_move, system_move_times))

        last_toolbar_move = next_toolbar_moves[0] if len(next_toolbar_moves) > 0 else BIG_NUMBER
        last_system_move = next_system_moves[0] if len(next_system_moves) > 0 else BIG_NUMBER
        end_time = min(last_toolbar_move, last_system_move)

        # We use the password keyboard both of the following hold:
        #   (1) The time between the first toolbar move and first keyboard move is "long"
        #   (2) The sound after the last keyboard move is a toolbar move
        keyboard_type = KeyboardType.APPLE_TV_SEARCH

        if len(next_toolbar_moves) > 0:
            first_keyboard_move = keyboard_move_times[0]
            first_toolbar_moves = list(filter(lambda t: t < first_keyboard_move, toolbar_move_times))

            if (len(next_system_moves) > 0) and (next_system_moves[0] < next_toolbar_moves[0]):
                keyboard_type = KeyboardType.APPLE_TV_SEARCH
            elif len(first_toolbar_moves) == 0:
                keyboard_type = KeyboardType.APPLE_TV_SEARCH
            elif (first_keyboard_move - first_toolbar_moves[-1]) < APPLETV_PASSWORD_THRESHOLD:
                keyboard_type = KeyboardType.APPLE_TV_SEARCH
            else:
                keyboard_type = KeyboardType.APPLE_TV_PASSWORD

        return move_sequence, False, keyboard_type


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-file', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    audio = SmartTVAudio(args.video_file)
    audio_signal = audio.get_audio()

    sound = sounds.SAMSUNG_MOVE

    extractor = SamsungMoveExtractor()

    spectrogram = create_spectrogram(audio_signal)
    similarity = extractor.compute_spectrogram_similarity_for_sound(spectrogram, sound=sound)
    instance_idx, instance_heights = extractor.find_instances_of_sound(spectrogram, sound=sound)

    move_seq, did_use_autocomplete, keyboard_type = extractor.extract_move_sequence(audio=audio_signal, include_moves_to_done=True)
    print('Move Sequence: {}'.format(list(map(lambda m: m.num_moves, move_seq))))
    print('Did use autocomplete: {}'.format(did_use_autocomplete))
    print('Keyboard type: {}'.format(keyboard_type.name))

    for idx, move in enumerate(move_seq):
        print('Move {}: {} ({}, {}), (Start: {}, End: {})'.format(idx, move.directions, move.num_moves, move.end_sound, move.start_time, move.end_time))

    with plt.style.context('seaborn-ticks'):
        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

        ax0.plot(list(range(audio_signal.shape[0])), audio_signal)
        ax1.plot(list(range(len(similarity))), similarity)
        ax1.scatter(instance_idx, instance_heights, marker='o', color='orange')

        ax0.set_xlabel('Time Step')
        ax0.set_ylabel('Audio Signal (dB)')
        ax0.set_title('Audio Signal for {}'.format(audio.file_name))

        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Match Similarity')
        ax1.set_title('Matches for Sound: {}'.format(' '.join((t.capitalize() for t in sound.split()))))

        plt.tight_layout()

        if args.output_file is not None:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
        else:
            plt.show()
