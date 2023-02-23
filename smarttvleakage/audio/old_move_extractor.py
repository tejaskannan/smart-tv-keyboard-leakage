import os.path
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from enum import Enum, auto
from scipy.signal import spectrogram, find_peaks, convolve
from typing import List, Dict, Tuple, DefaultDict, Union

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.utils.constants import SmartTVType, BIG_NUMBER, KeyboardType, Direction, SMALL_NUMBER
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz, iterate_dir
from smarttvleakage.audio.constellations import compute_constellation_map, match_constellations
from smarttvleakage.audio.data_types import Move
from smarttvleakage.audio.utils import filter_conflicts, get_sound_instances, perform_match_constellations, perform_match_binary
from smarttvleakage.audio.audio_extractor import SmartTVAudio


MatchParams = namedtuple('MatchParams', ['min_threshold', 'max_threshold', 'min_freq', 'max_freq'])
SoundProfile = namedtuple('SoundProfile', ['spectrogram', 'match_params', 'match_threshold', 'match_buffer', 'min_threshold'])


APPLETV_PASSWORD_THRESHOLD = 800
APPLETV_MOVE_DISTANCE = 3
APPLETV_MOVE_CONFLICT_DISTANCE = 20
APPLETV_SCROLL_DOUBLE_CONFLICT_DISTANCE = 5
APPLETV_SCROLL_CONFLICT_DISTANCE_FORWARD = 50
APPLETV_SCROLL_CONFLICT_DISTANCE_BACKWARD = 15
APPLETV_TOOLBAR_MOVE_DISTANCE = 50
APPLETV_PROMINENCE = 0.01
SAMSUNG_PROMINENCE = 0.0009
MIN_DISTANCE = 12
KEY_SELECT_DISTANCE = 25
MIN_DOUBLE_MOVE_DISTANCE = 30
MOVE_BINARY_THRESHOLD = -70
WINDOW_SIZE = 8
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


def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    return Pxx  # X is frequency, Y is time


def compute_masked_spectrogram(spectrogram: float, params: MatchParams) -> np.ndarray:
    clipped_spectrogram = spectrogram[params.min_freq:params.max_freq, :]
    return np.logical_and(clipped_spectrogram >= params.min_threshold, clipped_spectrogram <= params.max_threshold).astype(int)


def moving_window_similarity(target: np.ndarray, known: np.ndarray) -> List[float]:
    target = target.T
    known = known.T

    segment_size = known.shape[0]
    similarity: List[float] = []

    for start in range(target.shape[0]):
        end = start + segment_size
        target_segment = target[start:end]

        if len(target_segment) < segment_size:
            target_segment = np.pad(target_segment, pad_width=[(0, segment_size - len(target_segment)), (0, 0)], constant_values=0, mode='constant')

        if np.all(target_segment == known):
            sim_score = 1.0
        else:
            sim_score = 2 * np.sum(target_segment * known) / (np.sum(target_segment) + np.sum(known))

        similarity.append(sim_score)

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

            # Read in the audio
            path = os.path.join(sound_directory, '{}.pkl.gz'.format(sound))
            audio = read_pickle_gz(path)

            # Compute the spectrograms
            spectrogram = create_spectrogram(audio[:, 0])

            # Parse the sound configuration and pre-compute the clipped spectrograms
            for match_params_dict in config[sound]['match_params']:
                match_params = MatchParams(min_freq=match_params_dict['min_freq'],
                                           max_freq=match_params_dict['max_freq'],
                                           min_threshold=match_params_dict['min_threshold'],
                                           max_threshold=match_params_dict['max_threshold'])

                spectrogram_clipped = compute_masked_spectrogram(spectrogram, params=match_params)

                profile = SoundProfile(spectrogram=spectrogram_clipped,
                                       match_params=match_params,
                                       match_threshold=config[sound]['match_threshold'],
                                       match_buffer=config[sound]['match_buffer'],
                                       min_threshold=config[sound].get('min_threshold', 0.0))
                self._known_sounds[sound].append(profile)

    def compute_spectrogram_similarity_for_sound(self, audio: np.ndarray, sound: str) -> List[float]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            audio: A 1d audio signal
            sound: The name of the sound to use
        Returns:
            An array of the moving-window distances
        """
        assert sound in self._sound_names, 'Unknown sound {}. The sound must be one of {}'.format(sound, self._sound_names)
        assert len(audio.shape) == 1,  'Must provide a 1d audio signal array. Got: {}'.format(audio.shape)

        # Create the spectrogram from the known signal
        spectrogram = create_spectrogram(signal=audio)

        # For each sound type, compute the moving average distances
        similarity_lists: List[List[float]] = []

        for sound_profile in self._known_sounds[sound]:
            match_params = sound_profile.match_params

            spectrogram_clipped = compute_masked_spectrogram(spectrogram, params=match_params)

            spectrogram_sim = moving_window_similarity(target=spectrogram_clipped,
                                                       known=sound_profile.spectrogram)
            similarity_lists.append(spectrogram_sim)

        return np.sum(similarity_lists, axis=0)

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
        cutoff_factor = 0.5

        if (self._tv_type == SmartTVType.SAMSUNG) and (sound == SAMSUNG_KEY_SELECT):
            distance = KEY_SELECT_DISTANCE
            cutoff_factor = 0.6
        elif (self._tv_type == SmartTVType.APPLE_TV) and (sound == APPLETV_KEYBOARD_MOVE):
            distance = APPLETV_MOVE_DISTANCE
        elif (self._tv_type == SmartTVType.APPLE_TV) and (sound == APPLETV_TOOLBAR_MOVE):
            distance = APPLETV_TOOLBAR_MOVE_DISTANCE
        elif sound in (SAMSUNG_DOUBLE_MOVE, APPLETV_KEYBOARD_DOUBLE_MOVE):
            distance = MIN_DOUBLE_MOVE_DISTANCE
        else:
            distance = MIN_DISTANCE

        #peaks, peak_properties = find_peaks(x=similarity, height=threshold, distance=2, prominence=(SOUND_PROMINENCE, None))
        peaks, peak_properties = find_peaks(x=similarity, height=threshold, distance=2)

        peak_heights = peak_properties['peak_heights']
        avg_height = np.average(peak_heights)
        std_height = np.std(peak_heights)
        max_height = np.max(peak_heights) if len(peak_heights) > 0 else 0.0
        min_height = np.min(peak_heights) if len(peak_heights) > 0 else 0.0

        cutoff = cutoff_factor * (max_height - threshold) + threshold
        adaptive_threshold = max(avg_height + sound_profile.match_buffer * std_height, cutoff)
        adaptive_threshold = max(adaptive_threshold, sound_profile.min_threshold)

        prominence = SAMSUNG_PROMINENCE if (self._tv_type == SmartTVType.SAMSUNG) else APPLETV_PROMINENCE
        filter_threshold = adaptive_threshold if (self._tv_type != SmartTVType.APPLE_TV) or (sound != APPLETV_KEYBOARD_MOVE) else threshold
        filtered_peaks, filtered_peak_properties = find_peaks(x=similarity, height=filter_threshold, distance=distance, prominence=(prominence, None))
        filtered_peak_heights = filtered_peak_properties['peak_heights']

        #print('Sound: {}, Threshold: {}, Adaptive Threshold: {}, Avg Height: {}, Std Height: {}'.format(sound, threshold, adaptive_threshold, avg_height, std_height))

        return filtered_peaks, filtered_peak_heights

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

    def get_next_break_sound(self, key_idx: int, key_times: List[int], sel_idx: int, sel_times: List[int], del_idx: int, del_times: List[int]) -> str:
        # Get the next time for each sound
        key_time = key_times[key_idx] if key_idx < len(key_times) else BIG_NUMBER
        sel_time = sel_times[sel_idx] if sel_idx < len(sel_times) else BIG_NUMBER
        del_time = del_times[del_idx] if del_idx < len(del_times) else BIG_NUMBER

        min_time = min(key_time, min(sel_time, del_time))
        
        if key_time == min_time:
            return SAMSUNG_KEY_SELECT
        elif sel_time == min_time:
            return SAMSUNG_SELECT
        elif del_time == min_time:
            return SAMSUNG_DELETE
        else:
            return SAMSUNG_KEY_SELECT

    def extract_move_sequence(self, audio: np.ndarray, include_moves_to_done: bool) -> Tuple[List[Move], bool, KeyboardType]:
        """
        Extracts the number of moves between key selections in the given audio sequence.

        Args:
            audio: A 2d aduio signal where the last dimension is the channel.
        Returns:
            A list of moves before selections. The length of this list is the number of selections.
        """
        raw_key_select_times, _ = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_KEY_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(raw_key_select_times) == 0:
            return [], False, KeyboardType.SAMSUNG

        # Get occurances of the other sounds
        delete_times, _ = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_DELETE)
        raw_move_times, _ = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_MOVE)
        raw_double_move_times, _ = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_DOUBLE_MOVE)
        select_times, _ = self.find_instances_of_sound(audio=audio, sound=SAMSUNG_SELECT)

        # Filter out moves using delete and select detection
        move_times: List[int] = []
        for t in raw_move_times:
            select_diff = np.abs(np.subtract(select_times, t)) if len(select_times) > 0 else BIG_NUMBER
            delete_diff = np.abs(np.subtract(delete_times, t)) if len(delete_times) > 0 else BIG_NUMBER

            if np.all(select_diff > SELECT_MOVE_DISTANCE) and np.all(delete_diff > SELECT_MOVE_DISTANCE):
                move_times.append(t)

        double_move_times: List[int] = []
        for t in raw_double_move_times:
            select_diff = np.abs(np.subtract(select_times, t)) if len(select_times) > 0 else BIG_NUMBER
            delete_diff = np.abs(np.subtract(delete_times, t)) if len(delete_times) > 0 else BIG_NUMBER

            if np.all(select_diff > SELECT_MOVE_DISTANCE) and np.all(delete_diff > SELECT_MOVE_DISTANCE):
                double_move_times.append(t)

        # Filter out any conflicting key and normal selects
        key_select_times: List[int] = []
        last_select = select_times[-1] if len(select_times) > 0 else BIG_NUMBER

        for t in raw_key_select_times:
            select_time_diff = np.abs(np.subtract(select_times, t))
            move_time_diff = np.abs(np.subtract(move_times, t))

            if np.all(select_time_diff > KEY_SELECT_DISTANCE) and np.all(move_time_diff > KEY_SELECT_DISTANCE):
                key_select_times.append(t)

        if len(key_select_times) == 0:
            return [], False, KeyboardType.SAMSUNG

        # The first move starts before the first key select and after the nearest select
        first_key_time = key_select_times[0]
        selects_before = list(filter(lambda t: t < first_key_time, select_times))
        start_time = (selects_before[-1] + MIN_DISTANCE) if len(selects_before) > 0 else 0

        # Extract the number of moves between selections
        # TODO: Handle sequences with multiple keyboard interactions
        clipped_move_times = list(filter(lambda t: t > start_time, move_times))
        clipped_double_move_times = list(filter(lambda t: t > start_time, double_move_times))
        clipped_select_times = list(filter(lambda t: (t > start_time) and any(True for s in key_select_times if s > t), select_times))
        clipped_delete_times = list(filter(lambda t: t > start_time, delete_times))

        break_indices: Dict[str, int] = {
            SAMSUNG_KEY_SELECT: 0,
            SAMSUNG_SELECT: 0,
            SAMSUNG_DELETE: 0
        }

        break_sound_times: Dict[str, List[int]] = {
            SAMSUNG_KEY_SELECT: key_select_times,
            SAMSUNG_SELECT: clipped_select_times,
            SAMSUNG_DELETE: clipped_delete_times
        }

        move_idx = 0
        start_move_idx = 0
        double_move_idx = 0
        key_idx, sel_idx, del_idx = 0, 0, 0
        num_moves = 0
        last_num_moves = 0
        result: List[int] = []
        window_move_times: List[int] = []

        while move_idx < len(clipped_move_times):

            # TODO: Fix tie-breaking bug. Only select a key select when it is the next in line.
            break_sound = self.get_next_break_sound(key_idx=break_indices[SAMSUNG_KEY_SELECT],
                                                    key_times=break_sound_times[SAMSUNG_KEY_SELECT],
                                                    sel_idx=break_indices[SAMSUNG_SELECT],
                                                    sel_times=break_sound_times[SAMSUNG_SELECT],
                                                    del_idx=break_indices[SAMSUNG_DELETE],
                                                    del_times=break_sound_times[SAMSUNG_DELETE])

            sound_idx = break_indices[break_sound]
            sound_times = break_sound_times[break_sound]

            while (sound_idx < len(sound_times)) and (clipped_move_times[move_idx] > sound_times[sound_idx]):
                # Add the move to the running list
                directions = extract_move_directions(window_move_times)
                start_time = clipped_move_times[start_move_idx] if num_moves > 0 else sound_times[sound_idx]
                result.append(Move(num_moves=num_moves, end_sound=break_sound, directions=directions, start_time=start_time, end_time=sound_times[sound_idx]))

                # Advance the sound index and get the next-nearest sound
                break_indices[break_sound] += 1

                break_sound = self.get_next_break_sound(key_idx=break_indices[SAMSUNG_KEY_SELECT],
                                                        key_times=break_sound_times[SAMSUNG_KEY_SELECT],
                                                        sel_idx=break_indices[SAMSUNG_SELECT],
                                                        sel_times=break_sound_times[SAMSUNG_SELECT],
                                                        del_idx=break_indices[SAMSUNG_DELETE],
                                                        del_times=break_sound_times[SAMSUNG_DELETE])

                sound_idx = break_indices[break_sound]
                sound_times = break_sound_times[break_sound]

                # Reset the state
                num_moves = 0
                start_move_idx = move_idx
                window_move_times = []

            window_move_times.append(clipped_move_times[move_idx])

            if (double_move_idx < len(clipped_double_move_times)) and (abs(clipped_double_move_times[double_move_idx] - clipped_move_times[move_idx]) <= MIN_DOUBLE_MOVE_DISTANCE):
                last_move_time = window_move_times[-1] if len(window_move_times) > 0 else 0
                window_move_times.append(last_move_time)  # Account for the double move

                move_idx += 1
                while (move_idx < len(clipped_move_times)) and (abs(clipped_double_move_times[double_move_idx] - clipped_move_times[move_idx]) <= MIN_DOUBLE_MOVE_DISTANCE):
                    move_idx += 1

                double_move_idx += 1
                num_moves += 2
            else:
                num_moves += 1
                move_idx += 1

        # Write out the last group if we haven't reached the last key
        key_idx, key_select_times = break_indices[SAMSUNG_KEY_SELECT], break_sound_times[SAMSUNG_KEY_SELECT]
        if key_idx < len(key_select_times):
            directions = extract_move_directions(window_move_times)
            start_time = clipped_move_times[start_move_idx] if num_moves > 0 else key_select_times[key_idx]
            result.append(Move(num_moves=num_moves, end_sound=SAMSUNG_KEY_SELECT, directions=directions, start_time=start_time, end_time=key_select_times[key_idx]))

        # If the last number of moves was 0 or 1, then the user leveraged the word autocomplete feature
        # NOTE: We can also validate this based on the number of possible moves (whether it was possible to get
        # to the top of the keyboard on this turn)
        last_key_select = key_select_times[-1]
        selects_after = list(filter(lambda t: t > last_key_select, select_times))
        next_select = selects_after[0] if len(selects_after) > 0 else BIG_NUMBER

        moves_between = list(filter(lambda t: (t <= next_select) and (t >= last_key_select), clipped_move_times))
        did_use_autocomplete = (len(moves_between) == 0) and (len(selects_after) > 0)

        if did_use_autocomplete and len(result) > 0:
            return result[0:-1], did_use_autocomplete, KeyboardType.SAMSUNG

        # TODO: Include the 'done' sound here and track the number of move until 'done' as a way to find the
        # last key -> could be a good way around the randomized start key 'defense' on Samsung (APPLE TV search not suceptible)
        if include_moves_to_done and (len(selects_after) > 0):
            start_time = moves_between[0] if len(moves_between) > 0 else 0
            move_to_done = Move(num_moves=len(moves_between), end_sound=SAMSUNG_SELECT, directions=Direction.ANY, start_time=start_time, end_time=next_select)
            result.append(move_to_done)

        # TODO: Include tests for the 'done' autocomplete. On passwords with >= 8 characters, 1 move can mean
        # <Done> at the end (no special sound), so give the option to stop early (verify with recordings)

        return result, did_use_autocomplete, KeyboardType.SAMSUNG


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
        raw_keyboard_select_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_SELECT)

        # Signals without any key selections do not interact with the keyboard
        if len(raw_keyboard_select_times) == 0:
            return [], False, KeyboardType.APPLE_TV_SEARCH

        # Get occurances of the other sounds
        raw_keyboard_move_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_MOVE)
        raw_system_move_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_SYSTEM_MOVE)
        raw_keyboard_double_move_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_DOUBLE_MOVE)
        raw_keyboard_scroll_double_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_SCROLL_DOUBLE)
        keyboard_scroll_six_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_SCROLL_TRIPLE)
        keyboard_delete_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_DELETE)
        raw_toolbar_move_times, _ = self.find_instances_of_sound(audio=audio, sound=APPLETV_TOOLBAR_MOVE)

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

        keyboard_scroll_double_times: List[int] = filter_conflicts(target_times=raw_keyboard_scroll_double_times,
                                                                   comparison_times=[keyboard_scroll_six_times],
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
                                               comparison_times=[keyboard_scroll_six_times],
                                               forward_distance=APPLETV_SCROLL_CONFLICT_DISTANCE_FORWARD,
                                               backward_distance=APPLETV_SCROLL_CONFLICT_DISTANCE_BACKWARD)

        # Create the move sequence by (1) marking the end sounds and (2) calculating the number of moves in each block
        key_select_pairs = list(map(lambda t: (t, APPLETV_KEYBOARD_SELECT), keyboard_select_times))
        delete_pairs = list(map(lambda t: (t, APPLETV_KEYBOARD_DELETE), keyboard_delete_times))
        toolbar_move_pairs = list(map(lambda t: (t, APPLETV_TOOLBAR_MOVE), toolbar_move_times))

        end_time_pairs = list(sorted(key_select_pairs + delete_pairs + toolbar_move_pairs, key=lambda pair: pair[0]))

        start_time = 0
        move_sequence: List[Move] = []

        for end_time_pair in end_time_pairs:
            end_time, end_sound = end_time_pair
            
            keyboard_moves = [t for t in keyboard_move_times if (t >= start_time) and (t <= end_time)]
            keyboard_double_moves = [t for t in keyboard_double_move_times if (t >= start_time) and (t <= end_time)]
            scroll_double_moves = [t for t in keyboard_scroll_double_times if (t >= start_time) and (t <= end_time)]
            scroll_full_moves = [t for t in keyboard_scroll_six_times if (t >= start_time) and (t <= end_time)]

            num_moves = len(keyboard_moves) + 2 * len(keyboard_double_moves) + 2 * len(scroll_double_moves) + 6 * len(scroll_full_moves)

            window_move_times = sorted(keyboard_moves + keyboard_double_moves + scroll_double_moves + scroll_full_moves)
            move_start_time = window_move_times[0] if len(window_move_times) > 0 else (move_sequence[-1].end_time if len(move_sequence) > 0 else 0)

            if (num_moves > 0) or (end_sound == APPLETV_KEYBOARD_SELECT):
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

    sound = SAMSUNG_KEY_SELECT

    extractor = SamsungMoveExtractor()
    similarity = extractor.compute_spectrogram_similarity_for_sound(audio=audio_signal, sound=sound)
    instance_idx, instance_heights = extractor.find_instances_of_sound(audio=audio_signal, sound=sound)

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