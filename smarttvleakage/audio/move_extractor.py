import numpy as np
import os.path
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List, Set, Tuple

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.data_types import Move
from smarttvleakage.audio.utils import get_sound_instances_samsung, get_sound_instances_appletv, create_spectrogram
from smarttvleakage.audio.utils import perform_match_spectrograms, get_num_scrolls, extract_move_directions
from smarttvleakage.audio.utils import get_directions_appletv
from smarttvleakage.audio.constellations import compute_constellation_map
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz
from smarttvleakage.utils.constants import SmartTVType, Direction, BIG_NUMBER, SMALL_NUMBER


CONSTELLATION_THRESHOLD = 0.80
MOVE_FREQS = (2, 3, 4)

TIME_DELTA = 'time_delta'
FREQ_DELTA = 'freq_delta'
PEAK_THRESHOLD = 'threshold'
MIN_FREQ = 'min_freq'
MIN_SIMILARITY = 'min_similarity'
START_TIME = 'start_time'
END_TIME = 'end_time'
DEDUP_THRESHOLD = 'dedup_threshold'
ENERGY_THRESHOLD = 'energy_threshold'
DEDUP_BUFFER = 'dedup_buffer'
FORCE_THRESHOLD = 'force_threshold'
TIME_THRESHOLD = 'time_threshold'
Constellation = namedtuple('Constellation', ['peak_times', 'peak_freqs', 'spectrogram'])


class MoveExtractor:

    def __init__(self):
        # Read in the configurations
        current_dir = os.path.dirname(__file__)
        sound_folder = os.path.join(current_dir, '..', 'sounds', self.tv_type.name.lower())
        self._config = read_json(os.path.join(sound_folder, 'config.json'))

        # Pre-compute the constellation maps for each reference sound
        self._ref_constellation_maps: Dict[str, Constellation] = dict()
        self._ref_spectrograms: Dict[str, np.ndarray] = dict()

        for sound, sound_config in sorted(self._config.items()):
            # Read in the saved audio for this sound
            sound_path = os.path.join(sound_folder, '{}.pkl.gz'.format(sound))
            audio = read_pickle_gz(sound_path)[:, 0]  # [N]

            # Compute the spectrogram
            spectrogram = create_spectrogram(audio)  # [F, T]
            min_freq = sound_config.get(MIN_FREQ, self.spectrogram_freq_min)

            clipped_spectrogram = spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max, :]
            min_val, max_val = np.min(clipped_spectrogram), np.max(clipped_spectrogram)

            spectrogram = spectrogram[min_freq:self.spectrogram_freq_max, :]
            spectrogram = (spectrogram - min_val) / (max_val - min_val)

            if (START_TIME in sound_config) and (END_TIME in sound_config):
                start_time, end_time = sound_config[START_TIME], sound_config[END_TIME] + 1
                spectrogram = spectrogram[:, start_time:end_time]

            # Compute the constellations
            peak_times, peak_freqs = compute_constellation_map(spectrogram=spectrogram,
                                                               freq_delta=sound_config[FREQ_DELTA],
                                                               time_delta=sound_config[TIME_DELTA],
                                                               threshold=CONSTELLATION_THRESHOLD)

            self._ref_constellation_maps[sound] = Constellation(peak_times=peak_times,
                                                                peak_freqs=peak_freqs,
                                                                spectrogram=spectrogram)

            self._ref_spectrograms[sound] = spectrogram

    @property
    def tv_type(self) -> SmartTVType:
        raise NotImplementedError()

    @property
    def spectrogram_freq_min(self) -> int:
        raise NotImplementedError()

    @property
    def spectrogram_freq_max(self) -> int:
        raise NotImplementedError()

    @property
    def tv_sounds(self) -> Set[str]:
        if self.tv_type == SmartTVType.SAMSUNG:
            return sounds.SAMSUNG_SOUNDS
        elif self.tv_type == SmartTVType.APPLE_TV:
            return sounds.APPLETV_SOUNDS
        else:
            raise ValueError('Unknown sounds for tv type: {}'.format(self.tv_type))

    def update_num_moves(self, sound: str, num_moves: int, count: int, move_times: List[int], start_time: int, end_time: int, current_results: List[Move]) -> Tuple[List[Move], int, List[int]]:
        raise NotImplementedError()

    def get_sound_instances(self, spect: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, List[int]]:
        raise NotImplementedError()

    def handle_matched_sound(self, sound: str, target_segment: np.ndarray, time: int, prev_time: int, similarity_scores: Dict[str, float], max_segment_energy: float, cluster_size: int) -> Tuple[str, float]:
        return sound, similarity_scores[sound], 1

    def clean_move_sequence(self, move_seq: List[Move]) -> List[Move]:
        return move_seq

    def num_sound_instances(self, target_spectrogram: np.ndarray, target_sound: str) -> int:
        assert len(target_spectrogram.shape) == 2, 'Must provide a 2d spectrogram'
        assert target_sound in self.tv_sounds, 'Sound must be in: {}'.format(self.tv_sounds)

        # Find instances of any sounds for later matching
        clipped_target_spectrogram = target_spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max]
        clipped_target_spectrogram[clipped_target_spectrogram < -BIG_NUMBER] = 0.0

        start_times, end_times, _, cluster_sizes = self.get_sound_instances(spectrogram=clipped_target_spectrogram)

        num_matches = 0

        for start_time, end_time, cluster_size in zip(start_times, end_times, cluster_sizes):
            # Extract the target spectrogram during this window
            target_segment = target_spectrogram[0:self.spectrogram_freq_max, start_time:end_time]

            best_sim = 0.0
            best_sound = None

            # Normalize the spectrogram based on the full range (not a tighter range)
            clipped_spectrogram = target_segment[self.spectrogram_freq_min:self.spectrogram_freq_max, :]
            min_value, max_value = np.min(clipped_spectrogram), np.max(clipped_spectrogram)
            normalized_spectrogram = (clipped_spectrogram - min_value) / (max_value - min_value)

            for sound in sorted(self.tv_sounds):
                # Match the sound on the spectrograms
                sound_config = self._config[sound]

                # Compute the constellation maps
                min_freq = sound_config.get(MIN_FREQ, self.spectrogram_freq_min)
                min_freq_diff = min_freq - self.spectrogram_freq_min
                normalized_target_segment = normalized_spectrogram[min_freq_diff:, :]

                similarity = perform_match_spectrograms(first_spectrogram=normalized_target_segment,
                                                        second_spectrogram=self._ref_spectrograms[sound],
                                                        mask_threshold=sound_config[PEAK_THRESHOLD],
                                                        should_plot=False)

                if (similarity > best_sim):
                    best_sim = similarity
                    best_sound = sound

            # Skip sounds are are poor matches with all references
            if best_sim < self._config[best_sound][MIN_SIMILARITY]:
                continue

            if best_sound == target_sound:
                num_matches += 1

        return num_matches

    def extract_moves(self, target_spectrogram: np.ndarray) -> List[Move]:
        assert len(target_spectrogram.shape) == 2, 'Must provide a 2d spectrogram'

        # Find instances of any sounds for later matching
        clipped_target_spectrogram = target_spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max]
        clipped_target_spectrogram[clipped_target_spectrogram < -BIG_NUMBER] = 0.0

        start_times, end_times, max_energy, cluster_sizes = self.get_sound_instances(spectrogram=clipped_target_spectrogram)

        results: List[Move] = []
        move_times: List[int] = []
        current_num_moves = 0
        move_start_time, move_end_time = 0, 0
        prev_time = 0

        for start_time, end_time, cluster_size in zip(start_times, end_times, cluster_sizes):
            # If we haven't seen any moves, set the move start time
            if current_num_moves == 0:
                move_start_time = start_time

            # Extract the target spectrogram during this window
            target_segment = target_spectrogram[0:self.spectrogram_freq_max, start_time:end_time]
            max_segment_energy = np.max(max_energy[start_time:end_time])
            peak_time = np.argmax(max_energy[start_time:end_time]) + start_time

            best_sim = 0.0
            best_sound = None
            similarity_scores: Dict[str, float] = dict()

            # Normalize the spectrogram based on the full range (not a tighter range)
            clipped_spectrogram = target_segment[self.spectrogram_freq_min:self.spectrogram_freq_max, :]
            min_value, max_value = np.min(clipped_spectrogram), np.max(clipped_spectrogram)
            normalized_spectrogram = (clipped_spectrogram - min_value) / (max_value - min_value)

            for sound in sorted(self.tv_sounds):
                # Match the sound on the spectrograms
                sound_config = self._config[sound]

                # Compute the constellation maps
                min_freq = sound_config.get(MIN_FREQ, self.spectrogram_freq_min)
                min_freq_diff = min_freq - self.spectrogram_freq_min
                normalized_target_segment = normalized_spectrogram[min_freq_diff:, :]

                similarity = perform_match_spectrograms(first_spectrogram=normalized_target_segment,
                                                        second_spectrogram=self._ref_spectrograms[sound],
                                                        mask_threshold=sound_config[PEAK_THRESHOLD],
                                                        should_plot=False)

                similarity_scores[sound] = similarity  # Track the sim score for every sound

                if (similarity > best_sim) and (similarity > self._config[sound][MIN_SIMILARITY]):
                    best_sim = similarity
                    best_sound = sound

            # Handle TV-specific tie-breakers
            best_sound, best_sim, count = self.handle_matched_sound(sound=best_sound,
                                                                    target_segment=normalized_spectrogram,
                                                                    time=peak_time,
                                                                    prev_time=prev_time,
                                                                    similarity_scores=similarity_scores,
                                                                    max_segment_energy=max_segment_energy,
                                                                    cluster_size=cluster_size)

            # Skip sounds are are poor matches with all references
            if (best_sound is None) or (best_sim < self._config[best_sound][MIN_SIMILARITY]) or (max_segment_energy < self._config[best_sound][ENERGY_THRESHOLD]):
                continue

            # Update the results
            current_time = int((start_time + end_time) / 2)
            results, current_num_moves, move_times = self.update_num_moves(sound=best_sound,
                                                                           num_moves=current_num_moves,
                                                                           count=count,
                                                                           move_times=move_times,
                                                                           start_time=move_start_time,
                                                                           current_time=current_time,
                                                                           current_results=results)

            prev_time = peak_time

        return self.clean_move_sequence(results)


class AppleTVMoveExtractor(MoveExtractor):

    @property
    def tv_type(self) -> SmartTVType:
        return SmartTVType.APPLE_TV

    @property
    def spectrogram_freq_min(self) -> int:
        return 5

    @property
    def spectrogram_freq_max(self) -> int:
        return 50

    def get_sound_instances(self, spectrogram: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, List[int]]:
        return get_sound_instances_appletv(spectrogram, smooth_window_size=8)

    def update_num_moves(self, sound: str, num_moves: int, count: int, move_times: List[int], start_time: int, current_time: int, current_results: List[Move]) -> Tuple[List[Move], int, List[int]]:
        if sound in (sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_DELETE, sounds.APPLETV_TOOLBAR_MOVE):
            directions = get_directions_appletv(move_times)

            move = Move(num_moves=num_moves,
                        end_sound=sound,
                        directions=directions,
                        start_time=start_time,
                        end_time=current_time,
                        move_times=move_times,
                        num_scrolls=get_num_scrolls(move_times, 3))

            current_results.append(move)
            num_moves = 0
            move_times = []
        elif sound in sounds.APPLETV_MOVE_SOUNDS:
            num_moves += count

            for _ in range(count):
                move_times.append(current_time)
        else:
            raise ValueError('Unknown sound: {}'.format(sound))

        return current_results, num_moves, move_times

    def handle_matched_sound(self, sound: str, target_segment: np.ndarray, time: int, prev_time: int, similarity_scores: Dict[str, float], max_segment_energy: float, cluster_size: int) -> Tuple[str, float]:
        if sound is None:
            return sound, 0.0, 0
        elif (sound != sounds.APPLETV_KEYBOARD_MOVE):
            return sound, similarity_scores[sound], 1
        elif (similarity_scores[sounds.APPLETV_KEYBOARD_MOVE] < 30.0) and (similarity_scores[sounds.APPLETV_TOOLBAR_MOVE] > 10.25):
            return sounds.APPLETV_TOOLBAR_MOVE, similarity_scores[sounds.APPLETV_TOOLBAR_MOVE], 1

        # Get the constellations for `move`
        move_constellation = self._ref_constellation_maps[sound]
        move_spectrogram = self._ref_spectrograms[sound]
        sound_config = self._config[sound]

        # Compute the constellation for the target segment
        constellation_threshold = 0.9
        peak_cutoff = 0.96
        similarity_factor = 0.97  # Within 3% of adjacent (if nearby)
        time_cutoff = 7

        target_times, target_freqs = compute_constellation_map(spectrogram=target_segment,
                                                               freq_delta=sound_config[FREQ_DELTA],
                                                               time_delta=sound_config[TIME_DELTA],
                                                               threshold=constellation_threshold)

        num_low_freq_peaks = 0
        sorted_peaks_by_time = list(sorted(zip(target_times, target_freqs), key=lambda x: x[0]))
        low_freq_peaks = [(t, freq) for t, freq in sorted_peaks_by_time if (freq in MOVE_FREQS)]

        if len(low_freq_peaks) == 0:
            return sound, similarity_scores[sound], 0

        # It is possible for neighboring peaks to have the same value, and the constellation map will
        # include both. This is not ideal, so we re-check the results to keep only one peak within the
        # time delta
        filtered_peaks: List[Tuple[int, int]] = [low_freq_peaks[0]]

        for idx in range(1, len(low_freq_peaks)):
            prev_time, prev_freq = filtered_peaks[-1]
            curr_time, curr_freq = low_freq_peaks[idx]

            if (curr_time - prev_time) >= sound_config[TIME_DELTA]:
                filtered_peaks.append((curr_time, curr_freq))

        for idx, (t, freq) in enumerate(filtered_peaks):
            curr_peak = target_segment[freq, t]

            if (idx > 0) and (idx < (len(filtered_peaks) - 1)):
                prev_peak_time, prev_peak_freq = filtered_peaks[idx - 1]
                next_peak_time, next_peak_freq = filtered_peaks[idx + 1]

                prev_peak = target_segment[prev_peak_freq, prev_peak_time]
                next_peak = target_segment[next_peak_freq, next_peak_time]

                time_diff = max(abs(prev_peak_time - t), abs(next_peak_time - t))

                if (curr_peak >= peak_cutoff) or (((t - prev_peak_time) > time_cutoff) or (curr_peak >= (similarity_factor * prev_peak))) and (((next_peak_time - t) > time_cutoff) or (curr_peak >= (similarity_factor * next_peak))):
                    num_low_freq_peaks += 1
            elif (idx == (len(filtered_peaks) - 1)):
                prev_peak_time, prev_peak_freq = filtered_peaks[idx - 1]
                prev_peak = target_segment[prev_peak_freq, prev_peak_time]

                if (curr_peak > peak_cutoff) or ((t - prev_peak_time) > time_cutoff) or (curr_peak >= (similarity_factor * prev_peak)):
                    num_low_freq_peaks += 1
            else:
                num_low_freq_peaks += 1

        best_sim = similarity_scores[sounds.APPLETV_KEYBOARD_MOVE]
        return sounds.APPLETV_KEYBOARD_MOVE, best_sim, max(num_low_freq_peaks, 1)

    def get_reference_constellation(self, sound: str) -> Constellation:
        if sound in sounds.APPLETV_SOUNDS:
            return self._ref_constellation_maps[sound]

        # Stitch together multiple moves
        num_reps = sounds.APPLETV_MOVE_COUNTS[sound]
        base_spectrogram = self._ref_constellation_maps[sounds.APPLETV_KEYBOARD_MOVE].spectrogram

        # Clip the spectrogram further to better match the observed 'scrolling' sounds
        max_energy = np.max(base_spectrogram, axis=0)
        start_times, end_times = get_sound_instances(max_energy=max_energy,
                                                     threshold_factor=1.2,
                                                     peak_height=self.detection_peak_height,
                                                     peak_distance=self.detection_peak_distance,
                                                     peak_prominence=self.detection_peak_prominence,
                                                     max_merge_height=self.detection_max_merge_height)
        base_spectrogram = base_spectrogram[:, start_times[0]:end_times[0]]

        spectrogram = np.concatenate([base_spectrogram for _ in range(num_reps)], axis=-1)

        # Recompute the constellations
        sound_config = self._config[sounds.APPLETV_KEYBOARD_MOVE]
        peak_times, peak_freqs = compute_constellation_map(spectrogram=spectrogram,
                                                           freq_delta=sound_config[FREQ_DELTA],
                                                           time_delta=sound_config[TIME_DELTA],
                                                           threshold=sound_config[PEAK_THRESHOLD])

        return Constellation(peak_times=peak_times, peak_freqs=peak_freqs, spectrogram=spectrogram)


class SamsungMoveExtractor(MoveExtractor):

    @property
    def tv_type(self) -> SmartTVType:
        return SmartTVType.SAMSUNG

    @property
    def spectrogram_freq_min(self) -> int:
        return 5

    @property
    def spectrogram_freq_max(self) -> int:
        return 50

    @property
    def smooth_window_size(self) -> int:
        return 8  # 8

    def get_sound_instances(self, spectrogram: np.ndarray) -> Tuple[List[int], List[int], np.ndarray, List[int]]:
        return get_sound_instances_samsung(spectrogram, smooth_window_size=self.smooth_window_size)

    def handle_matched_sound(self, sound: str, target_segment: np.ndarray, time: int, prev_time: int, similarity_scores: Dict[str, float], max_segment_energy: float, cluster_size: int) -> Tuple[str, float]:
        if sound is None:
            return sound, 0.0, 0

        if (cluster_size > 1) and (sound in (sounds.SAMSUNG_MOVE, sounds.SAMSUNG_DELETE)):
            return sounds.SAMSUNG_MOVE, similarity_scores[sounds.SAMSUNG_MOVE], cluster_size

        best_sim = similarity_scores[sound]
        key_select_sim = similarity_scores[sounds.SAMSUNG_KEY_SELECT]

        if (sound == sounds.SAMSUNG_SELECT):
            # Deduplicate with key selection using an alternative threshold
            key_select_sim = similarity_scores[sounds.SAMSUNG_KEY_SELECT]

            if (max_segment_energy < self._config[sounds.SAMSUNG_SELECT][ENERGY_THRESHOLD]) and (key_select_sim > self._config[sounds.SAMSUNG_KEY_SELECT][MIN_SIMILARITY]):
                return sounds.SAMSUNG_KEY_SELECT, key_select_sim, 1
        elif (sound == sounds.SAMSUNG_MOVE):
            select_sim = similarity_scores[sounds.SAMSUNG_SELECT]
            key_select_sim = similarity_scores[sounds.SAMSUNG_KEY_SELECT]
            key_select_buffer = self._config[sounds.SAMSUNG_KEY_SELECT][DEDUP_THRESHOLD]

            if (key_select_sim >= self._config[sounds.SAMSUNG_KEY_SELECT][MIN_SIMILARITY]) and ((key_select_sim + key_select_buffer) >= best_sim):
                return sounds.SAMSUNG_KEY_SELECT, key_select_sim, 1
            elif (best_sim < self._config[sound][MIN_SIMILARITY]) and (select_sim >= self._config[sounds.SAMSUNG_SELECT][MIN_SIMILARITY]):
                return sounds.SAMSUNG_SELECT, select_sim, 1
        elif (sound == sounds.SAMSUNG_DELETE):
            move_sim = similarity_scores[sounds.SAMSUNG_MOVE]
            move_threshold = self._config[sounds.SAMSUNG_MOVE][DEDUP_THRESHOLD]
            force_threshold = self._config[sounds.SAMSUNG_MOVE][FORCE_THRESHOLD]
            move_buffer = self._config[sounds.SAMSUNG_MOVE][DEDUP_BUFFER]
            delete_threshold = self._config[sounds.SAMSUNG_DELETE][DEDUP_THRESHOLD]

            time_buffer = self._config[sounds.SAMSUNG_DELETE][TIME_THRESHOLD]

            if (best_sim >= (3.0 * move_sim)):
                return sound, best_sim, 1

            if ((time - prev_time) <= time_buffer) or ((move_sim > force_threshold) and ((move_sim + 3.0 * move_buffer) > best_sim)) or ((move_sim > move_threshold) and ((best_sim < delete_threshold) or ((move_sim + move_buffer) > best_sim))):
                return sounds.SAMSUNG_MOVE, move_sim, 1

        return sound, similarity_scores[sound], 1

    def update_num_moves(self, sound: str, num_moves: int, count: int, move_times: List[int], start_time: int, current_time: int, current_results: List[Move]) -> Tuple[List[Move], int, List[int]]:
        assert len(move_times) == num_moves, 'Got {} move times but {} moves'.format(len(move_times), num_moves)

        if sound in (sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_DELETE, sounds.SAMSUNG_SELECT):
            directions = extract_move_directions(move_times)

            move = Move(num_moves=num_moves,
                        end_sound=sound,
                        directions=directions,
                        start_time=start_time,
                        end_time=current_time,
                        move_times=move_times,
                        num_scrolls=get_num_scrolls(move_times, 4))

            current_results.append(move)
            num_moves = 0
            move_times = []
        elif sound == sounds.SAMSUNG_MOVE:
            num_moves += count
            for _ in range(count):
                move_times.append(current_time)
        else:
            raise ValueError('Unknown sound: {}'.format(sound))

        return current_results, num_moves, move_times

    def clean_move_sequence(self, move_seq: List[Move]) -> List[Move]:
        if len(move_seq) == 0:
            return []

        first_move = move_seq[0]
        cleaned: List[Move] = [first_move]
        move_carry = 0
        start_idx = 1

        for move_idx in range(start_idx, len(move_seq)):
            prev_move = cleaned[-1]
            curr_move = move_seq[move_idx]

            # It is impossible to go from a non-delete sound to a delete sound in 0 moves. Since `delete` sounds look similar to those of `move`
            # we merge this `delete` sound into the NEXT sequence element by adding a single move carry
            should_merge = (curr_move.num_moves == 0) and (prev_move.end_sound != sounds.SAMSUNG_DELETE) and (curr_move.end_sound == sounds.SAMSUNG_DELETE)

            if should_merge:
                move_carry = 1
            elif move_carry == 1:
                directions = prev_move.directions
                if isinstance(directions, list):
                    directions = [Direction.ANY] + directions

                merged = Move(num_moves=curr_move.num_moves + move_carry,
                              end_sound=curr_move.end_sound,
                              directions=directions,
                              start_time=prev_move.start_time,
                              end_time=curr_move.end_time,
                              move_times=prev_move.move_times + [prev_move.end_time] + curr_move.move_times)

                cleaned.append(merged)
                move_carry = 0
            else:
                cleaned.append(curr_move)
                move_carry = 0

        return cleaned
