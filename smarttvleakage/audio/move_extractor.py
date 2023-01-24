import numpy as np
import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from scipy.signal import find_peaks, convolve
from typing import Dict, List, Set, Tuple

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.audio_extractor import SmartTVAudio
from smarttvleakage.audio.data_types import Move
from smarttvleakage.audio.utils import get_sound_instances, create_spectrogram, extract_move_directions
from smarttvleakage.audio.utils import perform_match_spectrograms
from smarttvleakage.audio.constellations import compute_constellation_map
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz
from smarttvleakage.utils.constants import SmartTVType, Direction, BIG_NUMBER, SMALL_NUMBER


SMOOTH_FILTER_SIZE = 8
CONSTELLATION_THRESHOLD = 0.80
CONSTELLATION_THRESHOLD_END = 0.905
CONSTELLATION_TIME_DIST = 10
MOVE_FREQS = (2, 3, 4)
#END_THRESHOLD = 0.9

CONV_MODE = 'full'
TIME_TOL = 'time_tol'
FREQ_TOL = 'freq_tol'
TIME_DELTA = 'time_delta'
FREQ_DELTA = 'freq_delta'
PEAK_THRESHOLD = 'threshold'
MIN_FREQ = 'min_freq'
MIN_SIMILARITY = 'min_similarity'
START_TIME = 'start_time'
END_TIME = 'end_time'
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

            # Detect where the sound is in the reference spectrogram
            #max_energy = np.max(spectrogram[self.detection_freq_min:self.detection_freq_max, :], axis=0)  # [T]

            #if self.should_smooth_for_detection:
            #    smooth_filter = np.ones(shape=(SMOOTH_FILTER_SIZE, ), dtype=max_energy.dtype) / SMOOTH_FILTER_SIZE
            #    max_energy = convolve(max_energy, smooth_filter, mode=CONV_MODE)

            #start_times, end_times = get_sound_instances(max_energy=max_energy,
            #                                             threshold_factor=self.detection_threshold_factor,
            #                                             peak_height=self.detection_peak_height,
            #                                             peak_distance=self.detection_peak_distance,
            #                                             peak_prominence=self.detection_peak_prominence)

            min_freq = sound_config.get(MIN_FREQ, self.spectrogram_freq_min)

            #if (len(start_times) > 0) and (len(end_times) > 0):
            #    spectrogram = spectrogram[min_freq:self.spectrogram_freq_max, start_times[0]:end_times[0]]

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

            #fig, ax = plt.subplots()
            #ax.imshow(spectrogram, cmap='gray_r')
            #ax.scatter(peak_times, peak_freqs, marker='o', color='red')
            #ax.set_title(sound)
            #plt.show()

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
    def detection_factor(self) -> float:
        raise NotImplementedError()

    @property
    def detection_tolerance(self) -> float:
        raise NotImplementedError()

    @property
    def detection_peak_height(self) -> float:
        raise NotImplementedError()

    @property
    def detection_peak_distance(self) -> int:
        raise NotImplementedError()

    @property
    def detection_peak_prominence(self) -> float:
        raise NotImplementedError()

    @property
    def detection_merge_peak_factor(self) -> float:
        raise NotImplementedError()

    @property
    def smooth_detection_window_size(self) -> int:
        raise NotImplementedError()

    @property
    def tv_sounds(self) -> Set[str]:
        if self.tv_type == SmartTVType.SAMSUNG:
            return sounds.SAMSUNG_SOUNDS
        elif self.tv_type == SmartTVType.APPLE_TV:
            return sounds.APPLETV_SOUNDS
        else:
            raise ValueError('Unknown sounds for tv type: {}'.format(self.tv_type))

    def update_num_moves(self, sound: str, num_moves: int, move_times: List[int], start_time: int, end_time: int, current_results: List[Move]) -> Tuple[List[Move], int, List[int]]:
        raise NotImplementedError()

    def handle_matched_sound(self, sound: str, target_segment: np.ndarray, time: int) -> str:
        return sound

    def clean_move_sequence(self, move_seq: List[Move]) -> List[Move]:
        return move_seq

    def num_sound_instances(self, target_spectrogram: np.ndarray, target_sound: str) -> int:
        assert len(target_spectrogram.shape) == 2, 'Must provide a 2d spectrogram'
        assert target_sound in self.tv_sounds, 'Sound must be in: {}'.format(self.tv_sounds)

        # Find instances of any sounds for later matching
        clipped_target_spectrogram = target_spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max]
        clipped_target_spectrogram[clipped_target_spectrogram < -BIG_NUMBER] = 0.0

        start_times, end_times = get_sound_instances(spect=clipped_target_spectrogram,
                                                     forward_factor=self.detection_factor,
                                                     backward_factor=self.detection_factor,
                                                     peak_height=self.detection_peak_height,
                                                     peak_distance=self.detection_peak_distance,
                                                     peak_prominence=self.detection_peak_prominence,
                                                     smooth_window_size=self.smooth_detection_window_size,
                                                     tolerance=self.detection_tolerance,
                                                     merge_peak_factor=self.detection_merge_peak_factor)

        num_matches = 0

        for start_time, end_time in zip(start_times, end_times):
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

        start_times, end_times = get_sound_instances(spect=clipped_target_spectrogram,
                                                     forward_factor=self.detection_factor,
                                                     backward_factor=self.detection_factor,
                                                     peak_height=self.detection_peak_height,
                                                     peak_distance=self.detection_peak_distance,
                                                     peak_prominence=self.detection_peak_prominence,
                                                     smooth_window_size=self.smooth_detection_window_size,
                                                     tolerance=self.detection_tolerance,
                                                     merge_peak_factor=self.detection_merge_peak_factor)

        results: List[Move] = []
        move_times: List[int] = []
        current_num_moves = 0
        move_start_time, move_end_time = 0, 0

        #fig, ax = plt.subplots()
        #ax.imshow(clipped_target_spectrogram, cmap='gray_r')

        #ax1.plot(max_energy)
        #for t0, t1 in zip(start_times, end_times):
        #    ax1.axvline(t0, color='orange')
        #    ax1.axvline(t1, color='red')

        #plt.show()

        for start_time, end_time in zip(start_times, end_times):
            # If we haven't seen any moves, set the move start time
            if current_num_moves == 0:
                move_start_time = start_time

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

                #should_plot = (start_time > 975) and (sound == sounds.APPLETV_KEYBOARD_SELECT)
                #should_plot = (start_time >= 10800) and (start_time <= 11200) and (sound in (sounds.SAMSUNG_MOVE, sounds.SAMSUNG_DELETE))
                should_plot = False

                similarity = perform_match_spectrograms(first_spectrogram=normalized_target_segment,
                                                        second_spectrogram=self._ref_spectrograms[sound],
                                                        mask_threshold=sound_config[PEAK_THRESHOLD],
                                                        should_plot=False)

                if should_plot:
                    print('Sound: {}, Similarity: {:.4f}'.format(sound, similarity))

                    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

                    ax0.imshow(normalized_target_segment, cmap='gray_r')
                    ax1.imshow(self._ref_spectrograms[sound], cmap='gray_r')

                    ax0.set_title('Target')
                    ax1.set_title('Reference')

                    plt.show()

                if (similarity > best_sim):
                    best_sim = similarity
                    best_sound = sound

            #print('Best Sound: {}, Sim: {:.4f}, Time: {}'.format(best_sound, best_sim, start_time))
            #print('----------')

            # Skip sounds are are poor matches with all references
            if best_sim < self._config[best_sound][MIN_SIMILARITY]:
                continue

            # Handle TV-specific tie-breakers
            best_sound = self.handle_matched_sound(sound=best_sound, target_segment=normalized_spectrogram, time=start_time)

            #if (start_time >= 70800) and (start_time <= 71050):
            #    print('Best Sound: {}, Sim: {:.4f}'.format(best_sound, best_sim))

            #print('Adjusted Best Sound: {}, Time: {}'.format(best_sound, start_time))
            #print('==========')

            # Update the results
            current_time = int((start_time + end_time) / 2)
            results, current_num_moves, move_times = self.update_num_moves(sound=best_sound,
                                                                           num_moves=current_num_moves,
                                                                           move_times=move_times,
                                                                           start_time=move_start_time,
                                                                           current_time=current_time,
                                                                           current_results=results)
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

    @property
    def detection_factor(self) -> float:
        return 0.9

    @property
    def detection_tolerance(self) -> float:
        return 0.1

    @property
    def detection_peak_height(self) -> float:
        return 1.75

    @property
    def detection_peak_distance(self) -> int:
        return 2

    @property
    def detection_peak_prominence(self) -> int:
        return 0.1

    @property
    def detection_merge_peak_factor(self) -> float:
        return 1.12

    @property
    def smooth_detection_window_size(self) -> int:
        return 0

    def update_num_moves(self, sound: str, num_moves: int, move_times: List[int], start_time: int, current_time: int, current_results: List[Move]) -> Tuple[List[Move], int, List[int]]:
        if sound in (sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_DELETE, sounds.APPLETV_TOOLBAR_MOVE):
            move = Move(num_moves=num_moves,
                        end_sound=sound,
                        directions=Direction.ANY,  # TODO: Handle direction inference
                        start_time=start_time,
                        end_time=current_time,
                        move_times=move_times)

            current_results.append(move)
            num_moves = 0
            move_times = []
        elif sound in sounds.APPLETV_MOVE_SOUNDS:
            num_moves += sounds.APPLETV_MOVE_COUNTS[sound]
            move_times.append(current_time)
        else:
            raise ValueError('Unknown sound: {}'.format(sound))

        return current_results, num_moves, move_times

    def handle_matched_sound(self, sound: str, target_segment: np.ndarray, time: int) -> str:
        if (sound != sounds.APPLETV_KEYBOARD_MOVE):
            return sound

        # Get the constellations for `move`
        move_constellation = self._ref_constellation_maps[sound]
        move_spectrogram = self._ref_spectrograms[sound]
        sound_config = self._config[sound]

        # Compute the constellation for the target segment
        target_times, target_freqs = compute_constellation_map(spectrogram=target_segment,
                                                               freq_delta=sound_config[FREQ_DELTA],
                                                               time_delta=sound_config[TIME_DELTA],
                                                               threshold=CONSTELLATION_THRESHOLD)

        num_low_freq_peaks = 0
        sorted_peaks_by_time = list(sorted(zip(target_times, target_freqs), key=lambda x: x[0]))
        filtered_peaks = [(t, freq) for t, freq in sorted_peaks_by_time if (freq in MOVE_FREQS)]

        should_print = False

        for idx, (t, freq) in enumerate(filtered_peaks):
            curr_peak = target_segment[freq, t]

            if curr_peak >= CONSTELLATION_THRESHOLD_END:
                num_low_freq_peaks += 1
            elif (idx > 0) and (idx < (len(filtered_peaks) - 1)):
                prev_peak_time, prev_peak_freq = filtered_peaks[idx - 1]
                next_peak_time, next_peak_freq = filtered_peaks[idx + 1]

                prev_peak = target_segment[prev_peak_freq, prev_peak_time]
                next_peak = target_segment[next_peak_freq, next_peak_time]

                time_diff = max(abs(prev_peak_time - t), abs(next_peak_time - t))

                if (should_print):
                    print('Prev Peak Params: t -> {}, f -> {}, h -> {}'.format(prev_peak_time, prev_peak_freq, prev_peak))
                    print('Curr Peak Params: t -> {}, f -> {}, h -> {}'.format(t, freq, curr_peak))
                    print('Next Peak Params: t -> {}, f -> {}, h -> {}'.format(next_peak_time, next_peak_freq, next_peak))

                if (prev_peak >= CONSTELLATION_THRESHOLD_END) and (next_peak >= CONSTELLATION_THRESHOLD_END) and (time_diff <= CONSTELLATION_TIME_DIST):
                    num_low_freq_peaks += 1

        #num_low_freq_peaks = np.sum((target_freqs < 7).astype(int))

        if should_print:
            print(num_low_freq_peaks)

            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
            ax0.imshow(move_spectrogram, cmap='gray_r')
            ax0.scatter(move_constellation.peak_times, move_constellation.peak_freqs, marker='o', color='red')

            ax1.imshow(target_segment, cmap='gray_r')
            ax1.scatter(target_times, target_freqs, marker='o', color='red')

            plt.show()
            plt.close()

        if num_low_freq_peaks <= 1:
            return sounds.APPLETV_KEYBOARD_MOVE
        elif num_low_freq_peaks == 2:
            return sounds.APPLETV_KEYBOARD_SCROLL_DOUBLE
        elif num_low_freq_peaks == 3:
            return sounds.APPLETV_KEYBOARD_SCROLL_TRIPLE
        elif num_low_freq_peaks == 4:
            return sounds.APPLETV_KEYBOARD_SCROLL_FOUR
        elif num_low_freq_peaks == 5:
            return sounds.APPLETV_KEYBOARD_SCROLL_FIVE
        else:
            return sounds.APPLETV_KEYBOARD_SCROLL_SIX

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
                                                     peak_prominence=self.detection_peak_prominence)
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
    def detection_factor(self) -> float:
        return 0.9

    @property
    def detection_tolerance(self) -> float:
        return 0.02

    @property
    def detection_peak_height(self) -> float:
        return 1.8

    @property
    def detection_peak_distance(self) -> int:
        return 10

    @property
    def detection_peak_prominence(self) -> int:
        return 0.1

    @property
    def detection_merge_peak_factor(self) -> float:
        return 1.6

    @property
    def smooth_detection_window_size(self) -> int:
        return 8

    def update_num_moves(self, sound: str, num_moves: int, move_times: List[int], start_time: int, current_time: int, current_results: List[Move]) -> Tuple[List[Move], int, List[int]]:
        assert len(move_times) == num_moves, 'Got {} move times but {} moves'.format(len(move_times), num_moves)

        if sound in (sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_DELETE, sounds.SAMSUNG_SELECT):
            directions = extract_move_directions(move_times)

            move = Move(num_moves=num_moves,
                        end_sound=sound,
                        directions=directions,
                        start_time=start_time,
                        end_time=current_time,
                        move_times=move_times)

            current_results.append(move)
            num_moves = 0
            move_times = []
        elif sound == sounds.SAMSUNG_MOVE:
            num_moves += 1
            move_times.append(current_time)
        else:
            raise ValueError('Unknown sound: {}'.format(sound))

        return current_results, num_moves, move_times

    def clean_move_sequence(self, move_seq: List[Move]) -> List[Move]:
        if len(move_seq) == 0:
            return []

        cleaned: List[Move] = [move_seq[0]]

        for move_idx in range(1, len(move_seq)):
            prev_move = cleaned[-1]
            curr_move = move_seq[move_idx]

            should_merge = (curr_move.num_moves == 0) and (prev_move.end_sound == sounds.SAMSUNG_DELETE) and (curr_move.end_sound != sounds.SAMSUNG_DELETE)

            if should_merge:
                directions = prev_move.directions
                if isinstance(directions, list):
                    directions.append(Direction.ANY)

                merged = Move(num_moves=prev_move.num_moves + 1,
                              end_sound=curr_move.end_sound,
                              directions=directions,
                              start_time=prev_move.start_time,
                              end_time=curr_move.end_time,
                              move_times=prev_move.move_times + [prev_move.end_time] + curr_move.move_times)

                cleaned.pop(-1)
                cleaned.append(merged)
            else:
                cleaned.append(curr_move)

        return cleaned


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--tv-type', type=str, required=True)
    args = parser.parse_args()

    audio_extractor = SmartTVAudio(path=args.video_path)
    audio = audio_extractor.get_audio()
    target_spectrogram = create_spectrogram(audio)

    if args.tv_type.lower() == 'samsung':
        extractor = SamsungMoveExtractor()
    elif args.tv_type.lower() in ('appletv', 'apple-tv', 'apple_tv'):
        extractor = AppleTVMoveExtractor()
    else:
        raise ValueError('Unknown TV type: {}'.format(args.tv_type))

    moves = extractor.extract_moves(target_spectrogram)
    print(moves)
