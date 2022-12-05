import numpy as np
import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple
from scipy.signal import find_peaks, convolve
from typing import Dict, List

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.audio_extractor import SmartTVAudio
from smarttvleakage.audio.data_types import Move
from smarttvleakage.audio.utils import get_sound_instances, create_spectrogram
from smarttvleakage.audio.utils import perform_match_constellations, perform_match_binary, perform_match_constellations_geometry, perform_match_spectrograms
from smarttvleakage.audio.constellations import compute_constellation_map
from smarttvleakage.utils.file_utils import read_json, read_pickle_gz
from smarttvleakage.utils.constants import SmartTVType, Direction, BIG_NUMBER, SMALL_NUMBER


SMOOTH_FILTER_SIZE = 8
CONV_MODE = 'full'
TIME_TOL = 'time_tol'
FREQ_TOL = 'freq_tol'
TIME_DELTA = 'time_delta'
FREQ_DELTA = 'freq_delta'
PEAK_THRESHOLD = 'threshold'
MIN_FREQ = 'min_freq'
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

            if self.should_normalize:
                clipped_spectrogram = spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max, :]
                min_val, max_val = np.min(clipped_spectrogram), np.max(clipped_spectrogram)

            spectrogram = spectrogram[min_freq:self.spectrogram_freq_max, :]

            if self.should_normalize:
                spectrogram = (spectrogram - min_val) / (max_val - min_val)

            # Compute the constellations
            peak_times, peak_freqs = compute_constellation_map(spectrogram=spectrogram,
                                                               freq_delta=sound_config[FREQ_DELTA],
                                                               time_delta=sound_config[TIME_DELTA],
                                                               threshold=sound_config[PEAK_THRESHOLD])

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
    def detection_freq_min(self) -> int:
        raise NotImplementedError()

    @property
    def detection_freq_max(self) -> int:
        raise NotImplementedError()

    @property
    def detection_threshold_factor(self) -> int:
        raise NotImplementedError()

    @property
    def detection_peak_height(self) -> int:
        raise NotImplementedError()

    @property
    def detection_peak_distance(self) -> int:
        raise NotImplementedError()

    @property
    def detection_peak_prominence(self) -> int:
        raise NotImplementedError()

    @property
    def smooth_detection_window_size(self) -> int:
        raise NotImplementedError()

    def extract_moves(self, audio: np.ndarray) -> List[Move]:
        raise NotImplementedError()


class AppleTVMoveExtractor(MoveExtractor):

    @property
    def tv_type(self) -> SmartTVType:
        return SmartTVType.APPLE_TV

    @property
    def spectrogram_freq_min(self) -> int:
        return 5

    @property
    def spectrogram_freq_max(self) -> int:
        return 150

    @property
    def detection_freq_min(self) -> int:
        return 5

    @property
    def detection_freq_max(self) -> int:
        return 150

    @property
    def detection_threshold_factor(self) -> int:
        return 1.2

    @property
    def detection_peak_height(self) -> int:
        return -47

    @property
    def detection_peak_distance(self) -> int:
        return 2

    @property
    def detection_peak_prominence(self) -> int:
        return 0.0

    @property
    def smooth_detection_window_size(self) -> int:
        return 0

    @property
    def should_normalize(self) -> bool:
        return False

    def extract_moves(self, audio: np.ndarray) -> List[Move]:
        assert len(audio.shape) == 1, 'Must provide a 1d audio signal'

        # Compute the spectrogram of the target audio signal
        target_spectrogram = create_spectrogram(audio)  # [F, T]

        # Find instances of any sounds for later matching
        max_energy = np.max(target_spectrogram[self.detection_freq_min:self.detection_freq_max, :], axis=0)
        start_times, end_times = get_sound_instances(max_energy=max_energy,
                                                     threshold_factor=self.detection_threshold_factor,
                                                     peak_height=self.detection_peak_height,
                                                     peak_distance=self.detection_peak_distance,
                                                     peak_prominence=self.detection_peak_prominence)

        results: List[Move] = []
        current_num_moves = 0
        move_start_time, move_end_time = 0, 0

        fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        ax0.imshow(target_spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max, :], cmap='gray_r')

        ax1.plot(max_energy)
        for t0, t1 in zip(start_times, end_times):
            ax1.axvline(t0, color='orange')
            ax1.axvline(t1, color='red')

        plt.show()

        for start_time, end_time in zip(start_times, end_times):
            # Set the start time when beginning a new move
            if current_num_moves == 0:
                move_start_time = start_time

            # Get the spectrogram for this segment
            target_segment = target_spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max, start_time:end_time]

            # Find the closest known sound to the observed noise
            best_sim, second_best_sim = 0.0, 0.0
            best_sound, second_best_sound = None, None

            for sound in sorted(sounds.APPLETV_SOUNDS_EXTENDED):
                # Match the sound on the spectrograms
                sound_config = self._config[sound] if (sound not in sounds.APPLETV_MOVE_SOUNDS) else self._config[sounds.APPLETV_KEYBOARD_MOVE]

                # Compute the constellation maps
                target_times, target_freq = compute_constellation_map(spectrogram=target_segment,
                                                                      freq_delta=sound_config[FREQ_DELTA],
                                                                      time_delta=sound_config[TIME_DELTA],
                                                                      threshold=sound_config[PEAK_THRESHOLD])

                ref_constellation = self.get_reference_constellation(sound=sound)
                ref_times, ref_freq = ref_constellation.peak_times, ref_constellation.peak_freqs

                # Shift by the minimum time to avoid offset issues
                shifted_target_times = target_times - np.min(target_times)
                shifted_ref_times = ref_times - np.min(ref_times)

                similarity = perform_match_constellations(target_times=shifted_target_times,
                                                          target_freq=target_freq,
                                                          ref_times=shifted_ref_times,
                                                          ref_freq=ref_freq,
                                                          time_tol=sound_config.get(TIME_TOL, 1),
                                                          freq_tol=sound_config.get(FREQ_TOL, 1))

                #if (start_time >= 0) and (sound == sounds.APPLETV_KEYBOARD_SCROLL_TRIPLE):
                #    fig, ax = plt.subplots()

                #    ax.imshow(target_segment, cmap='gray_r')
                #    ax.scatter(target_times, target_freq, marker='o', color='red')
                #    ax.scatter(ref_times, ref_freq, marker='x', color='green')

                #    ax.set_title(sound)

                #    plt.show()

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

                best_spectrogram = self._ref_constellation_maps[best_sound].spectrogram
                second_best_spectrogram = self._ref_constellation_maps[second_best_sound].spectrogram

                best_ref_spectrogram_binary = (best_spectrogram > binary_threshold).astype(int)
                second_best_ref_spectrogram_binary = (second_best_spectrogram > binary_threshold).astype(int)
                target_spectrogram_binary = (target_segment > binary_threshold).astype(int)

                # Get the matching scores
                best_sim_binary = perform_match_binary(target_spectrogram_binary, best_ref_spectrogram_binary)
                second_best_sim_binary = perform_match_binary(target_spectrogram_binary, second_best_ref_spectrogram_binary)

                # Determine the best sound based on this `tie-breaker`
                best_sound = best_sound if (best_sim_binary > second_best_sim_binary) else second_best_sound
                best_sim = best_sim if (best_sim_binary > second_best_sim_binary) else second_best_sim

            # Set the end time of this move to the end of the most recent sound we have
            move_end_time = end_time

            print('Best Sound: {}, Best Similarity: {:.6f}'.format(best_sound, best_sim))

            if best_sound in (sounds.APPLETV_KEYBOARD_SELECT, sounds.APPLETV_KEYBOARD_DELETE, sounds.APPLETV_TOOLBAR_MOVE):
                move = Move(num_moves=current_num_moves,
                            end_sound=best_sound,
                            directions=Direction.ANY,
                            start_time=move_start_time,
                            end_time=move_end_time)

                results.append(move)
                current_num_moves = 0
            else:
                current_num_moves += sounds.APPLETV_MOVE_COUNTS[best_sound]

        return results

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
    def detection_freq_min(self) -> int:
        return 15

    @property
    def detection_freq_max(self) -> int:
        return 50

    @property
    def detection_forward_factor(self) -> float:
        return 0.9

    @property
    def detection_backward_factor(self) -> float:
        return 0.6

    @property
    def detection_peak_height(self) -> float:
        return 1.5

    @property
    def detection_peak_distance(self) -> int:
        return 10

    @property
    def detection_peak_prominence(self) -> int:
        return 0.1

    @property
    def smooth_detection_window_size(self) -> int:
        return 8

    @property
    def should_normalize(self) -> bool:
        return True

    def extract_moves(self, audio: np.ndarray) -> List[Move]:
        assert len(audio.shape) == 1, 'Must provide a 1d audio signal'

        # Threshold for 'moves'
        move_threshold = 0.24

        # Compute the spectrogram of the target audio signal
        target_spectrogram = create_spectrogram(audio)  # [F, T]

        # Find instances of any sounds for later matching
        clipped_target_spectrogram = target_spectrogram[self.spectrogram_freq_min:self.spectrogram_freq_max]
        clipped_target_spectrogram[clipped_target_spectrogram < -BIG_NUMBER] = 0.0

        start_times, end_times = get_sound_instances(spect=clipped_target_spectrogram,
                                                     forward_factor=self.detection_forward_factor,
                                                     backward_factor=self.detection_backward_factor,
                                                     peak_height=self.detection_peak_height,
                                                     peak_distance=self.detection_peak_distance,
                                                     peak_prominence=self.detection_peak_prominence,
                                                     smooth_window_size=self.smooth_detection_window_size)

        results: List[Move] = []
        current_num_moves = 0
        move_start_time, move_end_time = 0, 0

        #fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
        #ax0.imshow(clipped_target_spectrogram, cmap='gray_r')

        #ax1.plot(max_energy)
        #for t0, t1 in zip(start_times, end_times):
        #    ax1.axvline(t0, color='orange')
        #    ax1.axvline(t1, color='red')

        #plt.show()

        for start_time, end_time in zip(start_times, end_times):
            if current_num_moves == 0:
                move_start_time = start_time

            # Extract the target spectrogram during this window
            target_segment = target_spectrogram[0:self.spectrogram_freq_max, start_time:end_time]

            best_sim = 0.0
            move_sim = 0.0
            best_num_peaks_diff = BIG_NUMBER
            best_sound = None
            geometry_matches: Dict[str, float] = dict()

            clipped_spectrogram = target_segment[self.spectrogram_freq_min:self.spectrogram_freq_max, :]
            min_value, max_value = np.min(clipped_spectrogram), np.max(clipped_spectrogram)

            for sound in sorted(sounds.SAMSUNG_SOUNDS):
                # Match the sound on the spectrograms
                sound_config = self._config[sound]

                # Compute the constellation maps
                min_freq = sound_config.get(MIN_FREQ, self.spectrogram_freq_min)
                clipped_target_segment = target_segment[min_freq:, :]

                #min_value, max_value = np.min(clipped_target_segment), np.max(clipped_target_segment)
                clipped_target_segment = (clipped_target_segment - min_value) / (max_value - min_value)
                clipped_target_segment = (clipped_target_segment > sound_config[PEAK_THRESHOLD]).astype(float) * clipped_target_segment

                ref_segment = (self._ref_spectrograms[sound] > sound_config[PEAK_THRESHOLD]).astype(float) * self._ref_spectrograms[sound]

                similarity = perform_match_spectrograms(first_spectrogram=clipped_target_segment,
                                                       second_spectrogram=ref_segment)

                #if start_time >= 1500:
                #    print('Ref Sound: {}, Sim: {:.4f}'.format(sound, similarity))

                #    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

                #    ax0.imshow(clipped_target_segment, cmap='gray_r')
                #    ax1.imshow(ref_segment, cmap='gray_r')

                #    ax0.set_title('Target')
                #    ax1.set_title('Ref')

                #    plt.show()

                if sound == sounds.SAMSUNG_MOVE:
                    move_sim = similarity

                #if (similarity > best_sim) or ((abs(similarity - best_sim) < SMALL_NUMBER) and (num_peaks_diff < best_num_peaks_diff)):
                if (similarity > best_sim):
                    best_sim = similarity
                    best_sound = sound

            if (best_sound == sounds.SAMSUNG_DELETE) and (move_sim >= move_threshold):
                best_sound = sounds.SAMSUNG_MOVE
                best_sim = move_sim

            print('Best Sound: {}, Similarity: {:.5f}'.format(best_sound, best_sim))

            if best_sound in (sounds.SAMSUNG_KEY_SELECT, sounds.SAMSUNG_DELETE, sounds.SAMSUNG_SELECT):
                move = Move(num_moves=current_num_moves,
                            end_sound=best_sound,
                            directions=Direction.ANY,  # TODO: Handle direction inference
                            start_time=move_start_time,
                            end_time=move_end_time)

                results.append(move)
                current_num_moves = 0
            elif best_sound == sounds.SAMSUNG_MOVE:
                current_num_moves += 1
            else:
                raise ValueError('Unknown sound: {}'.format(best_sound))

        return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    args = parser.parse_args()

    audio_extractor = SmartTVAudio(path=args.video_path)
    audio = audio_extractor.get_audio()

    extractor = SamsungMoveExtractor()
    moves = extractor.extract_moves(audio)

    print(moves)
