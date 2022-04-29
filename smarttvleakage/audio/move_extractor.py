import os.path
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mp
from collections import namedtuple
from scipy.signal import spectrogram, find_peaks
from typing import List, Dict, Tuple

from smarttvleakage.utils.file_utils import read_json, read_pickle_gz


SoundProfile = namedtuple('SoundProfile', ['channel0', 'channel1', 'start', 'end'])

SOUNDS = ['move', 'select', 'key_select']
MIN_DISTANCE = 20
SOUND_THRESHOLDS = {
    'move': 0.003,
    'select': 0.004,
    'key_select': 0.013
}


def create_spectrogram(signal: np.ndarray) -> np.ndarray:
    assert len(signal.shape) == 1, 'Must provide a 1d signal'

    _, _, Sxx = spectrogram(signal, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)
    
    return Pxx


def moving_window_distances(target: np.ndarray, known: np.ndarray) -> List[float]:
    target = target.T
    known = known.T

    segment_size = known.shape[0]

    distances: List[float] = []
    for start in range(target.shape[0]):
        end = start + segment_size
        target_segment = target[start:end]

        if len(target_segment) < segment_size:
            target_segment = np.pad(target_segment, pad_width=[(0, segment_size - len(target_segment)), (0, 0)], constant_values=0, mode='constant')

        dist = np.linalg.norm(target_segment - known, ord=1)
        distances.append(1.0 / dist)

    return distances


class MoveExtractor:

    def __init__(self):
        directory = os.path.dirname(__file__)
        sound_directory = os.path.join(directory, '..', 'sounds')

        self._known_sounds: Dict[str, SoundProfile] = dict()

        # Read in the start / end indices (known beforehand)
        freq_range_dict = read_json(os.path.join(sound_directory, 'freq_ranges.json'))

        for sound in SOUNDS:
            path = os.path.join(sound_directory, '{}.pkl.gz'.format(sound))
            audio = read_pickle_gz(path)

            start, end = freq_range_dict[sound]['start'], freq_range_dict[sound]['end']
            channel0 = create_spectrogram(signal=audio[:, 0])
            channel1 = create_spectrogram(signal=audio[:, 1])

            profile = SoundProfile(channel0=channel0, channel1=channel1, start=start, end=end)
            self._known_sounds[sound] = profile

    def compute_spectrogram_distances(self, audio: np.ndarray) -> Dict[str, List[float]]:
        """
        Computes the sum of the absolute distances between the spectrogram
        of the given audio signal and those of the known sounds in a moving-window fashion.

        Args:
            audio: A 2d audio signal where the last dimension is the channel.
        Returns:
            A dictionary mapping known sound name -> moving-window distances
        """
        # Create the spectrogram from the known signal
        channel0 = create_spectrogram(signal=audio[:, 0])
        channel1 = create_spectrogram(signal=audio[:, 1])

        # For each sound type, compute the moving average distances
        result: Dict[str, np.ndarray] = dict()

        for sound in SOUNDS:
            sound_profile = self._known_sounds[sound]
            start, end = sound_profile.start, sound_profile.end

            channel0_dist = moving_window_distances(target=channel0[start:end],
                                                    known=sound_profile.channel0[start:end])

            channel1_dist = moving_window_distances(target=channel1[start:end],
                                                    known=sound_profile.channel1[start:end])

            distances = [c0 + c1 for c0, c1 in zip(channel0_dist, channel1_dist)]
            result[sound] = distances

        return result

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
        distances = self.compute_spectrogram_distances(audio=audio)[sound]
        
        peaks, peak_properties = find_peaks(x=distances, height=SOUND_THRESHOLDS[sound], distance=MIN_DISTANCE)
        peak_heights = peak_properties['peak_heights']

        return peaks, peak_heights

    def extract_move_sequence(self, audio: np.ndarray) -> List[int]:
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
            return []

        # Get occurances of the other two sounds
        move_idx, move_heights = self.find_instances_of_sound(audio=audio, sound='move')
        select_idx, select_heights = self.find_instances_of_sound(audio=audio, sound='select')

        # The first move starts before the first key select and after the nearest select
        first_key = key_select_idx[0]
        selects_before = list(filter(lambda i: i < first_key, select_idx))
        start_idx = selects_before[-1] if len(selects_before) > 0 else 0

        # Extract the number of moves between selections
        # TODO: Handle sequences with multiple keyboard interactions
        clipped_move_idx = list(filter(lambda i: i > start_idx, move_idx))

        key_idx = 0
        num_moves = 0
        result: List[int] = []

        for i in range(len(clipped_move_idx)):
            while (key_idx < len(key_select_idx)) and (clipped_move_idx[i] > key_select_idx[key_idx]):
                result.append(num_moves)
                key_idx += 1
                num_moves = 0

            if key_idx >= len(key_select_idx):
                break

            num_moves += 1
            i += 1

        return result


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/smart-tv-3-letter/dog.mp4')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    sound = 'move'

    extractor = MoveExtractor()
    distance_dict = extractor.compute_spectrogram_distances(audio=audio_signal)
    instance_idx, instance_heights = extractor.find_instances_of_sound(audio=audio_signal, sound=sound)

    move_seq = extractor.extract_move_sequence(audio=audio_signal)
    print('Move Sequence: {}'.format(move_seq))

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    ax0.plot(list(range(audio_signal.shape[0])), audio_signal[:, 0])
    ax1.plot(list(range(len(distance_dict[sound]))), distance_dict[sound])

    ax1.scatter(instance_idx, instance_heights, marker='o', color='orange')

    plt.show()



