import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from scipy.io import wavfile
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.signal import find_peaks, convolve
from typing import Tuple, List, Dict, Any

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.data_types import Move
from smarttvleakage.audio.utils import create_spectrogram
from smarttvleakage.infrared.detect_remote import RemoteKey
from smarttvleakage.utils.file_utils import save_json
from smarttvleakage.utils.constants import Direction, BIG_NUMBER


RemotePress = namedtuple('RemotePress', ['time', 'key'])
SoundRange = namedtuple('SoundRange', ['start', 'end', 'peak_time', 'peak_height'])
THRESHOLD = 0.1
AUDIO_THRESHOLD = 1.0
WINDOW_SIZE = 256
TIME_THRESHOLD = 0.5

COLORS = {
    RemoteKey.UP: 'green',
    RemoteKey.DOWN: 'orange',
    RemoteKey.LEFT: 'black',
    RemoteKey.RIGHT: 'purple',
    RemoteKey.SELECT: 'gray',
    RemoteKey.BACK: 'yellow',
    RemoteKey.UNKNOWN: 'yellow'
}


def transform_audio_signal(audio: np.ndarray) -> np.ndarray:
    avg_signal = np.mean(audio)
    shifted_signal = np.abs(audio - avg_signal)
    avg_filter = np.ones(shape=(WINDOW_SIZE, ), dtype=audio.dtype) / WINDOW_SIZE
    return convolve(shifted_signal, avg_filter, mode='same')


def get_sound_ranges(audio: np.ndarray) -> List[SoundRange]:
    transformed_audio = transform_audio_signal(audio)

    audio_peaks, properties = find_peaks(transformed_audio, distance=1500, height=1.5)
    audio_peak_heights = properties['peak_heights']

    result: List[SoundRange] = []

    for peak, height in zip(audio_peaks, audio_peak_heights):
        start_idx = peak - 1
        while (start_idx >= 0) and (transformed_audio[start_idx] >= AUDIO_THRESHOLD):
            start_idx -= 1

        end_idx = peak + 1
        while (end_idx < len(transformed_audio)) and (transformed_audio[end_idx] >= AUDIO_THRESHOLD):
            end_idx += 1

        result.append(SoundRange(start=start_idx, end=end_idx, peak_time=peak, peak_height=height))

    return result


def create_split_points(sound_ranges: List[SoundRange]) -> List[int]:
    splits: List[int] = []

    for idx in range(1, len(sound_ranges)):
        prev = sound_ranges[idx - 1]
        curr = sound_ranges[idx]

        diff = curr.start - prev.end
        if (diff > 38000):
            split = int((curr.start + prev.end) / 2.0)
            splits.append(split)

    return splits


def get_tv_sound_times(audio: np.ndarray) -> Tuple[List[int], List[float]]:
    audio_peaks, properties = find_peaks(audio, distance=1500, height=135)
    audio_peak_heights = properties['peak_heights']

    filtered_peaks, filtered_heights = [], []
    for peak, height in zip(audio_peaks, audio_peak_heights):
        if height < 145:
            filtered_peaks.append(peak)
            filtered_heights.append(height)

    return filtered_peaks, filtered_heights


def read_ir_log(path: str, duration: int, start_time: float) -> List[RemotePress]:
    results: List[RemotePress] = []

    with open(path, 'r') as fin:
        for line in fin:
            tokens = line.split()
            time, key = float(tokens[0]), RemoteKey[tokens[1].upper()]

            if (time < duration) and (key != RemoteKey.BACK):
                press = RemotePress(time=time, key=key)
                results.append(press)

    min_time = min(map(lambda r: r.time, results))
    return [RemotePress(time=(r.time - min_time) + start_time, key=r.key) for r in results]


def to_direction(key: RemoteKey) -> Direction:
    if key == RemoteKey.UP:
        return Direction.UP
    elif key == RemoteKey.DOWN:
        return Direction.DOWN
    elif key == RemoteKey.LEFT:
        return Direction.LEFT
    elif key == RemoteKey.RIGHT:
        return Direction.RIGHT
    elif key == RemoteKey.UNKNOWN:
        return Direction.ANY
    else:
        raise ValueError('Cannot convert key {} to direction.'.format(key))


def create_instance_move_sequences(audio: np.ndarray, sound_ranges: List[SoundRange], remote_presses: List[RemotePress], split_times: List[int], sample_rate: int, min_length: int) -> List[List[Move]]:
    result: List[List[Move]] = []
    transformed = transform_audio_signal(audio)

    prev_time = 0
    for split_time in split_times:
        # Convert to seconds
        split_time_sec = int(split_time / sample_rate)
        prev_time_sec = int(prev_time / sample_rate)

        # Get the audio instances and IR reception for each split
        split_sound_ranges = [r for r in sound_ranges if (r.peak_time >= prev_time) and (r.peak_time < split_time)]
        split_remote_presses = [press for press in remote_presses if (press.time >= prev_time_sec) and (press.time < split_time_sec)]
        split_audio = clipped_audio[prev_time:split_time]

        # Idea: Split based on the `back` key if we see it
        split_audio_times = [float(r.peak_time / sample_rate) for r in split_sound_ranges]
        split_audio_heights = [r.peak_height for r in split_sound_ranges]

        # Match the audio to presses. This process accounts for `misses`
        move_seq = create_move_sequence(split_audio_times, audio_heights=split_audio_heights, remote_presses=split_remote_presses, sample_rate=sample_rate)

        split_audio = transformed[prev_time:split_time]

        #fig, ax = plt.subplots()
        #ax.plot(list(range(len(split_audio))), split_audio)
        #ax.scatter([r.peak_time - prev_time for r in split_sound_ranges], [r.peak_height for r in split_sound_ranges], marker='o', color='red')

        #for press in split_remote_presses:
        #    color = COLORS[press.key]
        #    ax.axvline(int(press.time * sample_rate) - prev_time, color=color)

        #plt.show()

        # Set the previous time to the end of the current split
        prev_time = split_time

        # Skip move sequences which are too small
        if len(move_seq) < min_length:
            continue

        result.append(move_seq)

    return result


def create_move_sequence(audio_times: List[int], audio_heights: List[int], remote_presses: List[RemotePress], sample_rate: int) -> List[Move]:
    remote_times = list(map(lambda press: press.time, remote_presses))

    # Compute the time differences and mask out any gaps above 0.5 seconds
    time_diffs = np.abs(np.expand_dims(audio_times, axis=-1) - np.expand_dims(remote_times, axis=0))  # [N, M]
    mask = (time_diffs < TIME_THRESHOLD).astype(time_diffs.dtype)
    time_diffs += (1.0 - mask) * BIG_NUMBER

    #biadj_matrix = csr_matrix(time_diffs)
    #row_idx, col_idx = min_weight_full_bipartite_matching(biadj_matrix)
    #row_idx_list = row_idx.astype(int).tolist()

    # Perform a matching in a greedy fashion
    match_indices: List[int] = []

    for audio_idx in range(len(audio_times)):
        # Get the mininmum time difference
        closest_remote_idx = np.argmin(time_diffs[audio_idx, :])
        closest_dist = time_diffs[audio_idx, closest_remote_idx]

        if (closest_dist < TIME_THRESHOLD):
            match_indices.append(closest_remote_idx)
            time_diffs[audio_idx, :] = BIG_NUMBER
            time_diffs[:, closest_remote_idx] = BIG_NUMBER
        else:
            match_indices.append(-1)

    #match_indices: List[int] = []
    #for audio_idx in range(len(audio_times)):
    #    try:
    #        match_idx = row_idx_list.index(audio_idx)
    #        match_indices.append(col_idx[match_idx])
    #    except ValueError:
    #        match_indices.append(-1)

    directions: List[Direction] = []
    moves: List[Move] = []

    for audio_idx, audio_time in enumerate(audio_times):
        # If we have matched against a remote press, then use the corresponding direction
        remote_idx = match_indices[audio_idx]

        if remote_idx != -1:
            remote_key = remote_presses[remote_idx].key
        else:
            remote_key = RemoteKey.UNKNOWN

        if remote_key == RemoteKey.SELECT:
            end_sound = sounds.SAMSUNG_KEY_SELECT if (audio_heights[audio_idx] < 5.0) else sounds.SAMSUNG_SELECT

            move = Move(num_moves=len(directions),
                        directions=directions,
                        end_sound=end_sound)
            moves.append(move)
            directions = []
        elif remote_key != RemoteKey.BACK:
            directions.append(to_direction(remote_key))

    return moves


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--audio-path', type=str, required=True)
    parser.add_argument('--infrared-path', type=str, required=True)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--should-plot', action='store_true')
    args = parser.parse_args()

    sample_rate, audio = wavfile.read(args.audio_path)
    peak_times, peak_heights = get_tv_sound_times(audio)
    
    duration = int(max(peak_times) / sample_rate) + 1
    start_time = min(peak_times) / sample_rate
    remote_presses = read_ir_log(args.infrared_path, duration=duration, start_time=start_time)

    max_time = max(peak_times) + sample_rate
    clipped_audio = audio[0:max_time]
    sound_ranges = get_sound_ranges(clipped_audio)
    split_times = create_split_points(sound_ranges)

    move_sequences = create_instance_move_sequences(clipped_audio, sound_ranges, remote_presses, split_times=split_times, sample_rate=sample_rate, min_length=3)

    # Save the results
    if args.output_path is not None:
        move_sequence_dicts: List[List[Dict[str, Any]]] = []

        for move_seq in move_sequences:
            move_sequence_dicts.append(list(map(lambda m: m.to_dict(), move_seq)))

        save_json(move_sequence_dicts, args.output_path)

    if args.should_plot:
        #for idx, move in enumerate(move_seq):
        #    print('{}. {}'.format(idx + 1, move))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(list(range(clipped_audio.shape[0])), clipped_audio)

        ax.set_xlabel('Sample')
        ax.set_title('Smart TV Audio')

        for split_point in split_times:
            ax.axvline(split_point, color='red')

        #for sound_range in sound_ranges:
        #    ax.axvline(sound_range.start, color='orange')
        #    ax.axvline(sound_range.end, color='red')

        #for remote_press in remote_presses:
        #    ax.axvline(remote_press.time * sample_rate, color='black')

        plt.show()
