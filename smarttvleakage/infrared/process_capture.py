import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from scipy.io import wavfile
from scipy.sparse.csgraph import min_weight_full_biprartite_matching
from scipy.signal import find_peaks
from typing import Tuple, List

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.data_types import Move
from smarttvleakage.infrared.detect_remote import RemoteKey
from smarttvleakage.utils.file_utils import save_json
from smarttvleakage.utils.constants import Direction


RemotePress = namedtuple('RemotePress', ['time', 'key'])
THRESHOLD = 0.1


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

            if time < duration:
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


def create_move_sequence(audio_times: List[int], remote_presses: List[RemotePress], sample_rate: int) -> List[Move]:
    audio_idx, remote_idx = 0, 0

    num_moves = 0
    directions: List[Direction] = []
    moves: List[Move] = []

    while (audio_idx < len(audio_times)) and (remote_idx < len(remote_presses)):
        use_remote = False
        use_audio = False

        if audio_idx >= len(audio_times):
            use_remote = True
        elif remote_idx >= len(remote_presses):
            use_audio = True
        else:
            remote_event, audio_event = remote_presses[remote_idx], audio_times[audio_idx]
            audio_event /= sample_rate

            if abs(remote_event.time - audio_event) < THRESHOLD:
                use_remote = True
                use_audio = True
            elif (remote_event.time < audio_event):
                use_remote = True
            else:
                use_audio = True

        if use_remote:
            remote_event = remote_presses[remote_idx]

            # Only register movements for which we have the corresponding audio
            if use_audio:
                if remote_event.key == RemoteKey.SELECT:
                    move = Move(num_moves=num_moves,
                                directions=directions,
                                end_sound=sounds.SAMSUNG_KEY_SELECT)
                    moves.append(move)

                    num_moves = 0
                    directions = []
                else:
                    directions.append(to_direction(remote_event.key))
                    num_moves += 1

            remote_idx += 1

        if use_audio:
            if not use_remote:
                num_moves += 1
                directions.append(Direction.ANY)

            audio_idx += 1

    return moves


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--audio-path', type=str, required=True)
    parser.add_argument('--infrared-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--should-plot', action='store_true')
    args = parser.parse_args()

    sample_rate, audio = wavfile.read(args.audio_path)
    peak_times, peak_heights = get_tv_sound_times(audio)
    
    duration = int(max(peak_times) / sample_rate) + 1
    start_time = min(peak_times) / sample_rate
    remote_presses = read_ir_log(args.infrared_path, duration=duration, start_time=start_time)

    move_seq = create_move_sequence(audio_times=peak_times, remote_presses=remote_presses, sample_rate=sample_rate)
    
    # Save the results
    move_seq_dicts = list(map(lambda m: m.to_dict(), move_seq))
    save_json(move_seq_dicts, args.output_path)

    if args.should_plot:
        for idx, move in enumerate(move_seq):
            print('{}. {}'.format(idx + 1, move))

        fig, ax = plt.subplots()
        ax.plot(list(range(audio.shape[0])), np.abs(audio))
        ax.scatter(peak_times, peak_heights, marker='o', color='red')

        for remote_press in remote_presses:
            ax.axvline(remote_press.time * sample_rate, color='black')

        plt.show()
