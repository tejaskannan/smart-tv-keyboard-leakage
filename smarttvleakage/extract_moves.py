import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy.signal import spectrogram


OFF_THRESHOLD = 0.001
MOVE_THRESHOLD = 0.15
SELECT_THRESHOLD = 0.02


def extract_moves(audio_signal: np.ndarray) -> int:
    channel0, channel1 = audio_signal[:, 0], audio_signal[:, 1]

    freq, time, Sxx = spectrogram(x=channel1)

    norms = np.linalg.norm(Sxx, axis=0, ord=2)  # [T]

    num_moves = 0
    move_starts: List[int] = []
    move_ends: List[int] = []

    start_idx = None
    for idx, amplitude_norm in enumerate(norms):
        if (amplitude_norm > MOVE_THRESHOLD) and (start_idx is None):
            start_idx = idx
            move_starts.append(idx)
        elif (amplitude_norm < OFF_THRESHOLD) and (start_idx is not None):
            num_moves += 1
            move_ends.append(idx)
            start_idx = None

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    ax0.plot(list(range(len(channel1))), channel1)
    ax1.plot(list(range(len(norms))), norms)

    (ymin, ymax) = ax1.get_ylim()

    for x in move_starts:
        ax1.axvline(x=x, ymin=ymin, ymax=ymax, color='black')

    for x in move_ends:
        ax1.axvline(x=x, ymin=ymin, ymax=ymax, color='red')

    plt.show()

    return num_moves


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio

    signal = audio.to_soundarray()
    num_moves = extract_moves(audio_signal=signal)

    print('Number of Moves: {}'.format(num_moves))
