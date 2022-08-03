import moviepy.editor as mp
import matplotlib.pyplot as plt
import numpy as np
import os.path
from argparse import ArgumentParser
from scipy.signal import correlate, convolve, find_peaks

from smarttvleakage.utils.file_utils import read_pickle_gz


WINDOW_SIZE = 256


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--sound-path', type=str, required=True)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    channel0, channel1 = audio_signal[:, 0], audio_signal[:, 1]

    target_sound = read_pickle_gz(args.sound_path)

    sound_correlation = correlate(in1=audio_signal, in2=target_sound)
    sound_correlation_norm = np.sum(np.abs(sound_correlation), axis=-1)

    avg_filter = (1.0 / WINDOW_SIZE) * np.ones(shape=(WINDOW_SIZE, ))
    filtered_correlation = convolve(in1=sound_correlation_norm, in2=avg_filter)

    fig, ax = plt.subplots()

    xs = list(range(len(filtered_correlation)))
    ax.plot(xs, filtered_correlation)

    plt.show()


