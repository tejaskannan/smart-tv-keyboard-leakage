import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from matplotlib import image
from scipy.signal import spectrogram
from typing import List

from smarttvleakage.utils.file_utils import read_pickle_gz


def moving_window_distances(target: np.ndarray, known: np.ndarray) -> List[float]:
    target = target.T
    known = known.T

    segment_size = known.shape[0]

    distances: List[float] = []
    for start in range(target.shape[0] - segment_size):
        end = start + segment_size
        target_segment = target[start:end]
        dist = np.linalg.norm(target_segment - known, ord=1)
        distances.append(1.0 / dist)

    return distances


def create_spectrogram(samples: np.ndarray, path: str):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, Fs=2, noverlap=128)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--sound-file', type=str, required=True)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    channel0, channel1 = audio_signal[:, 0], audio_signal[:, 1]

    known_sound = read_pickle_gz(args.sound_file)
    known_sound_channel0 = known_sound[:, 0]

    freq, times, known_Sxx = spectrogram(known_sound_channel0, fs=44100, nfft=1024)
    known_Pxx = 10 * np.log10(known_Sxx)

    _, _, Sxx = spectrogram(channel0, fs=44100, nfft=1024)
    Pxx = 10 * np.log10(Sxx)

    #distances = moving_window_distances(target=Pxx[0:100], known=known_Pxx[0:100])

    plt.pcolormesh(times, freq[0:75], known_Pxx[0:75])
    plt.show()

    #fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    #ax0.plot(list(range(len(channel0))), channel0)
    #ax1.plot(list(range(len(distances))), distances)
    #plt.show()



