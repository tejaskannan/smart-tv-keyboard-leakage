import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import os.path
from scipy.signal import spectrogram
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    channel0, channel1 = audio_signal[:, 0], audio_signal[:, 1]

    freq, times, Sxx = spectrogram(channel0, nperseg=1024)

    print(Sxx.shape)
    print(freq.shape)
    print(times.shape)

    #channel1_stft = stft(channel1, nperseg=256)

    #fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)

    #xs = list(range(len(channel0)))
    #ax0.plot(xs, np.abs(channel0_stft))
    #ax1.plot(xs, np.abs(channel1_stft))

    #ax0.set_ylabel('Magnitude')
    #ax1.set_ylabel('Magnitude')
    #ax1.set_xlabel('Freq')

    #ax.matshow(np.abs(Sxx))

    fig, ax = plt.subplots()

    Sxx, _, _, _ = ax.specgram(channel0)

    #ax1.plot(list(range(len(channel0))), channel0)

    file_name = os.path.basename(args.video_path)
    string = file_name.replace('.mp4', '').replace('.MOV', '')
    ax.set_title('Short Time Fourier Transform of the Audio Signal for {}'.format(string))

    plt.tight_layout()

    if args.output_path is None:
        plt.show()
    else:
        plt.savefig(args.output_path, bbox_inches='tight', transparent=True)
