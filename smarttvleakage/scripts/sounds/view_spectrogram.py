import moviepy.editor as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from smarttvleakage.audio.move_extractor import create_spectrogram


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    #parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio

    signal = audio.to_soundarray()
    channel0 = signal[:, 0]
    channel1 = signal[:, 1]

    spectrogram0 = create_spectrogram(channel0) 
    spectrogram1 = create_spectrogram(channel1)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    
    ax0.imshow(spectrogram0, cmap='gray_r')
    ax1.imshow(spectrogram1, cmap='gray_r')

    plt.show()

    #print('Mark an input time: ', end=' ')
    #start = int(input())

    #print('Mark and output time: ', end=' ')
    #end = int(input())

    #spectrogram0 = spectrogram0[:, start:end]
    #spectrogram1 = spectrogram1[:, start:end]
