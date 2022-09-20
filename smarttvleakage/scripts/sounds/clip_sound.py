import moviepy.editor as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from smarttvleakage.utils.file_utils import save_pickle_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio

    signal = audio.to_soundarray()
 
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    xs = list(range(len(signal)))
    ax0.plot(xs, signal[:, 0])
    ax1.plot(xs, signal[:, 1])

    plt.show()
    plt.close()

    print('Enter the start timestep: ')
    start = int(input())

    print('Enter the end timestep: ')
    end = int(input())

    assert end >= start, 'End must be >= start'

    clipped_signal = signal[start:end]

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)
    
    xs = list(range(len(clipped_signal)))
    ax0.plot(xs, clipped_signal[:, 0])
    ax1.plot(xs, clipped_signal[:, 1])

    plt.show()

    print('Do you want to save this clip? (y/n): ')
    confirm = input()

    if confirm.lower() in ('yes', 'y'):
        save_pickle_gz(clipped_signal, args.output_path)
