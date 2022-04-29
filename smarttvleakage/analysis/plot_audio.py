import moviepy.editor as mp
import matplotlib.pyplot as plt
import os.path
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

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)

    xs = list(range(len(channel0)))
    ax0.plot(xs, channel0)
    ax1.plot(xs, channel1)

    ax0.set_ylabel('Audio Signal (dB)')
    ax1.set_ylabel('Audio Signal (dB)')
    ax1.set_xlabel('Step')

    file_name = os.path.basename(args.video_path)
    string = file_name.replace('.mp4', '').replace('.MOV', '')
    ax0.set_title('Audio Signal for {}'.format(string))

    plt.tight_layout()

    if args.output_path is None:
        plt.show()
    else:
        plt.savefig(args.output_path, bbox_inches='tight', transparent=True)
