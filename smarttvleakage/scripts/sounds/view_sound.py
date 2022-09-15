import moviepy.editor as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from smarttvleakage.utils.file_utils import save_pickle_gz


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--output-path', type=str)
    args = parser.parse_args()

    video_clip = mp.VideoFileClip(args.video_path)
    audio = video_clip.audio

    signal = audio.to_soundarray()
    channel0 = signal[:, 0]

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        times = list(range(len(channel0)))
        ax.plot(times, channel0)

        ax.set_title('Audio Signal from Smart TV Typing')

        if args.output_path is not None:
            plt.savefig(args.output_path, transparent=True, bbox_inches='tight')
        else:
            plt.show()

