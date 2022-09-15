import moviepy.editor as mp
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
    save_pickle_gz(signal, args.output_path)
