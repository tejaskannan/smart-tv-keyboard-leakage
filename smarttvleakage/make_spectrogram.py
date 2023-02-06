import os.path
from argparse import ArgumentParser

from smarttvleakage.audio import create_spectrogram, SmartTVAudio
from smarttvleakage.utils.file_utils import save_pickle_gz, iterate_dir


if __name__ == '__main__':
    parser = ArgumentParser('Script to create a spectrogram from the audio of a given video file.')
    parser.add_argument('--video-path', type=str, required=True, help='Path to the video file or folder.')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the output folder. The output file will be a compressed pickle with the same name as the input video.')
    parser.add_argument('--should-print', action='store_true', help='Whether to print progress information to the stdout.')
    args = parser.parse_args()

    # Extract the file or directory
    if not os.path.isdir(args.video_path):
        video_paths = [args.video_path]
    else:
        video_paths = list(iterate_dir(args.video_path))

    video_paths = [path for path in video_paths if (path.endswith('.mp4') or path.endswith('.MOV') or path.endswith('.mov'))]

    for idx, path in enumerate(video_paths):
        # Extract the audio from the Smart TV video recording
        audio_extractor = SmartTVAudio(path)

        # Create the spectrogram
        recording_spectrogram = create_spectrogram(audio_extractor.get_audio())

        # Save the result
        output_path = os.path.join(args.output_folder, '{}.pkl.gz'.format(audio_extractor.file_name))
        save_pickle_gz(recording_spectrogram, output_path)

        print('Completed {} / {} files.'.format(idx + 1, len(video_paths)), end='\r')
    print()
