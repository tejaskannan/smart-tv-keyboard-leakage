import subprocess as sp
import os
import sys
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser('Driver script to log signals and capture audio simulateously.')
    parser.add_argument('--audio-path', type=str, required=True, help='Path to the output audio file.')
    parser.add_argument('--infrared-path', type=str, required=True, help='Path to the logged remote commands via infrared.')
    parser.add_argument('--duration', type=int, required=True, help='The duration of recording in seconds.')
    args = parser.parse_args()

    if os.path.exists(args.infrared_path):
        print('The file {} already exists. Do you want to overwrite it? [y/n]'.format(args.infrared_path), end=' ')
        user_decision = input()

        if user_decision.lower() not in ('y', 'yes'):
            print('Quitting.')
            sys.exit(0)
        
        os.remove(args.infrared_path)

    # Launch the audio process
    audio_process = sp.Popen(['arecord', '-D', 'plughw:1,0', '--duration', str(args.duration), args.audio_path], stdout=sp.PIPE, stderr=sp.PIPE)

    # Launch the infrared capture process
    infrared_cmd = 'mode2 -d /dev/lirc0 | ./detect_remote.py --output-path {}'.format(args.infrared_path)
    print(infrared_cmd)
    infrared_process = sp.Popen(infrared_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

    # Wait for both processes to finish. The infrared process will technically run forever so we just kill it.
    audio_process.wait()
    infrared_process.kill()
