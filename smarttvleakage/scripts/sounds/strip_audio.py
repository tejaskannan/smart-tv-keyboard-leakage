from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--audio-file', type=str, required=True)
    args = parser.parse_args()

