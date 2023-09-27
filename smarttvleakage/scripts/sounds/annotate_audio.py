import matplotlib
import matplotlib.pyplot as plt
import scipy.io as io
from argparse import ArgumentParser

from smarttvleakage.analysis.utils import PLOT_STYLE, FIGSIZE, COLORS_0, AXIS_SIZE, TITLE_SIZE
from smarttvleakage.analysis.utils import LABEL_SIZE

matplotlib.rc('pdf', fonttype=42)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    path = '../scripts/test.wav'
    rate, audio = io.wavfile.read(path)

    audio = audio[300000:]
    sample_times = [(sample_count / rate) for sample_count in range(len(audio))]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(sample_times, audio, color=COLORS_0[1])

        # Annotate relevant sounds
        ax.annotate('KeyMovement', (1.94, 1.18e4), (0, 1.8e4), arrowprops=dict(facecolor='k', arrowstyle='->'), fontsize=LABEL_SIZE) 
        ax.annotate('KeySelect', (3.44, 4.2e3), (3.25, 1.8e4), arrowprops=dict(facecolor='k', arrowstyle='->'), fontsize=LABEL_SIZE) 
        ax.annotate('SystemSelect', (8.1, 2.0e4), (8.6, 2.0e4), arrowprops=dict(facecolor='k', arrowstyle='->'), fontsize=LABEL_SIZE) 

        ax.xaxis.set_tick_params(labelsize=LABEL_SIZE)
        ax.yaxis.set_tick_params(labelsize=LABEL_SIZE)
       
        ax.set_title('Samsung Smart TV Audio for Typing \'test\'', fontsize=TITLE_SIZE)
        ax.set_xlabel('Time (s)', fontsize=AXIS_SIZE)
        ax.set_ylabel('Raw 16-bit Audio', fontsize=AXIS_SIZE)

        plt.tight_layout()

        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, bbox_inches='tight', transparent=True)
