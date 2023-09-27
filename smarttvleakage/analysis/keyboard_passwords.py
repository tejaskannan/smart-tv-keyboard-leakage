import matplotlib
import matplotlib.pyplot as plt
import os.path
from argparse import ArgumentParser
from typing import List, Optional

from smarttvleakage.analysis.utils import PLOT_STYLE, AXIS_SIZE, TITLE_SIZE, LABEL_SIZE, LEGEND_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, MARKER, LINE_WIDTH, COLORS_0, TV_LABELS, FIGSIZE
from smarttvleakage.analysis.utils import compute_rank, top_k_accuracy, compute_accuracy, compute_baseline_accuracy
from smarttvleakage.utils.file_utils import iterate_dir, read_json
from benchmark_password_recovery import GUESSES_NAME, LABELS_NAME, PHPBB_COUNT, TOP


matplotlib.rc('pdf', fonttype=42)  # Embed fonts in pdf
plt.rcParams['pdf.fonttype'] = 42

PRIOR = 'phpbb'


def get_accuracy(folder: str, keyboard_name: str) -> List[float]:
    tv_name = 'samsung' if (keyboard_name == 'abc') else keyboard_name
    base = os.path.join(folder, '{}-passwords'.format(keyboard_name))

    correct_counts = [0 for _ in TOP]
    total_count = 0

    for part_folder in iterate_dir(base):
        # Read the guesses
        guesses_path = os.path.join(part_folder, GUESSES_NAME.format(tv_name, PRIOR))
        if not os.path.exists(guesses_path):
            continue

        saved_password_guesses = read_json(guesses_path)
        password_guesses = [entry['guesses'] for entry in saved_password_guesses]

        # Read the labels
        labels_path = os.path.join(part_folder, LABELS_NAME.format(tv_name))
        password_labels = read_json(labels_path)['labels']

        # Get the correct ranks for each entry
        ranks = [compute_rank(guesses=guesses, label=password_labels[idx]) for idx, guesses in enumerate(password_guesses)]
        total_count += len(password_labels)

        # Get the correct counts for each `top` entry
        for top_idx, top_count in enumerate(TOP):
            correct, _ = top_k_accuracy(ranks, top=top_count)
            correct_counts[top_idx] += correct

    return [100.0 * (correct / total_count) for correct in correct_counts]


def main(benchmark_folder: str, output_file: Optional[str]):
    # Get the top-k accuracy for each keyboard
    samsung_accuracy = get_accuracy(benchmark_folder, keyboard_name='samsung')
    apple_accuracy = get_accuracy(benchmark_folder, keyboard_name='appletv')
    abc_accuracy = get_accuracy(benchmark_folder, keyboard_name='abc')

    base_accuracy = [100.0 * (top / PHPBB_COUNT) for top in TOP]

    # Print out the results in a latex-style table
    print('Keyboard & {} \\\\'.format(' & '.join(map(lambda t: str(t), TOP))))
    
    names = ['Samsung', 'AppleTV', 'ABC', 'Random Guess']
    for name, accuracy in zip(names, [samsung_accuracy, apple_accuracy, abc_accuracy, base_accuracy]):
        fmt = '{:.2f}\%' if name != 'Random Guess' else '{:.5f}\%'

        formatted_accuracy = ' & '.join([fmt.format(acc) for acc in accuracy])
        print('{} & {} \\\\'.format(name, formatted_accuracy))

    #with plt.style.context(PLOT_STYLE):
    #    fig, ax = plt.subplots(figsize=FIGSIZE)

    #    ax.plot(TOP, samsung_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='QWERTY (Samsung)')
    #    ax.plot(TOP, apple_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='ABC (AppleTV)')
    #    ax.plot(TOP, abc_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='ABC (Samsung)')

    #    ax.plot(TOP, base_accuracy, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Random Guess')

    #    ax.legend(fontsize=LEGEND_SIZE)

    #    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark-folder', type=str, required=True, help='Path to the folder containing the results (two levels above the part_* files).')
    parser.add_argument('--output-file', type=str, help='Path to the (optional) output file in which to save the plot.')
    args = parser.parse_args()

    main(args.benchmark_folder, args.output_file)
