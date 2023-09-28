import os.path
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from enum import Enum, auto
from typing import List, Tuple, Dict, DefaultDict, Iterable

from smarttvleakage.utils.credit_card_detection import CreditCardSequence
from smarttvleakage.utils.file_utils import iterate_dir, read_json
from smarttvleakage.analysis.utils import PLOT_STYLE, compute_rank, top_k_accuracy, AXIS_SIZE, TITLE_SIZE, LEGEND_SIZE, LABEL_SIZE
from smarttvleakage.analysis.utils import MARKER_SIZE, LINE_WIDTH, MARKER, TV_COLORS, CREDIT_CARD_FIGSIZE, print_as_table


matplotlib.rc('pdf', fonttype=42)  # Embed fonts in pdf
plt.rcParams['pdf.fonttype'] = 42


TOP_CCN = [1, 5, 10, 50, 100, 250]
TOP_FULL = [1, 10, 100, 1000, 2500, 5000]
GUESSES_NAME = 'recovered_credit_card_details.json'
LABELS_NAME = 'credit_card_details_labels.json'

CCN_MAX_CUTOFF = 200
CCN_CUTOFF = 100
CVV_CUTOFF = 12
ZIP_CUTOFF = 12
MONTH_CUTOFF = 3
YEAR_CUTOFF = 3


class CreditCardProvider(Enum):
    VISA = auto()
    MASTERCARD = auto()
    AMEX = auto()

    def __lt__(self, other):
        if isinstance(other, CreditCardProvider):
            return self.name < other.name
        else:
            return False


def determine_provider(ccn: str) -> CreditCardProvider:
    if ccn.startswith('2') or ccn.startswith('5'):
        return CreditCardProvider.MASTERCARD
    elif ccn.startswith('4'):
        return CreditCardProvider.VISA
    elif ccn.startswith('3'):
        return CreditCardProvider.AMEX
    else:
        raise ValueError('Unknown credit card prefix: {}'.format(ccn[0]))


def get_ccn_offsets(top: int) -> Tuple[float, float]:
    if top == 250:
        return (-50.0, -7.0)
    elif top == 50:
        return (5.0, -7.0)
    elif top > 1:
        return (5.0, -5.0)
    else:
        return (5.0, 2.0)


def get_full_offsets(top: int) -> Tuple[float, float]:
    if top == 5000:
        return (-1000.0, -6.0)
    else:
        xoffset = 200.0

        if top == 2500:
            return (xoffset, -6.0)
        elif top == 1:
            return (xoffset, 0.0)
        else:
            return (xoffset, -4.0)


if __name__ == '__main__':
    parser = ArgumentParser('Script to display attack results against user credit card details.')
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    ccn_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }
    full_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }

    provider_ccn_correct_counts: Dict[CreditCardProvider, List[int]] = dict()  # { Provider -> [ counts per rank cutoff ] }
    provider_full_correct_counts: Dict[CreditCardProvider, List[int]] = dict()  # { Provider -> [ counts per rank cutoff ] }
    provider_total_counts: Dict[CreditCardProvider, int] = dict()  # { Provider -> total_count }

    for subject_folder in iterate_dir(args.user_folder):
        # Read the serialized password guesses
        guesses_path = os.path.join(subject_folder, GUESSES_NAME)
        if not os.path.exists(guesses_path):
            continue

        subject_name = os.path.split(subject_folder)[-1]
        ccn_correct_counts[subject_name] = [0 for _ in range(len(TOP_CCN))]
        full_correct_counts[subject_name] = [0 for _ in range(len(TOP_FULL))]

        # Unpack the guesses
        saved_guesses = read_json(guesses_path)
        ccn_guesses = [entry['ccn'] for entry in saved_guesses]  # List of list of strings
        total_ranks = [entry['rank'] for entry in saved_guesses]

        # Read the labels
        labels_path = os.path.join(subject_folder, LABELS_NAME)
        labels = read_json(labels_path)['labels']
        ccn_labels = [entry['credit_card'] for entry in labels]

        # Get the Credit Card Number rank
        ccn_ranks = [compute_rank(guesses, ccn_labels[idx]) for idx, guesses in enumerate(ccn_guesses)]

        # Get the credit card provider
        providers = [determine_provider(ccn) for ccn in ccn_labels]

        for top_idx, top_count in enumerate(TOP_CCN):
            correct, _ = top_k_accuracy(ccn_ranks, top=top_count)
            ccn_correct_counts[subject_name][top_idx] += correct

        for top_idx, top_count in enumerate(TOP_FULL):
            correct, _ = top_k_accuracy(total_ranks, top=top_count)
            full_correct_counts[subject_name][top_idx] += correct

        # Aggregate the results by provider
        for provider in providers:
            if provider not in provider_total_counts:
                provider_total_counts[provider] = 0
            provider_total_counts[provider] += 1

        for idx, (ccn_rank, full_rank) in enumerate(zip(ccn_ranks, total_ranks)):
            provider = providers[idx]

            if provider not in provider_ccn_correct_counts:
                provider_ccn_correct_counts[provider] = [0 for _ in range(len(TOP_CCN))]
                provider_full_correct_counts[provider] = [0 for _ in range(len(TOP_FULL))]

            for top_idx, top_count in enumerate(TOP_CCN):
                provider_ccn_correct_counts[provider][top_idx] += int((ccn_rank >= 1) and (ccn_rank <= top_count))

            for top_idx, top_count in enumerate(TOP_FULL):
                provider_full_correct_counts[provider][top_idx] += int((full_rank >= 1) and (full_rank <= top_count))

    # Compute the accuracy across all cutoffs for each prior
    ccn_correct_list: List[int] = [0 for _ in range(len(TOP_CCN))]
    full_correct_list: List[int] = [0 for _ in range(len(TOP_FULL))]

    for subject_name in ccn_correct_counts.keys():
        for top_idx in range(len(TOP_CCN)):
            ccn_correct_list[top_idx] += ccn_correct_counts[subject_name][top_idx]

    for subject_name in full_correct_counts.keys():
        for top_idx in range(len(TOP_FULL)):
            full_correct_list[top_idx] += full_correct_counts[subject_name][top_idx]

    total_count = 3 * len(ccn_correct_counts)  # Each user performs 3 credit card entries
    ccn_accuracy_list = [100.0 * (ccn_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_CCN))]
    full_accuracy_list = [100.0 * (full_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_FULL))]

    # Compute the accuracy across all cutoffs for each provider
    print('Credit Card Number')
    print_as_table(provider_ccn_correct_counts, provider_total_counts, TOP_CCN)
    print('==========')
    print('Full Details')
    print_as_table(provider_full_correct_counts, provider_total_counts, TOP_FULL)

    with plt.style.context(PLOT_STYLE):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=CREDIT_CARD_FIGSIZE)

        ax0.plot(TOP_CCN, ccn_accuracy_list, marker=MARKER, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Credit Card Number', color=TV_COLORS['samsung'][0])

        # Write the data labels
        for topk, accuracy in zip(TOP_CCN, ccn_accuracy_list):
            xoffset, yoffset = get_ccn_offsets(top=topk)
            ax0.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=AXIS_SIZE)

        ax1.plot(TOP_FULL, full_accuracy_list, marker='o', linewidth=3, markersize=8, label='Full Details', color=TV_COLORS['samsung'][0])

        for topk, accuracy in zip(TOP_FULL, full_accuracy_list):
            xoffset, yoffset = get_full_offsets(top=topk)
            ax1.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=AXIS_SIZE)

        ax0.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax0.yaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax1.xaxis.set_tick_params(labelsize=TITLE_SIZE)
        ax1.yaxis.set_tick_params(labelsize=TITLE_SIZE)

        ax0.set_title('CCN Accuracy on Users', fontsize=TITLE_SIZE)
        ax0.set_xlabel('Guess Cutoff', fontsize=TITLE_SIZE)
        ax0.set_ylabel('Accuracy (%)', fontsize=TITLE_SIZE)

        ax1.set_title('Full Accuracy on Users', fontsize=TITLE_SIZE)
        ax1.set_xlabel('Guess Cutoff', fontsize=TITLE_SIZE)
        ax1.set_ylabel('Accuracy (%)', fontsize=TITLE_SIZE)

        plt.tight_layout()

        # Show or save the result
        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
