import os.path
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, DefaultDict, Iterable

from smarttvleakage.utils.credit_card_detection import CreditCardSequence
from smarttvleakage.utils.file_utils import iterate_dir, read_json

TOP_CCN = [1, 5, 10, 50, 100, 500]
TOP_FULL = [1, 10, 100, 500, 1000, 5000, 10000]
GUESSES_NAME = 'recovered_credit_card_details.json'
LABELS_NAME = 'credit_card_details_labels.json'

CCN_MAX_CUTOFF = 200
CCN_CUTOFF = 100
CVV_CUTOFF = 12
ZIP_CUTOFF = 12
MONTH_CUTOFF = 3
YEAR_CUTOFF = 3


def iterate_full_guesses(ccns: List[str], cvvs: List[str], zips: List[str], months: List[str], years: List[str]) -> Iterable[CreditCardSequence]:
    # (1) Output the first guesses based on the top ranked results in each field
    guessed: Set[CreditCardSequence] = set()

    for year_idx in range(min(YEAR_CUTOFF, len(years))):
        for month_idx in range(min(MONTH_CUTOFF, len(months))):
            for zip_idx in range(min(ZIP_CUTOFF, len(zips))):
                for cvv_idx in range(min(CVV_CUTOFF, len(cvvs))):
                    for ccn_idx in range(min(CCN_CUTOFF, len(ccns))):
                        guess = CreditCardSequence(credit_card=ccns[ccn_idx],
                                                   zip_code=zips[zip_idx],
                                                   expiration_month=months[month_idx],
                                                   expiration_year=years[year_idx],
                                                   security_code=cvvs[cvv_idx])

                        if guess not in guessed:
                            yield guess
                            guessed.add(guess)
    

    # (2) Output the entire set by iterating over all of the results
    for year_idx in range(len(years)):
        for month_idx in range(len(months)):
            for zip_idx in range(len(zips)):
                for cvv_idx in range(len(cvvs)):
                    for ccn_idx in range(min(CCN_MAX_CUTOFF, len(ccns))):
                        guess = CreditCardSequence(credit_card=ccns[ccn_idx],
                                                   zip_code=zips[zip_idx],
                                                   expiration_month=months[month_idx],
                                                   expiration_year=years[year_idx],
                                                   security_code=cvvs[cvv_idx])

                        if guess not in guessed:
                            yield guess
                            guessed.add(guess)


def top_k_accuracy(guesses: List[List[str]], targets: List[str], top: int) -> Tuple[int, int]:
    assert top >= 1, 'Must provide a positive `top` count'
    assert len(guesses) == len(targets), 'Must provide the same number of guesses as targets'

    recovered_count = 0

    for entry_guesses, target in zip(guesses, targets):
        for rank, guess in enumerate(entry_guesses):
            if rank >= top:
                break
            elif guess == target:
                recovered_count += 1
                break
    
    return recovered_count, len(targets)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--user-folder', type=str, required=True, help='Name of the folder containing the user results.')
    parser.add_argument('--output-file', type=str, help='Path to (optional) output file in which to save the plot.')
    args = parser.parse_args()

    ccn_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }
    cvv_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }
    zip_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }
    month_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }
    year_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }

    full_correct_counts: DefaultDict[str, List[int]] = defaultdict(list)  # { Subject -> [ counts per rank cutoff ] }

    for subject_folder in iterate_dir(args.user_folder):
        # Read the serialized password guesses
        guesses_path = os.path.join(subject_folder, GUESSES_NAME)
        if not os.path.exists(guesses_path):
            continue

        subject_name = os.path.split(subject_folder)[-1]
        ccn_correct_counts[subject_name] = [0 for _ in range(len(TOP_CCN))]
        cvv_correct_counts[subject_name] = [0 for _ in range(len(TOP_CCN))]
        zip_correct_counts[subject_name] = [0 for _ in range(len(TOP_CCN))]
        month_correct_counts[subject_name] = [0 for _ in range(len(TOP_CCN))]
        year_correct_counts[subject_name] = [0 for _ in range(len(TOP_CCN))]
        full_correct_counts[subject_name] = [0 for _ in range(len(TOP_FULL))]

        # Unpack the guesses
        saved_guesses = read_json(guesses_path)
        ccn_guesses = [entry['ccn'] for entry in saved_guesses]  # List of list of strings
        cvv_guesses = [entry['cvv'] for entry in saved_guesses]
        zip_guesses = [entry['zip'] for entry in saved_guesses]
        month_guesses = [entry['exp_month'] for entry in saved_guesses]
        year_guesses = [entry['exp_year'] for entry in saved_guesses]
        total_ranks = [entry['rank'] for entry in saved_guesses]

        # Read the labels
        labels_path = os.path.join(subject_folder, LABELS_NAME)
        full_labels = [CreditCardSequence(credit_card=r['credit_card'], security_code=r['security_code'], expiration_month=r['exp_month'], expiration_year=r['exp_year'], zip_code=r['zip_code']) for r in read_json(labels_path)['labels']]
        ccn_labels = [entry.credit_card for entry in full_labels]
        cvv_labels = [entry.security_code for entry in full_labels]
        zip_labels = [entry.zip_code for entry in full_labels]
        month_labels = [entry.expiration_month for entry in full_labels]
        year_labels = [entry.expiration_year for entry in full_labels]

        # Get the correct counts for the credit card number alone
        for top_idx, top_count in enumerate(TOP_CCN):
            correct, total = top_k_accuracy(ccn_guesses, targets=ccn_labels, top=top_count)
            ccn_correct_counts[subject_name][top_idx] += correct

            correct, total = top_k_accuracy(cvv_guesses, targets=cvv_labels, top=top_count)
            cvv_correct_counts[subject_name][top_idx] += correct

            correct, total = top_k_accuracy(zip_guesses, targets=zip_labels, top=top_count)
            zip_correct_counts[subject_name][top_idx] += correct

            correct, total = top_k_accuracy(month_guesses, targets=month_labels, top=top_count)
            month_correct_counts[subject_name][top_idx] += correct

            correct, total = top_k_accuracy(year_guesses, targets=year_labels, top=top_count)
            year_correct_counts[subject_name][top_idx] += correct

        for idx in range(len(ccn_guesses)):
            rank = total_ranks[idx]

            #if total_ranks[idx] >= 1:
            #    for rank_idx, full_guess in enumerate(iterate_full_guesses(ccns=ccn_guesses[idx], cvvs=cvv_guesses[idx], zips=zip_guesses[idx], months=month_guesses[idx], years=year_guesses[idx])):
            #        if full_guess == full_labels[idx]:
            #            rank = rank_idx + 1

            for top_idx, topk in enumerate(TOP_FULL):
                full_correct_counts[subject_name][top_idx] += int((rank > 0) and (rank <= topk))

        #full_guesses: List[CreditCardSequence] = []
        print('Subject: {}, Full Correct Counts: {}, Total Ranks: {}'.format(subject_name, full_correct_counts[subject_name], total_ranks))

        #print('Prior: {}, Subject {}, Correct: {}, Total: {}'.format(prior_name, subject_name, correct_counts[prior_name][subject_name], total_counts[prior_name][subject_name]))

    # Compute the accuracy across all cutoffs for each prior
    ccn_correct_list: List[int] = [0 for _ in range(len(TOP_CCN))]
    zip_correct_list: List[int] = [0 for _ in range(len(TOP_CCN))]
    cvv_correct_list: List[int] = [0 for _ in range(len(TOP_CCN))]
    month_correct_list: List[int] = [0 for _ in range(len(TOP_CCN))]
    year_correct_list: List[int] = [0 for _ in range(len(TOP_CCN))]
    full_correct_list: List[int] = [0 for _ in range(len(TOP_FULL))]

    for subject_name in ccn_correct_counts.keys():
        for top_idx in range(len(TOP_CCN)):
            ccn_correct_list[top_idx] += ccn_correct_counts[subject_name][top_idx]
            zip_correct_list[top_idx] += zip_correct_counts[subject_name][top_idx]
            cvv_correct_list[top_idx] += cvv_correct_counts[subject_name][top_idx]
            month_correct_list[top_idx] += month_correct_counts[subject_name][top_idx]
            year_correct_list[top_idx] += year_correct_counts[subject_name][top_idx]

    for subject_name in full_correct_counts.keys():
        for top_idx in range(len(TOP_FULL)):
            full_correct_list[top_idx] += full_correct_counts[subject_name][top_idx]

    total_count = 3 * len(ccn_correct_counts)  # Each user performs 3 credit card entries
    ccn_accuracy_list = [100.0 * (ccn_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_CCN))]
    cvv_accuracy_list = [100.0 * (cvv_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_CCN))]
    zip_accuracy_list = [100.0 * (zip_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_CCN))]
    month_accuracy_list = [100.0 * (month_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_CCN))]
    year_accuracy_list = [100.0 * (year_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_CCN))]
    full_accuracy_list = [100.0 * (full_correct_list[top_idx] / float(total_count)) for top_idx in range(len(TOP_FULL))]

    with plt.style.context('seaborn-ticks'):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))

        ax0.plot(TOP_CCN, ccn_accuracy_list, marker='o', linewidth=3, markersize=8, label='Credit Card Number', color='#fd8d3c')
        #ax0.plot(TOP_CCN, cvv_accuracy_list, marker='o', linewidth=3, markersize=8, label='Security Code')
        #ax0.plot(TOP_CCN, zip_accuracy_list, marker='o', linewidth=3, markersize=8, label='Zip Code')
        #ax0.plot(TOP_CCN, month_accuracy_list, marker='o', linewidth=3, markersize=8, label='Expiration Month')
        #ax0.plot(TOP_CCN, year_accuracy_list, marker='o', linewidth=3, markersize=8, label='Expiration Year')

        # Write the data labels
        for idx, (topk, accuracy) in enumerate(zip(TOP_CCN, ccn_accuracy_list)):
            xoffset = 0 if (topk < 500) else -75.0
            yoffset = -3.0 if (topk > 1) else -2.0
            ax0.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=12)

        ax1.plot(TOP_FULL, full_accuracy_list, marker='o', linewidth=3, markersize=8, label='Full Details', color='#bd0026')

        for idx, (topk, accuracy) in enumerate(zip(TOP_FULL, full_accuracy_list)):
            xoffset = 0 if (topk < 10000) else -1500.0
            yoffset = -3.0 if (topk > 1) else -2.0
            ax1.annotate('{:.2f}%'.format(accuracy), xy=(topk, accuracy), xytext=(topk + xoffset, accuracy + yoffset), size=12)

        #ax.set_xticks(TOP)
        #ax0.set_xscale('log')
        #ax1.set_xscale('log')

        ax0.legend()
        ax1.legend()

        ax0.set_title('Credit Card Number Top-K Accuracy', size=16)
        ax0.set_xlabel('Guess Cutoff (K)', size=14)
        ax0.set_ylabel('Accuracy (%)', size=14)

        ax1.set_title('Credit Card Details Top-K Accuracy', size=16)
        ax1.set_xlabel('Guess Cutoff (K)', size=14)
        ax1.set_ylabel('Accuracy (%)', size=14)

        # Show or save the result
        if args.output_file is None:
            plt.show()
        else:
            plt.savefig(args.output_file, transparent=True, bbox_inches='tight')
