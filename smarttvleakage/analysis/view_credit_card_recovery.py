"""
Script to print out the recovery results for credit cards
"""
from argparse import ArgumentParser

from smarttvleakage.analysis.utils import compute_rank
from smarttvleakage.utils.file_utils import read_json


def main(recovery_path: str, labels_path: str):
    recovery_results = read_json(recovery_path)
    labels = read_json(labels_path)['labels']

    found: List[Tuple[str, int]] = []
    not_found: List[str] = []

    for targets, recovery_record in zip(labels, recovery_results):
        # Get the rank for every field
        ccn_rank = compute_rank(guesses=recovery_record['ccn'], label=targets['credit_card'])
        cvv_rank = compute_rank(guesses=recovery_record['cvv'], label=targets['security_code'])
        month_rank = compute_rank(guesses=recovery_record['exp_month'], label=targets['exp_month'])
        year_rank = compute_rank(guesses=recovery_record['exp_year'], label=targets['exp_year'])
        zip_rank = compute_rank(guesses=recovery_record['zip'], label=targets['zip_code'])

        print('Target: CCN -> {}, CVV -> {}, Month -> {}, Year -> {}, ZIP -> {}'.format(targets['credit_card'], targets['security_code'], targets['exp_month'], targets['exp_year'], targets['zip_code']))
        print('Ranks: CCN -> {}, CVV -> {}, Month -> {}, Year -> {}, ZIP -> {}, Overall -> {}'.format(ccn_rank, cvv_rank, month_rank, year_rank, zip_rank, recovery_record['rank']))
        print('======')


if __name__ == '__main__':
    parser = ArgumentParser('Script to display recovery results.')
    parser.add_argument('--recovery-file', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    args = parser.parse_args()

    main(recovery_path=args.recovery_file,
         labels_path=args.labels_file)
