import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from typing import List, Optional, Dict, Any

from smarttvleakage.audio import Move
from smarttvleakage.dictionary.dictionaries import restore_dictionary, ZipCodeDictionary
from smarttvleakage.graphs.keyboard_graph import MultiKeyboardGraph
from smarttvleakage.utils.file_utils import read_json_gz, save_jsonl_gz, make_dir
from smarttvleakage.utils.constants import KeyboardType, BIG_NUMBER, SmartTVType
from smarttvleakage.utils.credit_card_detection import extract_credit_card_sequence, CreditCardSequence
from smarttvleakage.credit_card_recovery import get_correct_rank


CreditCardResult = namedtuple('CreditCardResult', ['ccn_rank', 'cvv_rank', 'zip_code_rank', 'month_rank', 'year_rank', 'total_rank'])
MAX_NUM_RESULTS = 5


def compute_recovery_rank(ccn_rank: Optional[int], month_rank: Optional[int], year_rank: Optional[int], cvv_rank: Optional[int], zip_code_rank: Optional[int]) -> Optional[int]:
    if (ccn_rank is None) or (month_rank is None) or (year_rank is None) or (cvv_rank is None) or (zip_code_rank is None):
        return None

    rank_sum = ccn_rank - 1
    rank_sum += (cvv_rank - 1) * MAX_NUM_RESULTS
    rank_sum += (zip_code_rank - 1) * MAX_NUM_RESULTS**2
    rank_sum += (month_rank - 1) * MAX_NUM_RESULTS**3
    rank_sum += (year_rank - 1) * MAX_NUM_RESULTS**4

    return rank_sum + 1


def recover_credit_card_info(credit_card_seq: CreditCardSequence, credit_card_record: Dict[str, str], graph: MultiKeyboardGraph, zip_code_dictionary: ZipCodeDictionary, tv_type: SmartTVType) -> CreditCardResult:
    # Create the relevant dictionaries
    ccn_dictionary = restore_dictionary('credit_card')
    numeric_dictionary = restore_dictionary('numeric')
    exp_year_dictionary = restore_dictionary('exp_year')

    # Find the ranks for each individual field
    ccn_rank = get_correct_rank(move_seq=credit_card_seq.credit_card,
                                graph=graph,
                                dictionary=ccn_dictionary,
                                tv_type=tv_type,
                                max_num_guesses=MAX_NUM_RESULTS,
                                target=credit_card_record['credit_card_number'])

    month_rank = get_correct_rank(move_seq=credit_card_seq.expiration_month,
                                  graph=graph,
                                  dictionary=numeric_dictionary,
                                  tv_type=tv_type,
                                  max_num_guesses=MAX_NUM_RESULTS,
                                  target=credit_card_record['exp_month'])

    year_rank = get_correct_rank(move_seq=credit_card_seq.expiration_year,
                                 graph=graph,
                                 dictionary=exp_year_dictionary,
                                 tv_type=tv_type,
                                 max_num_guesses=MAX_NUM_RESULTS,
                                 target=credit_card_record['exp_year'])

    cvv_rank = get_correct_rank(move_seq=credit_card_seq.security_code,
                                graph=graph,
                                dictionary=numeric_dictionary,
                                tv_type=tv_type,
                                max_num_guesses=MAX_NUM_RESULTS,
                                target=credit_card_record['cvv'])

    zip_code_rank = get_correct_rank(move_seq=credit_card_seq.zip_code,
                                     graph=graph,
                                     dictionary=zip_code_dictionary,
                                     tv_type=tv_type,
                                     max_num_guesses=MAX_NUM_RESULTS,
                                     target=credit_card_record['zip_code'])

    # Compute the unified ranking (on all fields)
    total_rank = compute_recovery_rank(ccn_rank=ccn_rank,
                                       month_rank=month_rank,
                                       year_rank=year_rank,
                                       cvv_rank=cvv_rank,
                                       zip_code_rank=zip_code_rank)

    return CreditCardResult(ccn_rank=ccn_rank,
                            month_rank=month_rank,
                            year_rank=year_rank,
                            cvv_rank=cvv_rank,
                            zip_code_rank=zip_code_rank,
                            total_rank=total_rank)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--benchmark-path', type=str, required=True)
    parser.add_argument('--zip-code-dict-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Read in the credit card information
    credit_card_info_list = read_json_gz(args.benchmark_path)

    # Make the graph and build the dictionaries (one for each field)
    tv_type = SmartTVType.SAMSUNG
    keyboard_type = KeyboardType.SAMSUNG
    graph = MultiKeyboardGraph(keyboard_type=keyboard_type)

    #ccn_dictionary = restore_dictionary('credit_card')
    #numeric_dictionary = restore_dictionary('numeric')
    #exp_year_dictionary = restore_dictionary('exp_year')
    zip_code_dictionary = restore_dictionary(args.zip_code_dict_path)

    not_found: List[str] = []
    ranks: List[str] = []
    num_found = 0
    top1_found = 0
    top10_found = 0
    top50_found = 0
    total_count = 0
    results: List[Dict[str, Any]] = []

    for record_idx, credit_card_record in enumerate(credit_card_info_list):
        if (args.max_num_samples is not None) and (record_idx >= args.max_num_samples):
            break

        # Get the move sequence
        move_seq = list(map(lambda d: Move.from_dict(d), credit_card_record['move_seq']))

        # Break up the move sequence into the appropriate fields (done by matching lengths)
        credit_card_seq = extract_credit_card_sequence(move_seq)

        if credit_card_seq is None:
            print('WARNING: Could not extract credit card seq for {}'.format(credit_card_record))
            continue

        cc_result = recover_credit_card_info(credit_card_seq=credit_card_seq,
                                             credit_card_record=credit_card_record,
                                             zip_code_dictionary=zip_code_dictionary,
                                             graph=graph,
                                             tv_type=tv_type)

        # Create the result dictionary
        result_dict = {
            'total_rank': cc_result.total_rank,
            'ccn_rank': cc_result.ccn_rank,
            'month_rank': cc_result.month_rank,
            'year_rank': cc_result.year_rank,
            'cvv_rank': cc_result.cvv_rank,
            'zip_code_rank': cc_result.zip_code_rank,
            'target_record': credit_card_record
        }
        results.append(result_dict)

        total_rank = cc_result.total_rank

        if total_rank is not None:
            top1_found += int(total_rank == 1)
            top10_found += int(total_rank <= 10)
            top50_found += int(total_rank <= 50)
            num_found += 1
            ranks.append(total_rank)

        total_count += 1

        if (record_idx + 1) % 10 == 0:
            print('Completed {} records. Top 10 Accuracy So Far: {:.5f}'.format(record_idx + 1, top10_found / total_count), end='\r')

    print()
    print('Top 1 Accuracy: {:.4f}'.format(top1_found / total_count))
    print('Top 10 Accuracy: {:.4f}'.format(top10_found / total_count))
    print('Top 50 Accuracy: {:.4f}'.format(top50_found / total_count))
    print('Avg Rank: {:.4f}, Med Rank: {:.4f}'.format(np.average(ranks), np.median(ranks)))
    #print('Not Found: {}'.format(not_found))

    # Save the results
    save_jsonl_gz(results, args.output_path)
