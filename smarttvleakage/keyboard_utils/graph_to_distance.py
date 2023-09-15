import json
import numpy as np
import csv
import argparse
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from typing import DefaultDict, List, Dict, Tuple

from smarttvleakage.utils.file_utils import read_json, save_json_gz


def combine_dicts(dicts: List[Dict[str, List[str]]]) -> DefaultDict[str, List[str]]:
    if len(dicts) <= 0:
        return dict()

    first_entry = next(iter(dicts[0].values()))

    if isinstance(first_entry, dict):
        combined: Dict[str, Dict[str, str]] = defaultdict(dict)
    else:
        combined: Dict[str, List[str]] = defaultdict(list)
    
    for dictionary in dicts:
        for key, values in dictionary.items():
            if isinstance(values, dict):
                combined[key].update(values)
            else:
                combined[key].extend(values)

    return combined


def to_adjacency_matrix(adjacency_list: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    # Initialize the adjacency matrix
    num_vertices = len(adjacency_list)
    adj_matrix = np.zeros(shape=(num_vertices, num_vertices))

    # Assign unique indices to each key
    sorted_keys = list(sorted(adjacency_list.keys()))
    key_index: Dict[str, int] = dict()
    for idx, key in enumerate(sorted_keys):
        key_index[key] = idx

    for key, neighbors in adjacency_list.items():
        base_idx = key_index[key]

        if isinstance(neighbors, dict):
            neighbor_keys = neighbors.values()
        else:
            neighbor_keys = neighbors

        for neighbor in neighbor_keys:
            neighbor_idx = key_index[neighbor]
            adj_matrix[base_idx, neighbor_idx] = 1

    return adj_matrix, sorted_keys


def serialize_distance_matrix(distance_matrix: np.ndarray, key_lookup: List[str]) -> DefaultDict[str, Dict[str, int]]:
    num_vertices = distance_matrix.shape[0]
    result: DefaultDict[str, Dict[str, int]] = defaultdict(dict)

    for src in range(num_vertices):
        src_key = key_lookup[src]

        for dst in range(num_vertices):
            dst_key = key_lookup[dst]
            dist = distance_matrix[src, dst]

            if (not np.isinf(dist)) and (dist > 0):
                result[src_key][dst_key] = int(dist)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to generate distance matrix in keyboard graphs (for efficiency later on).')
    parser.add_argument('--graph-file', type=str, help='Path to the graph json file.', required=True)
    parser.add_argument('--output-file', type=str, help='Path to the output (.json.gz) file.', required=True)
    args = parser.parse_args()

    assert args.output_file.endswith('.json.gz'), 'Must provide a `.json.gz` output file.'

    # Read in the original graph
    graph_dictionary = read_json(args.graph_file)

    # Configure the possible graph setups
    graph_settings = {
        'normal': graph_dictionary['adjacency_list'],
        'shortcuts': combine_dicts([graph_dictionary['adjacency_list'], graph_dictionary['shortcuts']]),
        'wraparound': combine_dicts([graph_dictionary['adjacency_list'], graph_dictionary['wraparound']]),
        'all': combine_dicts([graph_dictionary['adjacency_list'], graph_dictionary['shortcuts'], graph_dictionary['wraparound']]),
    }

    for setting_name, adjacency_list in graph_settings.items():
        # Create the adjacency matrix
        adj_mat, key_lookup = to_adjacency_matrix(adjacency_list)        
        sp_adj_mat = csr_matrix(adj_mat)

        # Compute the shortest paths
        dist_mat = floyd_warshall(csgraph=sp_adj_mat,
                                  directed=True,
                                  unweighted=True,
                                  return_predecessors=False)

        # Serialize the distance matrix
        serialized_mat = serialize_distance_matrix(dist_mat, key_lookup=key_lookup)

        # Save the result
        output_path = args.output_file.replace('.json.gz', '_{}.json.gz'.format(setting_name))
        save_json_gz(serialized_mat, output_path)
