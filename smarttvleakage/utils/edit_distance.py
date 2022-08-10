import numpy as np


def compute_edit_distance(str1: str, str2: str) -> int:
    str1_len = len(str1)
    str2_len = len(str2)

    distance_matrix = np.zeros(shape=(str1_len + 1, str2_len + 1), dtype=int)
    for idx1 in range(str1_len + 1):
        distance_matrix[idx1, 0] = idx1

    for idx2 in range(str2_len + 1):
        distance_matrix[0, idx2] = idx2

    for idx1 in range(str1_len):
        for idx2 in range(str2_len):
            if str1[idx1] == str2[idx2]:
                distance_matrix[idx1 + 1, idx2 + 1] = distance_matrix[idx1, idx2]
            else:
                distance_matrix[idx1 + 1, idx2 + 1] = min(min(distance_matrix[idx1, idx2 + 1], distance_matrix[idx1 + 1, idx2]), distance_matrix[idx1, idx2]) + 1

    return distance_matrix[str1_len, str2_len]
