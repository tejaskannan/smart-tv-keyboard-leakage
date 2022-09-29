from typing import List


def does_conflict(candidate_time: int, comparison_times: List[int], distance: int) -> bool:
    return any(((abs(t - candidate_time) < distance)  for t in comparison_times))


def filter_conflicts(target_times: List[int], comparison_times: List[List[int]], distance: int) -> List[int]:
    filtered_times: List[int] = []

    for target in target_times:
        is_a_conflict = any((does_conflict(target, comparison, distance) for comparison in comparison_times))
        if not is_a_conflict:
            filtered_times.append(target)

    return filtered_times
