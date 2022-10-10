from typing import List


def does_conflict(candidate_time: int, comparison_times: List[int], forward_distance: int, backward_distance: int) -> bool:
    for t in comparison_times:
        diff = abs(t - candidate_time)

        if (candidate_time > t) and (diff < forward_distance):
            return True
        elif (candidate_time <= t) and (diff < backward_distance):
            return True

    return False
    #return any(((abs(t - candidate_time) < distance)  for t in comparison_times))


def filter_conflicts(target_times: List[int], comparison_times: List[List[int]], forward_distance: int, backward_distance: int) -> List[int]:
    filtered_times: List[int] = []

    for target in target_times:
        is_a_conflict = any((does_conflict(target, comparison, forward_distance, backward_distance) for comparison in comparison_times))
        if not is_a_conflict:
            filtered_times.append(target)

    return filtered_times
