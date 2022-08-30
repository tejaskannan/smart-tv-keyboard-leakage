import math


def normalize_manual_score(manual_score : float, midpoint : float = .00067) -> float:
    if manual_score <= 0:
        return 0

    c1 = 1 - math.log(midpoint, 10)
    c2 = math.log(midpoint, 10) + 1
    if manual_score == midpoint:
        return .5
    elif manual_score > midpoint:
        x = math.log(manual_score, 10) + c1
        return 1-(pow(.5, x))
    else:
        x = c2 - math.log(manual_score, 10)
        return pow(.5, x)

def combine_confidences(ml_score : float, manual_score : float, midpoint : float = .00067) -> int:
    normalized_manual_score = normalize_manual_score(manual_score, midpoint)
    return (ml_score + normalized_manual_score) / 2