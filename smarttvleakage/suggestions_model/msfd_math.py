import math


def normalize_msfd_score(msfd_score : float, midpoint : float = .00067) -> float:
    if msfd_score <= 0:
        return 0

    c1 = 1 - math.log(midpoint, 10)
    c2 = math.log(midpoint, 10) + 1
    if msfd_score == midpoint:
        return .5
    elif msfd_score > midpoint:
        x = math.log(msfd_score, 10) + c1
        return 1-(pow(.5, x))
    else:
        x = c2 - math.log(msfd_score, 10)
        return pow(.5, x)


def normalize_db_score(score : float, peak : float = 30):
    if score <= 0:
        return 0
    if score >= peak:
        return 1
    return pow((score/peak), 2)

    


def combine_confidences(ml_score : float, msfd_score : float, db_score, midpoint : float = .00067, peak : float = 30) -> int:
    normalized_msfd_score = normalize_msfd_score(msfd_score, midpoint)
    normalized_db_score = normalize_db_score(db_score, peak)

    normalized_manual_score = max(normalized_msfd_score, normalized_db_score)

    return (ml_score + normalized_manual_score) / 2, normalized_manual_score


if __name__ == "__main__":
    score = normalize_msfd_score(10, 16)
    print(score)
    score = normalize_msfd_score(15, 16)
    print(score)
    score = normalize_msfd_score(20, 16)
    print(score)
    score = normalize_msfd_score(25, 16)
    print(score)

    for s in [.000067, .00067, .0067]:
        score = normalize_msfd_score(s)
        print(score)