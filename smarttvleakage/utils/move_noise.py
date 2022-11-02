import numpy as np
from typing import List

from smarttvleakage.audio.data_types import Move
from smarttvleakage.utils.constants import SMALL_NUMBER


def add_move_noise(move_sequence: List[Move], scale: int, rate: float) -> List[Move]:
    assert scale >= 0, 'Must provide a non-negative scale'
    assert (rate >= 0.0) and (rate <= 1.0), 'The rate must be in the range [0, 1]'

    if (abs(rate) < SMALL_NUMBER) or (scale == 0):
        return move_sequence

    result: List[Move] = []
    for move in move_sequence:
        noise = int(np.random.randint(low=1, high=(scale + 1)))
        num_moves = max(move.num_moves - int(np.random.uniform() < rate) * noise, 0)
        updated_move = Move(num_moves=num_moves,
                            end_sound=move.end_sound,
                            directions=move.directions,
                            end_time=move.end_time)

        result.append(updated_move)

    return result
