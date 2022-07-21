import numpy as np
from typing import Optional


class MistakeModel:

    def __init__(self, mistake_rate: float):
        self._mistake_rate = mistake_rate

    @property
    def mistake_rate(self) -> float:
        return self._mistake_rate

    def get_mistake_prob(self, move_num: int, num_moves: int, num_mistakes: int) -> float:
        raise NotImplementedError()


class DecayingMistakeModel(MistakeModel):

    def __init__(self, mistake_rate: float, decay_rate: float, suggestion_threshold: Optional[int], suggestion_factor: Optional[float]):
        super().__init__(mistake_rate)
        self._decay_rate = decay_rate
        self._suggestion_threshold = suggestion_threshold
        self._suggestion_factor = suggestion_factor

    @property
    def decay_rate(self) -> float:
        return self._decay_rate

    @property
    def suggestion_threshold(self) -> Optional[int]:
        return self._suggestion_threshold

    @property
    def suggestion_factor(self) -> Optional[float]:
        return self._suggestion_factor

    def get_mistake_prob(self, move_num: int, num_moves: int, num_mistakes: int) -> float:
        offset = max(num_moves - 2, 0) + num_mistakes

        if (self.suggestion_threshold is not None) and (self.suggestion_factor is not None) and (self.suggestion_threshold <= move_num):
            offset /= self.suggestion_factor

        decay_factor = np.power(self.decay_rate, offset)
        return self.mistake_rate * decay_factor
