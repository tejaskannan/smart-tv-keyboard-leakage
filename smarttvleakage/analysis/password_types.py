from smarttvleakage.analysis.utils import has_special, has_number, has_uppercase
from typing import List, Dict, Callable


class PasswordTypeCounts:

    def __init__(self, prior_names: List[str], top: List[int]):
        self._correct: Dict[str, Dict[str, List[int]]] = dict()
        self._total: Dict[str, Dict[str, int]] = dict()
        self._top = top

        self._types: Dict[str, Callable[[str], bool]] = {
            'special': has_special,
            'numeric': has_number,
            'upper': has_uppercase
        }

        for prior_name in prior_names:
            self._correct[prior_name] = dict()
            self._total[prior_name] = dict()

            for type_name in self._types.keys():
                self._correct[prior_name][type_name] = [0 for _ in top]
                self._total[prior_name][type_name] = 0

    def get_correct(self) -> Dict[str, Dict[str, List[int]]]:
        return self._correct

    def get_total(self) -> Dict[str, Dict[str, int]]:
        return self._total

    def count(self, rank: int, label: str, prior_name: str):
        for type_name, type_fn in self._types.items():
            if type_fn(label):
                self._total[prior_name][type_name] += 1

                for top_idx, top_count in enumerate(self._top):
                    self._correct[prior_name][type_name][top_idx] += int((rank > 0) and (rank <= top_count))
