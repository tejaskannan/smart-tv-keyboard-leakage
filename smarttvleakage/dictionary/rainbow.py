from collections import namedtuple, defaultdict
from typing import DefaultDict, List, Tuple

from smarttvleakage.audio import Move
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.utils.transformations import move_seq_to_vector


RainbowEntry = namedtuple('RainbowEntry', ['word', 'score'])


class PasswordRainbow:

    def __init__(self, path: str):
        self._precomputed: DefaultDict[Tuple[int, ...], List[RainbowEntry]] = defaultdict(list)

        for record in read_jsonl_gz(path):
            word = record['target']
            move_seq = record['move_seq']
            score = record['score']

            entry = RainbowEntry(word=word, score=score)
            key = tuple(move_seq)
            self._precomputed[key].append(entry)

    def get_strings_for_seq(self, move_seq: List[Move], tv_type: SmartTVType) -> List[RainbowEntry]:
        move_vector = move_seq_to_vector(move_seq, tv_type=tv_type)
        results = self._precomputed.get(move_vector, [])
        return list(sorted(results, key=lambda t: t.score))
