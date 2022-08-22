import sqlite3
from collections import namedtuple, defaultdict
from typing import DefaultDict, List, Optional

from smarttvleakage.audio import Move
from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_jsonl_gz
from smarttvleakage.utils.transformations import move_seq_to_vector


RainbowEntry = namedtuple('RainbowEntry', ['word', 'score'])
QUERY_WITHOUT_LIMIT = 'SELECT password, score FROM passwords WHERE seq=:seq'
QUERY_WITH_LIMIT = 'SELECT password, score FROM passwords WHERE seq=:seq ORDER BY score ASC LIMIT :limit'


class PasswordRainbow:

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._cursor = self._conn.cursor()

    def get_strings_for_seq(self, move_seq: List[Move], tv_type: SmartTVType, max_num_results: Optional[int]) -> List[RainbowEntry]:
        move_vector = move_seq_to_vector(move_seq, tv_type=tv_type)

        if max_num_results is None:
            query = self._cursor.execute(QUERY_WITHOUT_LIMIT, {'seq': move_vector})
        else:
            query = self._cursor.execute(QUERY_WITH_LIMIT, {'seq': move_vector, 'limit': max_num_results})

        query_result = query.fetchall()
        return list(sorted(map(lambda t: RainbowEntry(word=t[0], score=t[1]), query_result), key=lambda t: t.score))
