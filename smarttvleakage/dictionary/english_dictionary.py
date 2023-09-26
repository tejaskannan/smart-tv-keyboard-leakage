import sqlite3
from typing import Dict

from smarttvleakage.utils.constants import START_CHAR, END_CHAR
from smarttvleakage.dictionary.dictionaries import CHARACTER_TRANSLATION, REVERSE_CHARACTER_TRANSLATION


class SQLEnglishDictionary:

    def __init__(self, db_file: str):
        self._db_file = db_file
        self._conn = sqlite3.connect(db_file)

        self._cache: Dict[str, Dict[str, int]] = dict()
        self._query = 'SELECT next, count FROM prefixes WHERE prefix=:prefix'

    def get_letter_counts(self, prefix: str) -> Dict[str, int]:
        if len(prefix) == 0:
            prefix = START_CHAR

        # Use the cache if we already have the result
        if prefix in self._cache:
            return self._cache[prefix]

        # Look up the prefix in the SQL database         
        cursor = self._conn.cursor()
        query_exec = cursor.execute(self._query, {'prefix': prefix})
        query_results = query_exec.fetchall()
        character_counts = { char: count for (char, count) in query_results }

        # Convert any characters
        character_counts = {REVERSE_CHARACTER_TRANSLATION.get(char, char): count for char, count in character_counts.items()}

        return character_counts
