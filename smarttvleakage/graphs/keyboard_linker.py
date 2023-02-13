import os.path
from collections import namedtuple
from typing import Dict, List, Optional

from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_json


KeyboardPosition = namedtuple('KeyboardPosition', ['key', 'mode'])


class KeyboardLinker:

    def __init__(self, path: Optional[str]):
        self._link_map: Dict[str, Dict[str, Dict[str, str]]] = dict()

        if path is not None:
            self._link_map = read_json(path)

    def get_linked_states(self, current_key: str, keyboard_mode: str) -> List[KeyboardPosition]:
        if keyboard_mode not in self._link_map:
            return []

        keyboard_link_map = self._link_map[keyboard_mode]
        if current_key not in keyboard_link_map:
            return []

        return [KeyboardPosition(key=key, mode=mode) for mode, key in keyboard_link_map[current_key].items()]

    def get_linked_key_to(self, current_key: str, current_mode: str, target_mode: str) -> Optional[str]:
        # Get the linked states for the current keyboard
        keyboard_link_map = self._link_map[current_mode]

        if current_key not in keyboard_link_map:
            return None

        return keyboard_link_map[current_key].get(target_mode)
