import os.path
from collections import namedtuple
from typing import Dict, List

from smarttvleakage.utils.constants import SmartTVType
from smarttvleakage.utils.file_utils import read_json


KeyboardPosition = namedtuple('KeyboardPosition', ['key', 'mode'])


class KeyboardLinker:

    def __init__(self, tv_type: SmartTVType):
        self._tv_type = tv_type

        dir_name = os.path.dirname(__file__)

        self._link_map: Dict[str, Dict[str, Dict[str, str]]] = dict()

        if tv_type == SmartTVType.APPLE_TV:
            self._link_map = read_json(os.path.join(dir_name, 'apple_tv', 'link.json'))

    def get_linked_states(self, current_key: str, keyboard_mode: str) -> List[KeyboardPosition]:
        if keyboard_mode not in self._link_map:
            return []

        keyboard_link_map = self._link_map[keyboard_mode]
        if current_key not in keyboard_link_map:
            return []

        return [KeyboardPosition(key=key, mode=mode) for mode, key in keyboard_link_map[current_key].items()]


class IdentityLinker(KeyboardLinker):
    
    def get_linked_states(self, current_key: str, keyboard_mode: str) -> List[KeyboardPosition]:
        return []


def make_keyboard_linker(tv_type: SmartTVType) -> KeyboardLinker:
    if tv_type == SmartTVType.SAMSUNG:
        return IdentityLinker(tv_type=tv_type)
    elif tv_type == SmartTVType.APPLE_TV:
        return KeyboardLinker(tv_type=tv_type)
    else:
        raise ValueError('Unknown TV type: {}'.format(tv_type.name))
