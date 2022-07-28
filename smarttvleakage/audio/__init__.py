from smarttvleakage.utils.constants import SmartTVType
from .move_extractor import MoveExtractor, SamsungMoveExtractor, AppleTvMoveExtractor
from .constants import SAMSUNG_DELETE, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT
from .constants import APPLETV_KEYBOARD_DELETE, APPLETV_KEYBOARD_SELECT


def make_move_extractor(tv_type: SmartTVType) -> MoveExtractor:
    if tv_type == SmartTVType.SAMSUNG:
        return SamsungMoveExtractor()
    elif tv_type == SmartTVType.APPLE_TV:
        return AppleTvMoveExtractor()
    else:
        raise ValueError('Unknown TV type: {}'.format(tv_type.name))
