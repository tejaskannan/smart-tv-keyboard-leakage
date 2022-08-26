from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from .move_extractor import MoveExtractor, SamsungMoveExtractor, AppleTVMoveExtractor, Move
from .tv_classifier import SmartTVTypeClassifier
from .constants import SAMSUNG_DELETE, SAMSUNG_KEY_SELECT, SAMSUNG_SELECT, KEY_SELECT_SOUNDS, APPLETV_TOOLBAR_MOVE
from .constants import APPLETV_KEYBOARD_DELETE, APPLETV_KEYBOARD_SELECT, DONE_SOUNDS, SPACE_SOUNDS, CHANGE_SOUNDS


def make_move_extractor(tv_type: SmartTVType) -> MoveExtractor:
    if tv_type == SmartTVType.SAMSUNG:
        return SamsungMoveExtractor()
    elif tv_type == SmartTVType.APPLE_TV:
        return AppleTVMoveExtractor()
    else:
        raise ValueError('Unknown TV type: {}'.format(tv_type.name))
