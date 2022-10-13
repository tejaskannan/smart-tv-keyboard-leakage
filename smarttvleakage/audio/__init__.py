from smarttvleakage.utils.constants import SmartTVType, KeyboardType
from .move_extractor import MoveExtractor, SamsungMoveExtractor, AppleTVMoveExtractor, Move
from .tv_classifier import SmartTVTypeClassifier
from .data_types import Move


def make_move_extractor(tv_type: SmartTVType) -> MoveExtractor:
    if tv_type == SmartTVType.SAMSUNG:
        return SamsungMoveExtractor()
    elif tv_type == SmartTVType.APPLE_TV:
        return AppleTVMoveExtractor()
    else:
        raise ValueError('Unknown TV type: {}'.format(tv_type.name))
