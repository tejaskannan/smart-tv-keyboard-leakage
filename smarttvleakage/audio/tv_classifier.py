import numpy as np

import smarttvleakage.audio.sounds as sounds
from smarttvleakage.audio.move_extractor import MoveExtractor, AppleTVMoveExtractor, SamsungMoveExtractor
from smarttvleakage.utils.constants import SmartTVType


class SmartTVTypeClassifier:

    def __init__(self):
        # Make the classifiers for each TV type
        self._appletv_extractor = AppleTVMoveExtractor()
        self._samsung_extractor = SamsungMoveExtractor()

    def get_tv_type(self, target_spectrogram: np.ndarray) -> SmartTVType:
        # Test to see if either TV has any key presses for this spectrogram
        num_samsung = self._samsung_extractor.num_sound_instances(target_spectrogram=target_spectrogram,
                                                                  target_sound=sounds.SAMSUNG_KEY_SELECT)

        num_appletv = self._appletv_extractor.num_sound_instances(target_spectrogram=target_spectrogram,
                                                                  target_sound=sounds.APPLETV_KEYBOARD_SELECT)

        if (num_samsung > num_appletv):
            return SmartTVType.SAMSUNG
        elif (num_appletv > num_samsung):
            return SmartTVType.APPLE_TV
        else:
            return SmartTVType.UNKNOWN
