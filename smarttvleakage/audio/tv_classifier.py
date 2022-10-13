import moviepy.editor as mp
import numpy as np

from smarttvleakage.audio.move_extractor import MoveExtractor, AppleTVMoveExtractor, SamsungMoveExtractor
from smarttvleakage.audio.sounds import SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.utils.constants import SmartTVType


SAMSUNG_MIN_PEAK_HEIGHT = 1.1
APPLETV_MIN_PEAK_HEIGHT = 0.4



class SmartTVTypeClassifier:

    def __init__(self):
        self._apple_tv_extractor = AppleTVMoveExtractor()
        self._samsung_extractor = SamsungMoveExtractor()

    def get_tv_type(self, audio: np.ndarray) -> SmartTVType:
        # Get instances of samsung sounds
        _, samsung_heights = self._samsung_extractor.find_instances_of_sound(audio=audio, sound=SAMSUNG_KEY_SELECT)
        _, appletv_heights = self._apple_tv_extractor.find_instances_of_sound(audio=audio, sound=APPLETV_KEYBOARD_SELECT)

        if min(samsung_heights) > SAMSUNG_MIN_PEAK_HEIGHT:
            return SmartTVType.SAMSUNG
        elif min(appletv_heights) > APPLETV_MIN_PEAK_HEIGHT:
            return SmartTVType.APPLE_TV
        else:
            return SmartTVType.UNKNOWN


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/smart-tv-gettysburg/year.MOV')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    clf = SmartTVTypeClassifier()
    print(clf.get_tv_type(audio=audio_signal))
