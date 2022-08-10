import moviepy.editor as mp
import numpy as np

from smarttvleakage.audio.move_extractor import MoveExtractor, AppleTVMoveExtractor, SamsungMoveExtractor
from smarttvleakage.audio.constants import SAMSUNG_KEY_SELECT, APPLETV_KEYBOARD_SELECT
from smarttvleakage.utils.constants import SmartTVType


class SmartTVTypeClassifier:

    def __init__(self):
        self._apple_tv_extractor = AppleTVMoveExtractor()
        self._samsung_extractor = SamsungMoveExtractor()

    def get_tv_type(self, audio: np.ndarray) -> SmartTVType:
        # Get instances of samsung sounds
        samsung_times, _ = self._samsung_extractor.find_instances_of_sound(audio=audio, sound=SAMSUNG_KEY_SELECT)

        if len(samsung_times) > 0:
            return SmartTVType.SAMSUNG
        else:
            return SmartTVType.APPLE_TV


if __name__ == '__main__':
    video_clip = mp.VideoFileClip('/local/apple-tv/ten/wecrashed.MOV')
    audio = video_clip.audio
    audio_signal = audio.to_soundarray()

    clf = SmartTVTypeClassifier()
    print(clf.get_tv_type(audio=audio_signal))
