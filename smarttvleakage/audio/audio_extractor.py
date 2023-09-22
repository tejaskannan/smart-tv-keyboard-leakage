"""
This class extracts the audio from a video recording of a smart TV. The action is simple enough, as it relies
just on the moviepy library. Instead, we use a separate class to prevent bugs. Our attack relies only
on audio, so we `hide` the video in this class.
"""
import numpy as np
import os.path
from moviepy.editor import VideoFileClip


class SmartTVAudio:

    def __init__(self, path: str):
        self._path = path
        self._file_name = os.path.basename(path).replace('.mp4', '').replace('.MOV', '').replace('.mov', '')

        # Extract the audio from the video. We only use channel 0
        video_clip = VideoFileClip(path)
        self._audio = video_clip.audio.to_soundarray()[:, 0]  # [N]

    @property
    def path(self) -> str:
        return self._path

    @property
    def file_name(self) -> str:
        return self._file_name

    def get_audio(self) -> np.ndarray:
        return self._audio
