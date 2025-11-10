#!/usr/bin/env python3
"""
Plugin for video rendering with burned-in subtitles.
"""

from typing import Union, List
from pathlib import Path
from core.plugins import registry
from output_writers.render_black import render_black_video as _render_black_video


class VideoRenderer:
    """Provider for rendering video with burned-in subtitles."""

    @property
    def name(self) -> str:
        return "black-video"

    @property
    def description(self) -> str:
        return "Render black video with burned-in subtitles"

    def render_video(
        self,
        srt_path: Union[str, Path],
        output_path: Union[str, Path],
        audio_path: Union[str, Path, List[Union[str, Path]]],
        **kwargs
    ) -> None:
        """
        Render video with subtitles burned in.

        Args:
            srt_path: Path to SRT subtitle file
            output_path: Path where to save the rendered video
            audio_path: Path(s) to audio file(s) to use as video source
            **kwargs: Additional rendering options
        """
        _render_black_video(srt_path, output_path, audio_path=audio_path)


# Register the video renderer
video_renderer = VideoRenderer()
# Note: We'll add this to the registry when we integrate it into the main system