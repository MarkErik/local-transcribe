#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn
from local_transcribe.framework import registry


def render_video(subs_path: str | Path, output_mp4: str | Path, audio_path: str | Path | list[str | Path], width: int = 1920, height: int = 1080):
    """
    Create a video with a black background and burned-in subtitles + original audio.
    Requires ffmpeg on PATH. Uses SRT input.

    Args:
        subs_path: Path to SRT subtitle file
        output_mp4: Output MP4 file path
        audio_path: Single audio file path or list of two audio paths for dual-track mode
        width: Video width (default 1920)
        height: Video height (default 1080)
    """
    subs_path = Path(subs_path)
    output_mp4 = Path(output_mp4)

    # Handle single audio path or dual audio paths
    if isinstance(audio_path, (str, Path)):
        # Single audio track (combined_audio mode)
        audio_path = Path(audio_path)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=30",
            "-i", str(audio_path),
            "-vf", f"subtitles={subs_path.as_posix()}",
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-shortest",
            str(output_mp4),
        ]
    else:
        # Dual audio tracks (interviewer + participant)
        if len(audio_path) != 2:
            raise ValueError("When providing multiple audio paths, exactly 2 paths are required (interviewer and participant)")

        int_audio = Path(audio_path[0])
        part_audio = Path(audio_path[1])

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=30",
            "-i", str(int_audio),
            "-i", str(part_audio),
            "-filter_complex", "[1:a][2:a]amerge=inputs=2[a]",
            "-vf", f"subtitles={subs_path.as_posix()}",
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-map", "0:v", "-map", "[a]", "-shortest",
            str(output_mp4),
        ]

    subprocess.run(cmd, check=True)


class VideoWriter(OutputWriter):
    """Output writer for MP4 video with burned-in subtitles."""
    
    @property
    def name(self) -> str:
        return "video"
    
    @property
    def description(self) -> str:
        return "MP4 video with burned-in subtitles and original audio"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".mp4"]
    
    def write(self, turns: List[Turn], output_path: str, **kwargs) -> None:
        """Write MP4 video with subtitles.
        
        Args:
            turns: List of conversation turns
            output_path: Output MP4 file path
            **kwargs: Additional arguments including 'audio_path'
        """
        # Convert Turn to dict for compatibility with SRT writer
        turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
        
        # Import SRT writer to generate subtitles first
        from .str_writer import write_srt
        
        # Create temporary SRT file
        srt_path = Path(output_path).with_suffix('.srt')
        write_srt(turn_dicts, srt_path)
        
        try:
            # Get audio path from kwargs or use default
            audio_path = kwargs.get('audio_path')
            if audio_path is None:
                raise ValueError("audio_path is required for video generation")
            
            # Render the video
            render_video(srt_path, output_path, audio_path)
            
        finally:
            # Clean up temporary SRT file
            if srt_path.exists():
                srt_path.unlink()


# Register the video writer with the global registry
registry.register_output_writer(VideoWriter())
