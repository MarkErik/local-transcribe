from __future__ import annotations
import subprocess
from pathlib import Path

def render_black_video(subs_path: str | Path, output_mp4: str | Path, audio_path: str | Path, width: int = 1920, height: int = 1080):
    """
    Create a black video with burned-in subtitles + original audio.
    Requires ffmpeg on PATH. Uses SRT input.
    """
    subs_path = Path(subs_path)
    output_mp4 = Path(output_mp4)
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
    subprocess.run(cmd, check=True)

