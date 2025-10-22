from __future__ import annotations
import subprocess
from pathlib import Path

def render_black_video(subs_path: str | Path, output_mp4: str | Path, audio_path: str | Path | list[str | Path], width: int = 1920, height: int = 1080):
    """
    Create a black video with burned-in subtitles + original audio.
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
        # Single audio track (combined mode)
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

