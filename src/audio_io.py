from __future__ import annotations
import pathlib
import tempfile
import subprocess
from typing import Optional

# We use ffmpeg via command line to normalize: mono/16k WAV
# This avoids subtle differences between python audio stacks and keeps it robust.

def _ffmpeg_to_wav16k_mono(src: str | pathlib.Path, dst: str | pathlib.Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def standardize_and_get_path(src: str | pathlib.Path, tmpdir: Optional[str | pathlib.Path] = None) -> pathlib.Path:
    """
    Convert any supported audio to 16kHz mono WAV and return its path.
    Writes to a stable temporary path alongside source or in `tmpdir`.
    """
    src = pathlib.Path(src).resolve()
    assert src.exists(), f"File not found: {src}"
    out_dir = pathlib.Path(tmpdir) if tmpdir else src.parent
    out = out_dir / (src.stem + ".wav")
    _ffmpeg_to_wav16k_mono(src, out)
    return out

