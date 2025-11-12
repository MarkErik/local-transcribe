#!/usr/bin/env python3
from __future__ import annotations
import pathlib
from datetime import datetime

def ensure_session_dirs(output_dir: str | pathlib.Path, mode: str, speaker_files: dict = None) -> dict[str, pathlib.Path]:
    """
    Creates a consistent directory structure for outputs and returns paths.
    Mode can be 'single_file' or 'separate_audio'.
    For separate_audio, speaker_files dict maps speaker names to file paths.
    """
    root = pathlib.Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    merged = root / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root,
        "merged": merged,
    }

    if mode == "separate_audio" and speaker_files:
        for speaker_name in speaker_files.keys():
            speaker_dir = root / f"speaker_{speaker_name.lower()}"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            paths[f"speaker_{speaker_name.lower()}"] = speaker_dir

    return paths
