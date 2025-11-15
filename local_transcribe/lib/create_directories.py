#!/usr/bin/env python3
from __future__ import annotations
import pathlib
from datetime import datetime

def ensure_session_dirs(output_dir: str | pathlib.Path, mode: str, speaker_files: dict = None, verbose: bool = False) -> dict[str, pathlib.Path]:
    """
    Creates a consistent directory structure for outputs and returns paths.
    Mode can be 'combined_audio' or 'split_audio'.
    For split_audio, speaker_files dict maps speaker names to file paths.
    If verbose, creates Intermediate_Outputs subdirectories.
    """
    root = pathlib.Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root,
    }

    if mode == "split_audio" and speaker_files:
        for speaker_name in speaker_files.keys():
            speaker_dir = root / f"speaker_{speaker_name.lower()}"
            speaker_dir.mkdir(parents=True, exist_ok=True)
            paths[f"speaker_{speaker_name.lower()}"] = speaker_dir

    if verbose:
        intermediate_dir = root / "Intermediate_Outputs"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        (intermediate_dir / "transcription").mkdir(exist_ok=True)
        (intermediate_dir / "alignment").mkdir(exist_ok=True)
        (intermediate_dir / "transcription_alignment").mkdir(exist_ok=True)
        (intermediate_dir / "diarization").mkdir(exist_ok=True)
        (intermediate_dir / "turns").mkdir(exist_ok=True)
        paths["intermediate"] = intermediate_dir

    return paths
