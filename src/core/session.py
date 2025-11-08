#!/usr/bin/env python3
from __future__ import annotations
import pathlib
from datetime import datetime

def ensure_session_dirs(output_dir: str | pathlib.Path, mode: str) -> dict[str, pathlib.Path]:
    """
    Creates a consistent directory structure for outputs and returns paths.
    Mode can be 'combined' or 'dual_track'.
    """
    root = pathlib.Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    merged = root / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root,
        "merged": merged,
    }

    if mode == "dual_track":
        speaker_interviewer = root / "speaker_interviewer"
        speaker_participant = root / "speaker_participant"
        for d in (speaker_interviewer, speaker_participant):
            d.mkdir(parents=True, exist_ok=True)
        paths["speaker_interviewer"] = speaker_interviewer
        paths["speaker_participant"] = speaker_participant

    return paths
