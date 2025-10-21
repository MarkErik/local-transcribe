from __future__ import annotations
import pathlib
from datetime import datetime

def ensure_session_dirs(output_dir: str | pathlib.Path) -> dict[str, pathlib.Path]:
    """
    Creates a consistent directory structure for outputs and returns paths.
    """
    root = pathlib.Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    speaker_interviewer = root / "speaker_interviewer"
    speaker_participant = root / "speaker_participant"
    merged = root / "merged"

    for d in (speaker_interviewer, speaker_participant, merged):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "speaker_interviewer": speaker_interviewer,
        "speaker_participant": speaker_participant,
        "merged": merged,
    }

