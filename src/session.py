from __future__ import annotations
import pathlib
from datetime import datetime

def ensure_session_dirs(output_dir: str | pathlib.Path, *, create_speaker_dirs: bool = True) -> dict[str, pathlib.Path]:
    """
    Creates a consistent directory structure for outputs and returns paths.

    Parameters
    - output_dir: base output directory to create
    - create_speaker_dirs: when False, do not create `speaker_interviewer` and
      `speaker_participant` subdirectories. This is useful for combined/mixed
      audio mode where per-speaker folders are not needed.
    """
    root = pathlib.Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    speaker_interviewer = root / "speaker_interviewer"
    speaker_participant = root / "speaker_participant"
    merged = root / "merged"

    # Always ensure the merged folder exists; speaker folders are optional.
    merged.mkdir(parents=True, exist_ok=True)

    if create_speaker_dirs:
        speaker_interviewer.mkdir(parents=True, exist_ok=True)
        speaker_participant.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "speaker_interviewer": speaker_interviewer,
        "speaker_participant": speaker_participant,
        "merged": merged,
    }

