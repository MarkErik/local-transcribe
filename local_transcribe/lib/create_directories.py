#!/usr/bin/env python3
from __future__ import annotations
import pathlib
from datetime import datetime

def ensure_session_dirs(output_dir: str | pathlib.Path, mode: str, speaker_files: dict = None, verbose: bool = False, capabilities: dict = None) -> dict[str, pathlib.Path]:
    """
    Creates a consistent directory structure for outputs and returns paths.
    Mode can be 'combined_audio' or 'split_audio'.
    For split_audio, speaker_files dict maps speaker names to file paths.
    If verbose, creates Intermediate_Outputs subdirectories based on capabilities.
    capabilities: dict with keys 'mode', 'unified', 'has_builtin_alignment', 'aligner', 'diarization'
    """
    root = pathlib.Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root,
    }

    if verbose:
        needed_intermediate_dirs = _compute_needed_intermediate_dirs(capabilities or {})
        intermediate_dir = root / "Intermediate_Outputs"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        paths["intermediate"] = intermediate_dir
        
        for subdir in needed_intermediate_dirs:
            (intermediate_dir / subdir).mkdir(exist_ok=True)

    return paths


def _compute_needed_intermediate_dirs(capabilities: dict) -> set[str]:
    """
    Compute the set of intermediate subdirectories needed based on capabilities.
    """
    needed = set()
    mode = capabilities.get("mode", "")
    unified = capabilities.get("unified", False)
    has_builtin_alignment = capabilities.get("has_builtin_alignment", False)
    aligner = capabilities.get("aligner", False)
    diarization = capabilities.get("diarization", False)

    if mode == "single_speaker_audio":
        # Single speaker audio: just transcription directory
        needed.add("transcription")
    elif mode == "combined_audio":
        if unified:
            # Unified: only turns
            needed.add("turns")
        else:
            # Separate providers
            if not has_builtin_alignment:
                needed.add("transcription")
            if aligner:
                needed.add("alignment")
            if has_builtin_alignment:
                needed.add("transcription_alignment")
            if diarization:
                needed.add("diarization")
            needed.add("turns")
    elif mode == "split_audio":
        if not has_builtin_alignment:
            needed.add("transcription")
        if aligner:
            needed.add("alignment")
        if has_builtin_alignment:
            needed.add("transcription_alignment")
        needed.add("turns")
    
    return needed
