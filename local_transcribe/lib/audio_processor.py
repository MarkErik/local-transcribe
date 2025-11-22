#!/usr/bin/env python3
# lib/audio_processor.py - Audio processing utilities

import shutil
from pathlib import Path
from typing import Optional


def standardize_audio(audio_path: str, outdir: Path, api, speaker_name: Optional[str] = None):
    """
    Standardize audio file and return standardized path.
    
    Args:
        audio_path: Path to the audio file to standardize
        outdir: Output directory for temporary files
        api: API dictionary with services
        speaker_name: Optional speaker name for logging
        
    Returns:
        Path to the standardized audio file
    """
    # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
    temp_audio_dir = outdir / "temp_audio"
    temp_audio_dir.mkdir(exist_ok=True)
    
    # Standardize the audio file
    standardized_audio = api["standardize_and_get_path"](audio_path, tmpdir=temp_audio_dir)
    
    return standardized_audio


def cleanup_temp_audio(outdir: Path):
    """
    Clean up temporary audio files and directories.
    
    Args:
        outdir: Output directory containing temp_audio subdirectory
    """
    temp_audio_dir = outdir / "temp_audio"
    if temp_audio_dir.exists():
        try:
            shutil.rmtree(temp_audio_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary audio directory {temp_audio_dir}: {e}")