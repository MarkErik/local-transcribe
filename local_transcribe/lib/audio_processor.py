#!/usr/bin/env python3
# lib/audio_processor.py - Audio processing utilities

import shutil
from pathlib import Path
from typing import Optional


def standardize_audio(audio_path: str, outdir: Path, tracker, api, speaker_name: Optional[str] = None):
    """
    Standardize audio file and return standardized path.
    
    Args:
        audio_path: Path to the audio file to standardize
        outdir: Output directory for temporary files
        tracker: Progress tracker instance
        api: API dictionary with services
        speaker_name: Optional speaker name for progress tracking
        
    Returns:
        Path to the standardized audio file
    """
    # Create task description based on whether we're processing a specific speaker
    if speaker_name:
        task_description = f"Audio standardization for {speaker_name}"
        update_description_1 = f"Standardizing {speaker_name} audio"
        update_description_2 = f"{speaker_name} audio standardization complete"
    else:
        task_description = "Audio standardization"
        update_description_1 = "Standardizing combined audio"
        update_description_2 = "Audio standardization complete"
    
    # Add standardization task to tracker
    std_task = tracker.add_task(task_description, total=100, stage="standardization")
    
    # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
    temp_audio_dir = outdir / "temp_audio"
    temp_audio_dir.mkdir(exist_ok=True)
    
    # Update progress for standardization start
    tracker.update(std_task, advance=50, description=update_description_1)
    
    # Standardize the audio file
    standardized_audio = api["standardize_and_get_path"](audio_path, tmpdir=temp_audio_dir)
    
    # Complete the standardization task
    tracker.update(std_task, advance=50, description=update_description_2)
    tracker.complete_task(std_task, stage="standardization")
    
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