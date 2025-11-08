#!/usr/bin/env python3
from __future__ import annotations
import pathlib
import tempfile
import subprocess
from typing import Optional
from asr.logging_config import get_logger, AudioProcessingError, ErrorContext, error_context
try:
    from asr.progress import get_progress_tracker
    _HAVE_PROGRESS = True
except ImportError:
    _HAVE_PROGRESS = False

# We use ffmpeg via command line to normalize: mono/16k WAV
# This avoids subtle differences between python audio stacks and keeps it robust.


@error_context(reraise=True)
def _ffmpeg_to_wav16k_mono(src: str | pathlib.Path, dst: str | pathlib.Path) -> None:
    """
    Convert audio to 16kHz mono WAV using ffmpeg.
    
    Parameters
    ----------
    src : str | pathlib.Path
        Source audio file path
    dst : str | pathlib.Path
        Destination WAV file path
        
    Raises
    ------
    AudioProcessingError
        If ffmpeg conversion fails
    """
    logger = get_logger()
    
    try:
        src_path = pathlib.Path(src)
        dst_path = pathlib.Path(dst)
        
        logger.debug(f"Converting {src_path} to {dst_path}")
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-ac", "1",           # mono
            "-ar", "16000",       # 16kHz sample rate
            "-vn",                # no video
            str(dst_path),
        ]
        
        logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
        
        # Add progress tracking if available
        if _HAVE_PROGRESS:
            tracker = get_progress_tracker()
            task = tracker.add_task(f"Converting {src_path.name} to WAV", total=100, stage="audio_conversion")
            tracker.update(task, advance=50, description=f"Converting {src_path.name} - Running ffmpeg")
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if not dst_path.exists():
            raise AudioProcessingError(
                f"ffmpeg failed to create output file: {dst_path}",
                audio_path=str(src_path)
            )
        
        if _HAVE_PROGRESS:
            tracker.update(task, advance=50, description=f"Converting {src_path.name} - Complete")
            tracker.complete_task(task, stage="audio_conversion")
        
        logger.debug(f"Successfully converted {src_path} to {dst_path}")
        
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg conversion failed: {e.stderr}"
        logger.error(f"{error_msg} for {src}")
        raise AudioProcessingError(error_msg, audio_path=str(src), cause=e)
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        else:
            raise AudioProcessingError(
                f"Unexpected error during audio conversion: {e}",
                audio_path=str(src),
                cause=e
            )


@error_context(reraise=True)
def standardize_and_get_path(src: str | pathlib.Path, tmpdir: Optional[str | pathlib.Path] = None) -> pathlib.Path:
    """
    Convert any supported audio to 16kHz mono WAV and return its path.
    Writes to a stable temporary path alongside source or in `tmpdir`.
    
    Parameters
    ----------
    src : str | pathlib.Path
        Source audio file path
    tmpdir : Optional[str | pathlib.Path]
        Temporary directory for output. If None, uses source directory.
        
    Returns
    -------
    pathlib.Path
        Path to the standardized WAV file
        
    Raises
    ------
    AudioProcessingError
        If audio standardization fails
    """
    logger = get_logger()
    
    try:
        src_path = pathlib.Path(src).resolve()
        
        # Validate input
        if not src_path.exists():
            raise AudioProcessingError(f"Source audio file not found: {src_path}", audio_path=str(src_path))
        
        if not src_path.is_file():
            raise AudioProcessingError(f"Source path is not a file: {src_path}", audio_path=str(src_path))
        
        logger.info(f"Standardizing audio: {src_path}")
        
        # Determine output path
        out_dir = pathlib.Path(tmpdir) if tmpdir else src_path.parent
        out_path = out_dir / (src_path.stem + ".wav")
        
        # Check if output already exists and is newer than input
        if out_path.exists() and out_path.stat().st_mtime > src_path.stat().st_mtime:
            logger.debug(f"Using existing standardized file: {out_path}")
            return out_path
        
        # Convert audio
        _ffmpeg_to_wav16k_mono(src_path, out_path)
        
        # Validate output
        if not out_path.exists():
            raise AudioProcessingError(
                f"Standardized audio file was not created: {out_path}",
                audio_path=str(src_path)
            )
        
        if out_path.stat().st_size == 0:
            raise AudioProcessingError(
                f"Standardized audio file is empty: {out_path}",
                audio_path=str(src_path)
            )
        
        logger.info(f"Audio standardization complete: {out_path}")
        return out_path
        
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            logger.error(f"Audio processing error: {e}")
            raise
        else:
            logger.error(f"Unexpected error in audio standardization: {e}")
            raise AudioProcessingError(
                f"Unexpected error during audio standardization: {e}",
                audio_path=str(src),
                cause=e
            )
