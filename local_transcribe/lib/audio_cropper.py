#!/usr/bin/env python3
"""
Audio cropping utility for local_transcribe.

This module provides functionality to:
- Check if two audio files have the same duration
- Crop audio files to a specified number of minutes

Can be used as standalone CLI tool or imported as module.
"""

from __future__ import annotations
import pathlib
import subprocess
import json
import logging
import sys
from typing import Tuple, Optional

# Try to import from local_transcribe package, fall back to standalone if not available
try:
    from .system_output import get_logger, AudioProcessingError, error_context
    _HAVE_LOCAL_TRANSCRIBE = True
except ImportError:
    # Standalone mode - define minimal equivalents
    _HAVE_LOCAL_TRANSCRIBE = False
    
    class AudioProcessingError(Exception):
        """Custom exception for audio processing errors."""
        def __init__(self, message: str, audio_path: Optional[str] = None, cause: Optional[Exception] = None):
            super().__init__(message)
            self.audio_path = audio_path
            self.cause = cause
    
    def error_context(reraise: bool = False):
        """Minimal decorator for error context in standalone mode."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if reraise and not isinstance(e, AudioProcessingError):
                        raise AudioProcessingError(f"Error in {func.__name__}: {e}", cause=e) from e
                    raise
            return wrapper
        return decorator
    
    def get_logger():
        """Get logger instance."""
        return logging.getLogger(__name__)


class AudioCropper:
    """Handles audio file duration checking and cropping operations."""
    
    def __init__(self):
        if _HAVE_LOCAL_TRANSCRIBE:
            self.logger = get_logger()
        else:
            # Set up basic logging for standalone mode
            if not logging.getLogger().handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logging.getLogger().addHandler(handler)
                logging.getLogger().setLevel(logging.INFO)
            self.logger = get_logger()
    
    @error_context(reraise=True)
    def get_audio_duration(self, audio_path: str | pathlib.Path) -> float:
        """
        Get the duration of an audio file in seconds using ffprobe.
        
        Parameters
        ----------
        audio_path : str | pathlib.Path
            Path to the audio file
            
        Returns
        -------
        float
            Duration in seconds
            
        Raises
        ------
        AudioProcessingError
            If duration cannot be determined or ffprobe fails
        """
        try:
            audio_path = pathlib.Path(audio_path).resolve()
            
            if not audio_path.exists():
                raise AudioProcessingError(f"Audio file not found: {audio_path}", audio_path=str(audio_path))
            
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_path)
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            info = json.loads(result.stdout)
            duration = float(info['format']['duration'])
            
            self.logger.info(f"Duration of {audio_path.name}: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            return duration
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffprobe failed to get duration for {audio_path}: {e.stderr}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, audio_path=str(audio_path), cause=e)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            error_msg = f"Failed to parse duration from ffprobe output for {audio_path}: {e}"
            self.logger.error(error_msg)
            raise AudioProcessingError(error_msg, audio_path=str(audio_path), cause=e)
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            else:
                error_msg = f"Unexpected error getting duration for {audio_path}: {e}"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, audio_path=str(audio_path), cause=e)
    
    @error_context(reraise=True)
    def check_duration_compatibility(self, file1: str | pathlib.Path, file2: str | pathlib.Path) -> Tuple[float, float]:
        """
        Check if two audio files have the same duration.
        
        Parameters
        ----------
        file1 : str | pathlib.Path
            Path to first audio file
        file2 : str | pathlib.Path  
            Path to second audio file
            
        Returns
        -------
        Tuple[float, float]
            Durations of both files in seconds
            
        Raises
        ------
        AudioProcessingError
            If durations don't match or cannot be determined
        """
        self.logger.info(f"Checking duration compatibility between {pathlib.Path(file1).name} and {pathlib.Path(file2).name}")
        
        duration1 = self.get_audio_duration(file1)
        duration2 = self.get_audio_duration(file2)
        
        # Allow small tolerance for floating point differences (0.1 seconds)
        tolerance = 0.1
        if abs(duration1 - duration2) > tolerance:
            error_msg = (
                f"Audio files have different durations: "
                f"{pathlib.Path(file1).name} = {duration1:.2f}s, "
                f"{pathlib.Path(file2).name} = {duration2:.2f}s"
            )
            self.logger.error(error_msg)
            raise AudioProcessingError(
                error_msg,
                audio_path=f"{file1}, {file2}"
            )
        
        self.logger.info(f"✓ Audio files have compatible durations: {duration1:.2f}s")
        return duration1, duration2
    
    @error_context(reraise=True)
    def crop_audio(self, 
                   input_path: str | pathlib.Path,
                   output_path: str | pathlib.Path,
                   duration_minutes: float) -> None:
        """
        Crop an audio file to the specified duration using ffmpeg.
        
        Parameters
        ----------
        input_path : str | pathlib.Path
            Path to input audio file
        output_path : str | pathlib.Path
            Path for cropped output file
        duration_minutes : float
            Desired duration in minutes
            
        Raises
        ------
        AudioProcessingError
            If cropping fails or input duration is shorter than requested
        """
        try:
            input_path = pathlib.Path(input_path).resolve()
            output_path = pathlib.Path(output_path).resolve()
            
            # Get input duration
            input_duration = self.get_audio_duration(input_path)
            target_duration_seconds = duration_minutes * 60
            
            # Check if input is long enough
            if input_duration < target_duration_seconds:
                error_msg = (
                    f"Input audio ({input_duration:.2f}s) is shorter than requested "
                    f"duration ({target_duration_seconds:.2f}s)"
                )
                self.logger.error(error_msg)
                raise AudioProcessingError(
                    error_msg,
                    audio_path=str(input_path)
                )
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Cropping {input_path.name} to {duration_minutes:.2f} minutes ({target_duration_seconds:.2f}s)")
            
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-t", str(target_duration_seconds),  # Duration to record/copy
                "-c", "copy",  # Copy streams without re-encoding for speed
                str(output_path)
            ]
            
            self.logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Verify output was created
            if not output_path.exists():
                raise AudioProcessingError(
                    f"ffmpeg failed to create cropped file: {output_path}",
                    audio_path=str(input_path)
                )
            
            # Verify output duration
            actual_duration = self.get_audio_duration(output_path)
            if abs(actual_duration - target_duration_seconds) > 1.0:  # Allow 1s tolerance
                self.logger.warning(
                    f"Output duration ({actual_duration:.2f}s) differs from target "
                    f"({target_duration_seconds:.2f}s)"
                )
            
            self.logger.info(f"✓ Successfully cropped audio to {output_path}")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"ffmpeg cropping failed: {e.stderr}"
            self.logger.error(f"{error_msg} for {input_path}")
            raise AudioProcessingError(error_msg, audio_path=str(input_path), cause=e)
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            else:
                error_msg = f"Unexpected error during audio cropping: {e}"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, audio_path=str(input_path), cause=e)


@error_context(reraise=True)
def crop_audio_files(file1: str | pathlib.Path,
                    file2: str | pathlib.Path, 
                    duration_minutes: float,
                    output_dir: Optional[str | pathlib.Path] = None) -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Convenience function to crop two audio files to the same duration.
    
    Parameters
    ----------
    file1 : str | pathlib.Path
        Path to first audio file
    file2 : str | pathlib.Path  
        Path to second audio file
    duration_minutes : float
        Desired duration in minutes for both files
    output_dir : Optional[str | pathlib.Path]
        Directory for output files. If None, uses same directory as inputs.
        
    Returns
    -------
    Tuple[pathlib.Path, pathlib.Path]
        Paths to the cropped output files
        
    Raises
    ------
    AudioProcessingError
        If processing fails at any step
    """
    cropper = AudioCropper()
    
    # Check duration compatibility first
    duration1, duration2 = cropper.check_duration_compatibility(file1, file2)
    
    # Determine output directory
    if output_dir is None:
        output_dir = pathlib.Path(file1).parent
    else:
        output_dir = pathlib.Path(output_dir)
    
    # Generate output paths
    file1_path = pathlib.Path(file1)
    file2_path = pathlib.Path(file2)
    
    output1 = output_dir / f"{file1_path.stem}_cropped_{duration_minutes}min{file1_path.suffix}"
    output2 = output_dir / f"{file2_path.stem}_cropped_{duration_minutes}min{file2_path.suffix}"
    
    # Crop both files
    cropper.crop_audio(file1, output1, duration_minutes)
    cropper.crop_audio(file2, output2, duration_minutes)
    
    return output1, output2


def main():
    """Command-line interface for audio cropping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crop two audio files to the same duration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio1.wav audio2.wav 5.0
  %(prog)s /path/to/file1.mp3 /path/to/file2.mp3 10.5 --output-dir ./cropped
        """
    )
    
    parser.add_argument("file1", help="Path to first audio file")
    parser.add_argument("file2", help="Path to second audio file")
    parser.add_argument("duration_minutes", type=float, help="Desired duration in minutes")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        output1, output2 = crop_audio_files(
            args.file1,
            args.file2,
            args.duration_minutes,
            args.output_dir
        )
        
        print(f"\n✓ Successfully cropped audio files:")
        print(f"  {output1}")
        print(f"  {output2}")
        
    except AudioProcessingError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()