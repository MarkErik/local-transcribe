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
from typing import Tuple, Optional, List

# Try to import from local_transcribe package, fall back to standalone if not available
try:
    from .program_logger import get_logger, AudioProcessingError, error_context
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
                   duration_minutes: Optional[float] = None,
                   start_minutes: float = 0.0,
                   end_minutes: Optional[float] = None) -> None:
        """
        Crop an audio file to the specified duration or time range using ffmpeg.
        
        Parameters
        ----------
        input_path : str | pathlib.Path
            Path to input audio file
        output_path : str | pathlib.Path
            Path for cropped output file
        duration_minutes : float, optional
            Desired duration in minutes (from start of file). Deprecated in favor
            of start_minutes/end_minutes but kept for backward compatibility.
        start_minutes : float, optional
            Start time in minutes (default: 0.0)
        end_minutes : float, optional
            End time in minutes. If provided with start_minutes, crops the range
            [start_minutes, end_minutes]. If None and duration_minutes is set,
            uses duration_minutes from start_minutes.
            
        Raises
        ------
        AudioProcessingError
            If cropping fails or input duration is shorter than requested
            
        Examples
        --------
        # Crop first 5 minutes (traditional behavior)
        crop_audio(input, output, duration_minutes=5.0)
        
        # Crop from 4 minutes to 12 minutes (8 minute output)
        crop_audio(input, output, start_minutes=4.0, end_minutes=12.0)
        """
        try:
            input_path = pathlib.Path(input_path).resolve()
            output_path = pathlib.Path(output_path).resolve()
            
            # Get input duration
            input_duration = self.get_audio_duration(input_path)
            
            # Determine start and target duration in seconds
            start_seconds = start_minutes * 60
            
            if end_minutes is not None:
                # Range-based cropping: start_minutes to end_minutes
                if end_minutes <= start_minutes:
                    error_msg = (
                        f"End time ({end_minutes:.2f} min) must be greater than "
                        f"start time ({start_minutes:.2f} min)"
                    )
                    self.logger.error(error_msg)
                    raise AudioProcessingError(error_msg, audio_path=str(input_path))
                
                end_seconds = end_minutes * 60
                target_duration_seconds = end_seconds - start_seconds
                
                # Check if input is long enough for the requested end time
                if input_duration < end_seconds:
                    error_msg = (
                        f"Input audio ({input_duration:.2f}s / {input_duration/60:.2f} min) is shorter than "
                        f"requested end time ({end_seconds:.2f}s / {end_minutes:.2f} min)"
                    )
                    self.logger.error(error_msg)
                    raise AudioProcessingError(error_msg, audio_path=str(input_path))
                    
                self.logger.info(
                    f"Cropping {input_path.name} from {start_minutes:.2f} min to {end_minutes:.2f} min "
                    f"(output duration: {target_duration_seconds/60:.2f} min / {target_duration_seconds:.2f}s)"
                )
            elif duration_minutes is not None:
                # Traditional duration-based cropping from start_minutes
                target_duration_seconds = duration_minutes * 60
                required_duration = start_seconds + target_duration_seconds
                
                # Check if input is long enough
                if input_duration < required_duration:
                    error_msg = (
                        f"Input audio ({input_duration:.2f}s) is shorter than requested "
                        f"duration ({required_duration:.2f}s from start {start_seconds:.2f}s)"
                    )
                    self.logger.error(error_msg)
                    raise AudioProcessingError(error_msg, audio_path=str(input_path))
                
                self.logger.info(
                    f"Cropping {input_path.name} to {duration_minutes:.2f} minutes "
                    f"({target_duration_seconds:.2f}s) from {start_minutes:.2f} min"
                )
            else:
                error_msg = "Either duration_minutes or end_minutes must be provided"
                self.logger.error(error_msg)
                raise AudioProcessingError(error_msg, audio_path=str(input_path))
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
            ]
            
            # Add start time if not starting from beginning
            if start_seconds > 0:
                cmd.extend(["-ss", str(start_seconds)])
            
            cmd.extend([
                "-t", str(target_duration_seconds),  # Duration to record/copy
            ])
            
            # Choose appropriate codec based on output format
            if output_path.suffix.lower() == '.wav':
                cmd.extend(["-c:a", "pcm_s16le"])  # PCM 16-bit for WAV
            else:
                cmd.extend(["-c", "copy"])  # Copy streams for other formats
            
            cmd.append(str(output_path))
            
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
                    file2: Optional[str | pathlib.Path] = None, 
                    duration_minutes: Optional[float] = None,
                    output_dir: Optional[str | pathlib.Path] = None,
                    start_minutes: float = 0.0,
                    end_minutes: Optional[float] = None) -> List[pathlib.Path]:
    """
    Convenience function to crop one or two audio files to the same duration or time range.
    
    Parameters
    ----------
    file1 : str | pathlib.Path
        Path to first audio file
    file2 : str | pathlib.Path, optional
        Path to second audio file (optional)
    duration_minutes : float, optional
        Desired duration in minutes for both files (from start). Used when
        end_minutes is not provided.
    output_dir : Optional[str | pathlib.Path]
        Directory for output files. If None, uses same directory as inputs.
    start_minutes : float, optional
        Start time in minutes (default: 0.0)
    end_minutes : float, optional
        End time in minutes. If provided, crops the range [start_minutes, end_minutes].
        
    Returns
    -------
    list[pathlib.Path]
        List of paths to the cropped output files
        
    Raises
    ------
    AudioProcessingError
        If processing fails at any step
        
    Examples
    --------
    # Crop first 5 minutes of one file
    crop_audio_files(file1, duration_minutes=5.0)
    
    # Crop first 5 minutes of both files
    crop_audio_files(file1, file2, duration_minutes=5.0)
    
    # Crop from 4 minutes to 12 minutes (8 minute output)
    crop_audio_files(file1, file2, start_minutes=4.0, end_minutes=12.0)
    """
    cropper = AudioCropper()
    
    files = [file1]
    if file2 is not None:
        files.append(file2)
    
    # Check duration compatibility if two files
    if file2 is not None:
        duration1, duration2 = cropper.check_duration_compatibility(file1, file2)
    
    # Determine output directory
    if output_dir is None:
        output_dir = pathlib.Path(file1).parent
    else:
        output_dir = pathlib.Path(output_dir)
    
    output_paths = []
    for file_path in files:
        file_path_obj = pathlib.Path(file_path)
        
        if end_minutes is not None:
            # Range-based naming: e.g., "file_cropped_4-12min.wav"
            output_name = f"{file_path_obj.stem}_cropped_{start_minutes}-{end_minutes}min{file_path_obj.suffix}"
        else:
            # Traditional duration naming
            output_name = f"{file_path_obj.stem}_cropped_{duration_minutes}min{file_path_obj.suffix}"
        
        output_path = output_dir / output_name
        output_paths.append(output_path)
    
    # Crop all files
    for input_path, output_path in zip(files, output_paths):
        cropper.crop_audio(input_path, output_path, duration_minutes=duration_minutes, 
                           start_minutes=start_minutes, end_minutes=end_minutes)
    
    return output_paths


def main():
    """Command-line interface for audio cropping."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crop one or two audio files to the same duration or time range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop first 5 minutes of one file
  %(prog)s audio1.wav 5.0
  
  # Crop first 5 minutes of both files
  %(prog)s audio1.wav audio2.wav 5.0
  
  # Crop from 4 minutes to 12 minutes (8 minute output)
  %(prog)s audio1.wav audio2.wav 4.0 12.0
  
  # Crop with output directory
  %(prog)s /path/to/file1.mp3 10.5 --output-dir ./cropped
  %(prog)s /path/to/file1.mp3 /path/to/file2.mp3 10.5 --output-dir ./cropped
  %(prog)s /path/to/file1.mp3 /path/to/file2.mp3 2.0 8.5 --output-dir ./cropped
        """
    )
    
    parser.add_argument("file1", help="Path to first audio file")
    parser.add_argument("file2", nargs='?', help="Path to second audio file (optional)")
    parser.add_argument(
        "time_args", 
        type=float, 
        nargs='+',
        metavar="TIME",
        help="One number: duration in minutes from start. Two numbers: start_minutes end_minutes"
    )
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Parse time arguments
    if len(args.time_args) == 1:
        # Single number: duration from start
        duration_minutes = args.time_args[0]
        start_minutes = 0.0
        end_minutes = None
    elif len(args.time_args) == 2:
        # Two numbers: start and end times
        duration_minutes = None
        start_minutes = args.time_args[0]
        end_minutes = args.time_args[1]
    else:
        parser.error("Expected 1 or 2 time arguments (duration OR start_minutes end_minutes)")
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        output_paths = crop_audio_files(
            args.file1,
            args.file2,
            duration_minutes=duration_minutes,
            output_dir=args.output_dir,
            start_minutes=start_minutes,
            end_minutes=end_minutes
        )
        
        print(f"\n✓ Successfully cropped audio files:")
        for path in output_paths:
            print(f"  {path}")
        
    except AudioProcessingError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()