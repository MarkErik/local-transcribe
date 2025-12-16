#!/usr/bin/env python3
"""
Transcriber plugin using whisper.cpp for speech-to-text transcription.

This plugin uses the whisper.cpp binary (installed via Homebrew) to perform
efficient transcription with word-level timestamps. It supports chunked processing
for long audio files with stitching.

Installation:
    brew install whisper-cpp

Models should be placed in: .models/transcribers/whisper_cpp/
Download GGML models from: https://huggingface.co/ggerganov/whisper.cpp
"""

from typing import List, Optional, Union, Dict, Any
import os
import pathlib
import math
import tempfile
import subprocess
import re
import shutil
import numpy as np
import librosa
from scipy.io import wavfile
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.program_logger import get_logger, log_completion, log_progress


class WhisperCppTranscriberProvider(TranscriberProvider):
    """Transcriber provider using whisper.cpp for speech-to-text transcription.
    
    Supports chunked processing for long audio files with configurable chunk size and overlap.
    Uses the --output-wts flag internally to generate word-level timestamps.
    """

    def __init__(self):
        # Model mapping: user-friendly name -> model filename
        self.model_mapping = {
            "base": "ggml-base.bin",
            "base.en": "ggml-base.en.bin",
            "small": "ggml-small.bin",
            "small.en": "ggml-small.en.bin",
            "medium": "ggml-medium.bin",
            "medium.en": "ggml-medium.en.bin",
            "large-v3": "ggml-large-v3.bin",
            "turbo-v3": "ggml-large-v3-turbo.bin",
        }
        self.logger = get_logger()
        self.selected_model = None  # Will be set during transcription
        
        # Chunking configuration
        self.chunk_length_seconds = 600.0  # 10 minutes - configurable chunk length
        self.overlap_seconds = 10.0        # 10 seconds - configurable overlap between chunks
        self.min_chunk_seconds = 30.0      # 30 seconds - minimum chunk length
        
        # Binary path cache
        self._binary_path = None

    @property
    def name(self) -> str:
        return "whisper_cpp"

    @property
    def short_name(self) -> str:
        return "Whisper.cpp"

    @property
    def description(self) -> str:
        return "Whisper.cpp transcription with word-level timestamps"

    @property
    def has_builtin_alignment(self) -> bool:
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model and selected_model in self.model_mapping:
            return [self.model_mapping[selected_model]]
        # Default to base.en model
        return [self.model_mapping["base.en"]]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def _find_whisper_cpp_binary(self) -> str:
        """Find the whisper.cpp binary on the system.
        
        Checks in the following order:
        1. WHISPER_CPP_PATH environment variable
        2. Homebrew installation paths
        3. Common system paths
        
        Returns:
            Path to the whisper.cpp binary
            
        Raises:
            FileNotFoundError: If the binary cannot be found
        """
        if self._binary_path and os.path.exists(self._binary_path):
            return self._binary_path
        
        # Check environment variable
        env_path = os.environ.get("WHISPER_CPP_PATH")
        if env_path and os.path.exists(env_path):
            self._binary_path = env_path
            return self._binary_path
        
        # Check for whisper-cli command in PATH
        whisper_cpp_cmd = shutil.which("whisper-cli")
        if whisper_cpp_cmd:
            self._binary_path = whisper_cpp_cmd
            return self._binary_path
        
        # Common installation paths
        possible_paths = [
            "/opt/homebrew/bin/whisper-cpp",  # Apple Silicon Homebrew
            "/usr/local/bin/whisper-cpp",      # Intel Homebrew
            os.path.expanduser("~/whisper.cpp/main"),
            "./whisper.cpp/main",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self._binary_path = path
                return self._binary_path
        
        raise FileNotFoundError(
            "whisper.cpp binary not found. Install with: brew install whisper-cpp\n"
            "Or set WHISPER_CPP_PATH environment variable to the binary location.\n"
            "Note: The command is 'whisper-cli'."
        )

    def _parse_wts_file(self, wts_file_path: str, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse whisper.cpp word timestamp file (.wts).
        
        The .wts format from whisper.cpp --output-wts is:
        [start_ms] --> [end_ms] word
        
        Example:
        [1250] --> [1580] hello
        [1580] --> [2120] world
        
        Args:
            wts_file_path: Path to the .wts file
            role: Speaker role/name to assign to words
            
        Returns:
            List of word dicts with text, start, end, speaker
        """
        words = []
        
        if not os.path.exists(wts_file_path):
            self.logger.warning(f"WTS file not found: {wts_file_path}")
            return words
        
        # Pattern to match: [start_ms] --> [end_ms] word
        pattern = re.compile(r'\[(\d+)\]\s*-->\s*\[(\d+)\]\s+(.+)')
        
        try:
            with open(wts_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    match = pattern.match(line)
                    if match:
                        start_ms = int(match.group(1))
                        end_ms = int(match.group(2))
                        text = match.group(3).strip()
                        
                        # Convert milliseconds to seconds
                        words.append({
                            "text": text,
                            "start": start_ms / 1000.0,
                            "end": end_ms / 1000.0,
                            "speaker": role
                        })
                    else:
                        self.logger.warning(f"Could not parse WTS line: {line}")
        except Exception as e:
            self.logger.error(f"Error parsing WTS file {wts_file_path}: {e}")
        
        return words

    def _transcribe_single_file_with_wts(
        self,
        audio_path: str,
        model_path: str,
        role: Optional[str] = None,
        num_threads: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe a single audio file using whisper.cpp with word timestamps.
        
        Args:
            audio_path: Path to the audio file
            model_path: Path to the GGML model file
            role: Speaker role/name
            num_threads: Number of CPU threads to use (None = auto)
            
        Returns:
            List of word dicts with text, start, end, speaker
        """
        binary_path = self._find_whisper_cpp_binary()
        
        # Create temporary directory for output files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Build command
            cmd = [
                binary_path,
                "-m", model_path,
                "-f", audio_path,
                "--output-wts",  # Generate word timestamp file
                "--language", "en",
                "--output-dir", temp_dir,
            ]
            
            # Add thread count if specified
            if num_threads:
                cmd.extend(["-t", str(num_threads)])
            
            # Execute whisper.cpp (no timeout - allow long transcriptions)
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    self.logger.error(f"whisper.cpp failed: {result.stderr}")
                    return []
                
                # Find the .wts file in the output directory
                # whisper.cpp typically names output based on input filename
                audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
                wts_file = os.path.join(temp_dir, f"{audio_basename}.wts")
                
                # If not found with expected name, look for any .wts file
                if not os.path.exists(wts_file):
                    wts_files = [f for f in os.listdir(temp_dir) if f.endswith('.wts')]
                    if wts_files:
                        wts_file = os.path.join(temp_dir, wts_files[0])
                    else:
                        self.logger.error(f"No .wts file found in {temp_dir}")
                        return []
                
                # Parse the .wts file
                return self._parse_wts_file(wts_file, role)
                
            except Exception as e:
                self.logger.error(f"Error running whisper.cpp: {e}")
                return []

    def _transcribe_chunk_with_wts(
        self,
        chunk_audio: np.ndarray,
        sr: int,
        chunk_start_time: float,
        model_path: str,
        role: Optional[str] = None,
        num_threads: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe a single audio chunk and return timestamped words.
        
        Args:
            chunk_audio: Audio data as numpy array
            sr: Sample rate
            chunk_start_time: Absolute start time of this chunk in full audio (seconds)
            model_path: Path to the GGML model file
            role: Speaker role/name
            num_threads: Number of CPU threads to use
            
        Returns:
            List of word dicts with text, start, end, speaker (absolute timestamps)
        """
        # Create temporary WAV file for this chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write chunk to temporary file
            # Convert float32 [-1, 1] to int16 [-32768, 32767]
            audio_int16 = (chunk_audio * 32767).astype(np.int16)
            wavfile.write(tmp_path, sr, audio_int16)
            
            # Transcribe chunk
            words = self._transcribe_single_file_with_wts(tmp_path, model_path, role, num_threads)
            
            # Adjust timestamps to absolute time
            for word in words:
                word["start"] += chunk_start_time
                word["end"] += chunk_start_time
            
            return words
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """Transcribe audio using whisper.cpp.
        
        This method is required by the abstract base class but is never called
        in practice since whisper.cpp has builtin alignment. Use transcribe_with_alignment instead.
        """
        raise NotImplementedError("whisper.cpp only supports transcribe_with_alignment. This method should never be called.")

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Union[List[WordSegment], List[Dict[str, Any]]]:
        """
        Transcribe audio with word-level timestamps using whisper.cpp.
        
        For audio shorter than chunk_length_seconds, returns List[WordSegment].
        For longer audio, returns a list of chunk dictionaries for stitching.
        
        Args:
            audio_path: Path to the audio file
            role: Speaker role/name
            device: Device to use (ignored for whisper.cpp - uses CPU)
            **kwargs: Additional options:
                - transcriber_model: Model name (default: "base.en")
                - verbose: Enable progress logging (default: False)
                - num_threads: CPU threads to use (default: auto)
                - chunk_length_seconds: Override chunk size
                - overlap_seconds: Override overlap size
        
        Returns:
            List[WordSegment]: For short audio (< chunk_length_seconds)
            List[Dict[str, Any]]: For long audio, each dict has 'chunk_id', 'chunk_start_time',
                                  and 'words' (List[Dict] with text/start/end/speaker)
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Get configuration
        self.selected_model = kwargs.get('transcriber_model', 'base.en')
        model_filename = self.model_mapping.get(self.selected_model, self.model_mapping["base.en"])
        verbose = kwargs.get('verbose', False)
        num_threads = kwargs.get('num_threads', None)
        
        # Override chunking parameters if provided
        if 'chunk_length_seconds' in kwargs:
            self.chunk_length_seconds = kwargs['chunk_length_seconds']
        if 'overlap_seconds' in kwargs:
            self.overlap_seconds = kwargs['overlap_seconds']
        
        # Find model path
        models_dir = pathlib.Path(".models") / "transcribers" / "whisper_cpp"
        model_path = models_dir / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Download GGML models from: https://huggingface.co/ggerganov/whisper.cpp"
            )
        
        # Load audio to check duration
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(wav) / sr
        
        # Check if audio is too short
        if duration < self.min_chunk_seconds:
            raise ValueError(
                f"Audio duration ({duration:.1f}s) is too short. "
                f"Minimum required: {self.min_chunk_seconds}s"
            )
        
        # For short audio, use direct transcription (no chunking)
        if duration < self.chunk_length_seconds:
            words = self._transcribe_single_file_with_wts(
                audio_path,
                str(model_path),
                role,
                num_threads
            )
            
            # Convert to WordSegment format
            word_segments = []
            for word_info in words:
                word_segments.append(WordSegment(
                    text=word_info["text"],
                    start=word_info["start"],
                    end=word_info["end"],
                    speaker=role
                ))
            
            return word_segments
        
        # Chunked processing for long audio
        chunk_samples = int(self.chunk_length_seconds * sr)
        overlap_samples = int(self.overlap_seconds * sr)
        min_chunk_samples = int(self.min_chunk_seconds * sr)
        
        chunks_with_timestamps = []
        total_samples = len(wav)
        effective_chunk_length = self.chunk_length_seconds - self.overlap_seconds
        num_chunks = math.ceil(duration / effective_chunk_length) if effective_chunk_length > 0 else 1
        
        if verbose:
            log_progress(f"Audio duration: {duration:.1f}s - processing in {num_chunks} chunks")
        
        chunk_start = 0
        chunk_num = 0
        prev_chunk_wav = None
        
        while chunk_start < total_samples:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            chunk_wav = wav[chunk_start:chunk_end]
            
            # Calculate absolute start time of this chunk in the full audio
            chunk_start_time = chunk_start / sr
            
            if verbose:
                chunk_duration_sec = len(chunk_wav) / sr
                log_progress(f"Processing chunk {chunk_num} of {num_chunks} (starts at {chunk_start_time:.2f}s, duration: {chunk_duration_sec:.1f}s)")
            
            # Handle last chunk if it's too small
            if len(chunk_wav) < min_chunk_samples:
                if prev_chunk_wav is not None and chunks_with_timestamps:
                    # Merge with previous chunk
                    non_overlapping_part = chunk_wav[overlap_samples:] if len(chunk_wav) > overlap_samples else chunk_wav
                    merged_wav = np.concatenate([prev_chunk_wav, non_overlapping_part])
                    
                    # Use the start time from the previous chunk
                    prev_chunk_start_time = chunks_with_timestamps[-1].get("chunk_start_time", chunk_start_time - (len(prev_chunk_wav) / sr))
                    
                    # Re-transcribe merged chunk with timestamps
                    timestamped_words = self._transcribe_chunk_with_wts(
                        merged_wav, sr, prev_chunk_start_time, str(model_path), role, num_threads
                    )
                    
                    # Update last chunk
                    chunks_with_timestamps[-1] = {
                        "chunk_id": chunks_with_timestamps[-1]["chunk_id"],
                        "chunk_start_time": prev_chunk_start_time,
                        "words": timestamped_words
                    }
                    
                    if verbose:
                        log_progress(f"Merged small final chunk with previous chunk")
                else:
                    # Process as normal if it's the only chunk
                    timestamped_words = self._transcribe_chunk_with_wts(
                        chunk_wav, sr, chunk_start_time, str(model_path), role, num_threads
                    )
                    chunks_with_timestamps.append({
                        "chunk_id": chunk_num,
                        "chunk_start_time": chunk_start_time,
                        "words": timestamped_words
                    })
            else:
                # Normal chunk processing
                timestamped_words = self._transcribe_chunk_with_wts(
                    chunk_wav, sr, chunk_start_time, str(model_path), role, num_threads
                )
                chunks_with_timestamps.append({
                    "chunk_id": chunk_num,
                    "chunk_start_time": chunk_start_time,
                    "words": timestamped_words
                })
            
            prev_chunk_wav = chunk_wav
            
            # Break if we've reached the end
            if chunk_end == total_samples:
                break
            
            # Move to next chunk with overlap
            chunk_start = chunk_start + chunk_samples - overlap_samples
        
        if verbose:
            total_words = sum(len(chunk["words"]) for chunk in chunks_with_timestamps)
            log_progress(f"Transcription and alignment complete: {len(chunks_with_timestamps)} chunks, {total_words} words total")
        
        return chunks_with_timestamps

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload models (not applicable for whisper.cpp - models are just files)."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure whisper.cpp binary and models are available.
        
        Note: whisper.cpp models must be manually downloaded. This method checks
        for their existence and provides download instructions if missing.
        """
        # Check binary
        try:
            self._find_whisper_cpp_binary()
            log_completion("whisper.cpp binary found")
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise
        
        # Check model files
        whisper_cpp_models_dir = models_dir / "transcribers" / "whisper_cpp"
        whisper_cpp_models_dir.mkdir(parents=True, exist_ok=True)
        
        missing_models = []
        for model in models:
            model_path = whisper_cpp_models_dir / model
            if not model_path.exists():
                missing_models.append((model, model_path))
        
        if missing_models:
            self.logger.error("whisper.cpp models must be manually downloaded.")
            self.logger.error("\nMissing models:")
            for model_name, model_path in missing_models:
                self.logger.error(f"  - {model_name}")
            
            self.logger.error(f"\nDownload instructions:")
            self.logger.error(f"1. Visit: https://huggingface.co/ggerganov/whisper.cpp/tree/main")
            self.logger.error(f"2. Download the required .bin files")
            self.logger.error(f"3. Place them in: {whisper_cpp_models_dir}")
            self.logger.error(f"\nExample commands:")
            for model_name, _ in missing_models:
                url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_name}"
                self.logger.error(f"  curl -L '{url}' -o '{whisper_cpp_models_dir}/{model_name}'")
            
            raise FileNotFoundError(
                f"Missing whisper.cpp model files. See log above for download instructions."
            )

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline."""
        missing = []
        whisper_cpp_models_dir = models_dir / "transcribers" / "whisper_cpp"
        
        for model in models:
            model_path = whisper_cpp_models_dir / model
            if not model_path.exists():
                missing.append(model)
        
        return missing


def register_transcriber_plugins():
    """Register whisper.cpp transcriber plugin."""
    registry.register_transcriber_provider(WhisperCppTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
