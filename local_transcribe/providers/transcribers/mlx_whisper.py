#!/usr/bin/env python3
"""
Transcriber plugin using MLX Whisper for Apple Silicon.

Supports chunked processing for long audio files with intelligent stitching.
"""

from typing import List, Optional, Union, Dict, Any
import os
import pathlib
import math
import tempfile
import numpy as np
import librosa
from scipy.io import wavfile
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.program_logger import get_logger, log_completion, log_progress


class MLXWhisperTranscriberProvider(TranscriberProvider):
    """Transcriber provider using MLX Whisper for speech-to-text transcription on Apple Silicon.
    
    Supports chunked processing for long audio files with configurable chunk size and overlap.
    """

    def __init__(self):
        # Model mapping: user-friendly name -> MLX Whisper model repo
        self.model_mapping = {
            "base": "mlx-community/whisper-base-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "turbo-v3": "mlx-community/whisper-large-v3-turbo",
        }
        self.logger = get_logger()
        self.selected_model = None  # Will be set during transcription
        
        # Chunking configuration
        self.chunk_length_seconds = 600.0  # 10 minutes - configurable chunk length
        self.overlap_seconds = 10.0        # 10 seconds - configurable overlap between chunks
        self.min_chunk_seconds = 30.0      # 30 seconds - minimum chunk length

    @property
    def name(self) -> str:
        return "mlx_whisper"

    @property
    def short_name(self) -> str:
        return "MLX Whisper"

    @property
    def description(self) -> str:
        return "MLX Whisper transcription for Apple Silicon"

    @property
    def has_builtin_alignment(self) -> bool:
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model and selected_model in self.model_mapping:
            return [self.model_mapping[selected_model]]
        # Default to base model
        return [self.model_mapping["base"]]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload MLX Whisper models to cache."""
        try:
            import mlx_whisper
            for model in models:
                if model in self.model_mapping.values():  # It's an MLX Whisper model
                    try:
                        # Load and immediately discard to download/cache the model
                        # Use a dummy audio file path that doesn't exist to trigger download without transcribing
                        temp_result = mlx_whisper.transcribe("/dev/null", path_or_hf_repo=model, verbose=False)
                        log_completion(f"MLX Whisper {model} downloaded successfully")
                    except Exception as e:
                        self.logger.warning(f"Failed to preload {model}: {e}")
        except ImportError:
            self.logger.warning("mlx-whisper not available, skipping preload")

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which MLX Whisper models are available offline without downloading."""
        missing = []
        try:
            from huggingface_hub import snapshot_download
            for model in models:
                try:
                    # Try to download with local_files_only=True to check if available
                    snapshot_download(model, local_files_only=True)
                except Exception:
                    missing.append(model)
        except ImportError:
            # If huggingface_hub not available, assume all missing
            missing = models
        return missing

    def _transcribe_single_chunk_with_timestamps(
        self,
        chunk_audio: np.ndarray,
        sr: int,
        chunk_start_time: float,
        model_repo: str,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe a single audio chunk and return timestamped words.
        
        Args:
            chunk_audio: Audio data as numpy array
            sr: Sample rate
            chunk_start_time: Absolute start time of this chunk in full audio (seconds)
            model_repo: MLX Whisper model repository name
            role: Speaker role/name
            
        Returns:
            List of word dicts with text, start, end, speaker (absolute timestamps)
        """
        try:
            import mlx_whisper
        except ImportError:
            raise ImportError("mlx-whisper package is required. Install with: uv add mlx-whisper")
        
        # Create temporary WAV file for this chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write chunk to temporary file
            # Convert float32 [-1, 1] to int16 [-32768, 32767]
            audio_int16 = (chunk_audio * 32767).astype(np.int16)
            wavfile.write(tmp_path, sr, audio_int16)
            
            # Transcribe with word timestamps
            result = mlx_whisper.transcribe(
                tmp_path,
                path_or_hf_repo=model_repo,
                word_timestamps=True,
                verbose=True,
                condition_on_previous_text=False,
                language='en'
            )
            
            # Convert to word dicts with absolute timestamps
            word_dicts = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word_dicts.append({
                        "text": word_info["word"].strip(),
                        "start": word_info["start"] + chunk_start_time,  # Adjust to absolute time
                        "end": word_info["end"] + chunk_start_time,      # Adjust to absolute time
                        "speaker": role
                    })
            
            return word_dicts
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """Transcribe audio using MLX Whisper model.
        
        This method is required by the abstract base class but is never called
        in practice since MLX Whisper has builtin alignment. Use transcribe_with_alignment instead.
        """
        raise NotImplementedError("MLX Whisper only supports transcribe_with_alignment. This method should never be called.")

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> Union[List[WordSegment], List[Dict[str, Any]]]:
        """
        Transcribe audio with word-level timestamps using MLX Whisper.
        
        For audio shorter than chunk_length_seconds, returns List[WordSegment].
        For longer audio, returns a list of chunk dictionaries for stitching.
        
        Returns:
            List[WordSegment]: For short audio (< chunk_length_seconds)
            List[Dict[str, Any]]: For long audio, each dict has 'chunk_id', 'chunk_start_time',
                                  and 'words' (List[Dict] with text/start/end/speaker)
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            import mlx_whisper
        except ImportError:
            raise ImportError("mlx-whisper package is required. Install with: uv add mlx-whisper")

        # Set selected model from kwargs
        self.selected_model = kwargs.get('transcriber_model', 'base')
        model_repo = self.model_mapping.get(self.selected_model, self.model_mapping["base"])
        
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
            result = mlx_whisper.transcribe(
                tmp_path,
                path_or_hf_repo=model_repo,
                word_timestamps=True,
                verbose=True,
                condition_on_previous_text=False,
                language='en'
            )

            # Convert to WordSegment format
            word_segments = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    word_segments.append(WordSegment(
                        text=word_info["word"].strip(),
                        start=word_info["start"],
                        end=word_info["end"],
                        speaker=role
                    ))

            return word_segments
        
        # Chunked processing for long audio
        verbose = kwargs.get('verbose', False)
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
                    timestamped_words = self._transcribe_single_chunk_with_timestamps(
                        merged_wav, sr, prev_chunk_start_time, model_repo, role
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
                    timestamped_words = self._transcribe_single_chunk_with_timestamps(
                        chunk_wav, sr, chunk_start_time, model_repo, role
                    )
                    chunks_with_timestamps.append({
                        "chunk_id": chunk_num,
                        "chunk_start_time": chunk_start_time,
                        "words": timestamped_words
                    })
            else:
                # Normal chunk processing
                timestamped_words = self._transcribe_single_chunk_with_timestamps(
                    chunk_wav, sr, chunk_start_time, model_repo, role
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

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by downloading them."""
        try:
            from huggingface_hub import snapshot_download
            for model in models:
                snapshot_download(model)
                log_completion(f"MLX Whisper {model} downloaded successfully")
        except ImportError:
            raise ImportError("huggingface_hub package is required for downloading models.")


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(MLXWhisperTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()