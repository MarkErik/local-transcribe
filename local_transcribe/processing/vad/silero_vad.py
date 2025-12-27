#!/usr/bin/env python3
"""
Silero VAD wrapper for VAD-driven split-audio pipeline.

This module provides a library utility that wraps Silero VAD with
configurable parameters for speech detection in audio files.
"""

import logging
from typing import List, Optional, Any, Callable
from pathlib import Path
import numpy as np
import torch

from local_transcribe.processing.vad.data_structures import VADSegment
from local_transcribe.lib.program_logger import log_progress, log_debug, log_completion, get_logger


class SileroVADProcessor:
    """
    Wrapper for Silero VAD with configurable parameters.
    
    This processor detects speech regions in audio files and returns
    VADSegment objects with absolute timestamps.
    """
    
    # Standard sample rate for Silero VAD
    SAMPLE_RATE = 16000
    
    def __init__(
        self,
        threshold: float = 0.5,           # Speech probability threshold
        min_speech_duration_ms: int = 250, # Minimum speech segment duration
        min_silence_duration_ms: int = 100, # Minimum silence between segments
        window_size_samples: int = 512,    # VAD window size (512 for 16kHz)
        speech_pad_ms: int = 30,           # Padding around speech
        models_dir: Optional[Path] = None,
    ):
        """
        Initialize the Silero VAD processor.
        
        Args:
            threshold: Speech probability threshold (0-1). Higher = stricter.
            min_speech_duration_ms: Minimum speech segment duration in ms.
            min_silence_duration_ms: Minimum silence between segments in ms.
            window_size_samples: VAD window size (512 for 16kHz).
            speech_pad_ms: Padding around speech in ms.
            models_dir: Optional path for model caching.
        """
        self.logger = get_logger()
        
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.models_dir = models_dir
        
        # Model state (lazy loaded)
        self._model: Optional[Any] = None
        self._get_speech_timestamps: Optional[Callable[..., Any]] = None
        self._read_audio: Optional[Callable[..., Any]] = None
    
    def _load_model(self) -> None:
        """Load the Silero VAD model (lazy loading)."""
        if self._model is not None:
            return
        
        log_progress("Loading Silero VAD model...")
        
        try:
            # Try using the silero-vad package first (preferred)
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            
            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._read_audio = read_audio
            
            log_completion("Silero VAD model loaded successfully (via silero-vad package)")
            
        except ImportError:
            # Fallback to torch.hub
            log_debug("silero-vad package not available, falling back to torch.hub")
            try:
                self._model = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='utils',
                    force_reload=False,
                    trust_repo=True
                )
                (self._get_speech_timestamps, _, self._read_audio, _, _) = utils  # type: ignore
                
                log_completion("Silero VAD model loaded successfully (via torch.hub)")
                
            except Exception as e:
                self.logger.error(f"Failed to load Silero VAD model: {e}")
                raise
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file for VAD.
        
        Standardizes to 16kHz mono format.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio tensor (16kHz mono)
        """
        # Use silero_vad's read_audio if available
        if self._read_audio is not None:
            try:
                wav = self._read_audio(audio_path, sampling_rate=self.SAMPLE_RATE)
                return wav
            except Exception:
                pass  # Fall back to librosa
        
        # Use librosa for robust audio loading (handles many formats)
        import librosa
        import numpy as np
        
        audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        wav = torch.from_numpy(audio).float()
        
        return wav
    
    def get_speech_timestamps(self, wav: torch.Tensor) -> List[dict]:
        """
        Get raw speech timestamps from Silero VAD.
        
        Args:
            wav: Audio tensor (16kHz mono)
            
        Returns:
            List of dicts with 'start' and 'end' keys (in samples)
        """
        self._load_model()
        
        if self._get_speech_timestamps is None:
            raise RuntimeError("VAD model not properly loaded")
        
        timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.window_size_samples,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,  # Return samples for precision
            sampling_rate=self.SAMPLE_RATE,
        )
        
        return timestamps
    
    def process_audio(
        self,
        audio_path: str,
        speaker_id: str
    ) -> List[VADSegment]:
        """
        Process audio file and return VAD segments.
        
        Args:
            audio_path: Path to audio file
            speaker_id: Speaker identifier for these segments
            
        Returns:
            List of VADSegment objects with absolute timestamps
        """
        self._load_model()
        
        log_progress(f"Running VAD on {speaker_id} audio: {audio_path}")
        
        # Load audio
        wav = self._load_audio(audio_path)
        
        # Get speech timestamps
        timestamps = self.get_speech_timestamps(wav)
        
        # Convert to VADSegments
        segments: List[VADSegment] = []
        for i, ts in enumerate(timestamps):
            start_samples = ts['start']
            end_samples = ts['end']
            
            # Convert samples to seconds
            start_s = start_samples / self.SAMPLE_RATE
            end_s = end_samples / self.SAMPLE_RATE
            
            segment = VADSegment(
                segment_id=i,
                speaker_id=speaker_id,
                start_s=start_s,
                end_s=end_s,
            )
            segments.append(segment)
        
        total_speech = sum(s.duration_s for s in segments)
        log_completion(
            f"VAD complete for {speaker_id}: {len(segments)} segments, "
            f"{total_speech:.1f}s total speech"
        )
        
        return segments
    
    def process_audio_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_id: str
    ) -> List[VADSegment]:
        """
        Process audio array directly and return VAD segments.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            speaker_id: Speaker identifier
            
        Returns:
            List of VADSegment objects
        """
        self._load_model()
        
        # Convert to tensor
        wav = torch.from_numpy(audio_data).float()
        
        # Resample if necessary using librosa
        if sample_rate != self.SAMPLE_RATE:
            import librosa
            audio_resampled = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.SAMPLE_RATE)
            wav = torch.from_numpy(audio_resampled).float()
        
        # Get timestamps and convert to segments
        timestamps = self.get_speech_timestamps(wav)
        
        segments: List[VADSegment] = []
        for i, ts in enumerate(timestamps):
            start_s = ts['start'] / self.SAMPLE_RATE
            end_s = ts['end'] / self.SAMPLE_RATE
            
            segment = VADSegment(
                segment_id=i,
                speaker_id=speaker_id,
                start_s=start_s,
                end_s=end_s,
            )
            segments.append(segment)
        
        return segments
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of an audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        wav = self._load_audio(audio_path)
        return len(wav) / self.SAMPLE_RATE
    
    @property
    def config(self) -> dict:
        """Return current configuration as dictionary."""
        return {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "window_size_samples": self.window_size_samples,
            "speech_pad_ms": self.speech_pad_ms,
        }
