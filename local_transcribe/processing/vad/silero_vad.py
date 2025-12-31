#!/usr/bin/env python3
"""
Silero VAD wrapper for VAD-driven split-audio pipeline.

This module provides a pipeline-specific interface to Silero VAD for
speech detection in audio files, returning VADSegment objects.

NOTE: This module is maintained for backward compatibility.
For new code, use providers.vad.SileroVADProvider instead.

For core VAD functionality, see: lib/silero_vad_core.py
"""

from typing import List, Optional
from pathlib import Path
import numpy as np

from local_transcribe.processing.vad.types import VADSegment
from local_transcribe.lib.program_logger import log_progress, log_completion, get_logger
from local_transcribe.lib.silero_vad_core import SileroVADCore, DEFAULT_VAD_PARAMS


class SileroVADProcessor:
    """
    Wrapper for Silero VAD with configurable parameters.
    
    This processor detects speech regions in audio files and returns
    VADSegment objects with absolute timestamps. It builds upon
    SileroVADCore for model loading and raw VAD inference.
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
        
        # Store parameters for reference
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.models_dir = models_dir
        
        # Initialize the core VAD processor
        self._vad_core = SileroVADCore(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms,
            models_dir=models_dir,
        )
    
    def _load_model(self) -> None:
        """Ensure the VAD model is loaded (delegates to core)."""
        self._vad_core._load_model()
    
    def get_speech_timestamps(self, wav) -> List[dict]:
        """
        Get raw speech timestamps from Silero VAD.
        
        Args:
            wav: Audio tensor (16kHz mono)
            
        Returns:
            List of dicts with 'start' and 'end' keys (in samples)
        """
        return self._vad_core.get_speech_timestamps(
            wav,
            sample_rate=self.SAMPLE_RATE,
            return_seconds=False,  # Return samples for precision
            use_neg_threshold=False,  # This processor doesn't use neg_threshold
        )
    
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
        log_progress(f"Running VAD on {speaker_id} audio: {audio_path}")
        
        # Load audio using core
        wav = self._vad_core.load_audio(audio_path)
        
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
        # Convert and resample using core
        wav = self._vad_core.load_audio_array(audio_data, sample_rate)
        
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
        return self._vad_core.get_audio_duration(audio_path)
    
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
