#!/usr/bin/env python3
"""
Core Silero VAD model wrapper.

This module provides the foundational Silero VAD model loading and inference
functionality shared across the project. It handles model loading (with fallbacks),
audio preprocessing, and raw timestamp extraction.

Higher-level modules build upon this core:
- lib/vad_silero_segmenter.py: Adds segment combination and splitting logic
- processing/vad/silero_vad.py: Adds pipeline-specific data structures

Note: Heavy imports (torch) are lazily loaded when needed
to avoid slow startup times when this module is not used.
"""

from typing import List, Optional, Any, Callable, TYPE_CHECKING
from pathlib import Path
import numpy as np

# Type hints for lazy-loaded modules
if TYPE_CHECKING:
    import torch

from local_transcribe.lib.program_logger import (
    get_logger,
    log_progress,
    log_debug,
    log_completion,
)


class SileroVADCore:
    """
    Core Silero VAD model wrapper for shared use across the project.
    
    This class handles:
    - Lazy model loading with fallback (silero-vad package → torch.hub)
    - Audio loading and preprocessing
    - Raw speech timestamp extraction
    
    It does NOT handle:
    - Segment combination/splitting logic
    - Conversion to specific data structures
    - Pipeline-specific processing
    """
    
    # Standard sample rate for Silero VAD
    SAMPLE_RATE = 16000
    
    def __init__(
        self,
        threshold: float = 0.45,
        neg_threshold: Optional[float] = None,
        min_speech_duration_ms: int = 300,
        min_silence_duration_ms: int = 150,
        window_size_samples: int = 512,
        speech_pad_ms: int = 60,
        models_dir: Optional[Path] = None,
    ):
        """
        Initialize the Silero VAD core.
        
        Args:
            threshold: Speech probability threshold (0-1). Higher = stricter.
            neg_threshold: Negative threshold for speech end detection.
                          If None, defaults to (threshold - 0.15).
            min_speech_duration_ms: Minimum speech segment duration in ms.
            min_silence_duration_ms: Minimum silence between segments in ms.
            window_size_samples: VAD window size (512 for 16kHz).
            speech_pad_ms: Padding around speech in ms.
            models_dir: Optional path for model caching.
        """
        self.logger = get_logger()
        
        self.threshold = threshold
        self.neg_threshold = neg_threshold if neg_threshold is not None else (threshold - 0.15)
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.models_dir = models_dir
        
        # Model state (lazy loaded)
        self._model: Optional[Any] = None
        self._get_speech_timestamps_func: Optional[Callable[..., Any]] = None
        self._read_audio_func: Optional[Callable[..., Any]] = None
    
    def _load_model(self) -> None:
        """Load the Silero VAD model (lazy loading)."""
        if self._model is not None:
            return
        
        # Lazy import of torch
        import torch
        
        log_progress("Loading Silero VAD model...")
        
        try:
            # Try using the silero-vad package first (preferred)
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            
            self._model = load_silero_vad()
            self._get_speech_timestamps_func = get_speech_timestamps
            self._read_audio_func = read_audio
            
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
                (self._get_speech_timestamps_func, _, self._read_audio_func, _, _) = utils  # type: ignore
                
                log_completion("Silero VAD model loaded successfully (via torch.hub)")
                
            except Exception as e:
                self.logger.error(f"Failed to load Silero VAD model: {e}")
                raise
    
    def preload_model(self) -> None:
        """Preload the VAD model to cache it."""
        log_progress("Preloading Silero VAD model...")
        try:
            self._load_model()
            log_completion("Silero VAD model preloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to preload Silero VAD model: {e}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if the model is already loaded."""
        return self._model is not None
    
    def load_audio(self, audio_path: str) -> "torch.Tensor":
        """
        Load and preprocess audio file for VAD.
        
        Standardizes to 16kHz mono format.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio tensor (16kHz mono)
        """
        # Lazy import of torch
        import torch
        
        self._load_model()
        
        # Use silero_vad's read_audio if available
        if self._read_audio_func is not None:
            try:
                wav = self._read_audio_func(audio_path, sampling_rate=self.SAMPLE_RATE)
                return wav
            except Exception:
                pass  # Fall back to librosa
        
        # Use librosa for robust audio loading (handles many formats)
        import librosa
        
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        wav = torch.from_numpy(audio).float()
        
        return wav
    
    def load_audio_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> "torch.Tensor":
        """
        Convert and resample audio array for VAD.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Audio tensor (16kHz mono)
        """
        # Lazy import of torch
        import torch
        
        # Convert to tensor
        wav = torch.from_numpy(audio_data).float()
        
        # Resample if necessary
        if sample_rate != self.SAMPLE_RATE:
            import librosa
            audio_resampled = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=self.SAMPLE_RATE
            )
            wav = torch.from_numpy(audio_resampled).float()
        
        return wav
    
    def get_speech_timestamps(
        self,
        wav: "torch.Tensor",
        sample_rate: int = 16000,
        return_seconds: bool = False,
        use_neg_threshold: bool = True,
    ) -> List[dict]:
        """
        Get raw speech timestamps from Silero VAD.
        
        Args:
            wav: Audio tensor (should be 16kHz mono for best results)
            sample_rate: Sample rate of the audio (8000 or 16000)
            return_seconds: If True, return timestamps in seconds; 
                           if False, return in samples
            use_neg_threshold: If True, use neg_threshold parameter
                              (not all Silero versions support this)
            
        Returns:
            List of dicts with 'start' and 'end' keys
            (in samples or seconds depending on return_seconds)
        """
        self._load_model()
        
        if self._get_speech_timestamps_func is None:
            raise RuntimeError("VAD model not properly loaded")
        
        # Build kwargs for get_speech_timestamps
        kwargs = {
            "threshold": self.threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "window_size_samples": self.window_size_samples,
            "speech_pad_ms": self.speech_pad_ms,
            "return_seconds": return_seconds,
            "sampling_rate": sample_rate,
        }
        
        # Add neg_threshold if requested (not all versions support it)
        if use_neg_threshold:
            kwargs["neg_threshold"] = self.neg_threshold
        
        try:
            timestamps = self._get_speech_timestamps_func(
                wav,
                self._model,
                **kwargs
            )
        except TypeError as e:
            # Some versions don't support neg_threshold, retry without it
            if "neg_threshold" in str(e) and use_neg_threshold:
                log_debug("neg_threshold not supported, retrying without it")
                kwargs.pop("neg_threshold", None)
                timestamps = self._get_speech_timestamps_func(
                    wav,
                    self._model,
                    **kwargs
                )
            else:
                raise
        
        return timestamps
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of an audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        wav = self.load_audio(audio_path)
        return len(wav) / self.SAMPLE_RATE
    
    @property
    def config(self) -> dict:
        """Return current configuration as dictionary."""
        return {
            "threshold": self.threshold,
            "neg_threshold": self.neg_threshold,
            "min_speech_duration_ms": self.min_speech_duration_ms,
            "min_silence_duration_ms": self.min_silence_duration_ms,
            "window_size_samples": self.window_size_samples,
            "speech_pad_ms": self.speech_pad_ms,
        }


# Default parameters that work well for typical speech
DEFAULT_VAD_PARAMS = {
    "threshold": 0.45,
    "min_speech_duration_ms": 300,
    "min_silence_duration_ms": 150,
    "window_size_samples": 512,
    "speech_pad_ms": 60,
}

# Parameters tuned for the segmenter (slightly different defaults)
SEGMENTER_VAD_PARAMS = {
    "threshold": 0.45,
    "min_speech_duration_ms": 300,
    "min_silence_duration_ms": 150,
    "speech_pad_ms": 60,
}
