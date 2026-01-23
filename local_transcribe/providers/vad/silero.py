#!/usr/bin/env python3
"""
Silero VAD provider.

This module provides voice activity detection using the Silero VAD model.
"""

from typing import List, Optional, Any, Callable, TYPE_CHECKING, Tuple
from pathlib import Path
import numpy as np

from local_transcribe.lib.program_logger import (
    get_logger,
    log_progress,
    log_debug,
    log_completion,
)

if TYPE_CHECKING:
    import torch
    from local_transcribe.processing.vad.types import VADSegment


# VAD parameters
INTERVIEW_VAD_PARAMS = {
    "threshold": 0.45,
    "min_speech_duration_ms": 300,
    "min_silence_duration_ms": 150,
    "speech_pad_ms": 60,
}


class SileroVADProvider:
    """Silero VAD provider.
    
    Uses fixed parameters. No configuration needed.
    This provider handles:
    - Lazy model loading with fallback (silero-vad package → torch.hub)
    - Audio loading and preprocessing
    - Speech region detection returning VADSegment objects
    """
    
    # Standard sample rate for Silero VAD
    SAMPLE_RATE = 16000
    
    # Parameters
    THRESHOLD = INTERVIEW_VAD_PARAMS["threshold"]
    NEG_THRESHOLD = THRESHOLD - 0.15
    MIN_SPEECH_DURATION_MS = INTERVIEW_VAD_PARAMS["min_speech_duration_ms"]
    MIN_SILENCE_DURATION_MS = INTERVIEW_VAD_PARAMS["min_silence_duration_ms"]
    WINDOW_SIZE_SAMPLES = 512
    SPEECH_PAD_MS = INTERVIEW_VAD_PARAMS["speech_pad_ms"]
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the Silero VAD provider.
        
        Args:
            models_dir: Optional path for model caching.
        """
        self.logger = get_logger()
        self.models_dir = models_dir
        self._model: Optional[Any] = None
        self._get_speech_timestamps_func: Optional[Callable[..., Any]] = None
        self._read_audio_func: Optional[Callable[..., Any]] = None
    
    @property
    def name(self) -> str:
        return "silero"
    
    @property
    def description(self) -> str:
        return "Silero VAD"
    
    def _load_model(self) -> None:
        """Load the Silero VAD model (lazy loading)."""
        if self._model is not None:
            return
        
        import torch
        log_progress("Loading Silero VAD model...")
        
        try:
            # Try using the silero-vad package first (preferred)
            from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
            
            self._model = load_silero_vad()
            self._get_speech_timestamps_func = get_speech_timestamps
            self._read_audio_func = read_audio
            
            log_completion("Silero VAD model loaded (silero-vad package)")
            
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
                # utils returns a tuple of functions
                utils_tuple: Tuple[Any, Any, Any, Any, Any] = utils  # type: ignore
                (self._get_speech_timestamps_func, _, self._read_audio_func, _, _) = utils_tuple
                
                log_completion("Silero VAD model loaded (torch.hub)")
                
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
    
    def detect_speech(
        self,
        audio_path: str,
        speaker_id: str = "unknown"
    ) -> List["VADSegment"]:
        """Detect speech regions in audio file.
        
        Args:
            audio_path: Path to audio file
            speaker_id: Speaker identifier for returned segments
            
        Returns:
            List of VADSegment objects with speech regions
        """
        self._load_model()
        
        # Load audio using silero_vad's read_audio if available
        if self._read_audio_func is not None:
            try:
                wav = self._read_audio_func(audio_path, sampling_rate=self.SAMPLE_RATE)
            except Exception:
                # Fall back to librosa
                import librosa
                audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
                import torch
                wav = torch.from_numpy(audio).float()
        else:
            import librosa
            audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
            import torch
            wav = torch.from_numpy(audio).float()
        
        return self._detect_speech_impl(wav, speaker_id)
    
    def detect_speech_from_array(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_id: str = "unknown"
    ) -> List["VADSegment"]:
        """Detect speech regions from audio array.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of audio
            speaker_id: Speaker identifier for returned segments
            
        Returns:
            List of VADSegment objects with speech regions
        """
        import torch
        self._load_model()
        
        # Resample if needed
        if sample_rate != self.SAMPLE_RATE:
            import librosa
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=self.SAMPLE_RATE
            )
        
        wav = torch.from_numpy(audio_data).float()
        return self._detect_speech_impl(wav, speaker_id)
    
    def _detect_speech_impl(
        self,
        wav: "torch.Tensor",
        speaker_id: str
    ) -> List["VADSegment"]:
        """Internal implementation of speech detection.
        
        Args:
            wav: Audio tensor (16kHz mono)
            speaker_id: Speaker identifier
            
        Returns:
            List of VADSegment objects
        """
        # Import here to avoid circular imports
        from local_transcribe.processing.vad.types import VADSegment
        
        if self._get_speech_timestamps_func is None:
            raise RuntimeError("VAD model not properly loaded")
        
        # Build kwargs for get_speech_timestamps
        kwargs = {
            "threshold": self.THRESHOLD,
            "min_speech_duration_ms": self.MIN_SPEECH_DURATION_MS,
            "min_silence_duration_ms": self.MIN_SILENCE_DURATION_MS,
            "window_size_samples": self.WINDOW_SIZE_SAMPLES,
            "speech_pad_ms": self.SPEECH_PAD_MS,
            "return_seconds": False,  # Return samples for precision
            "sampling_rate": self.SAMPLE_RATE,
        }
        
        # Add neg_threshold (not all versions support it)
        kwargs["neg_threshold"] = self.NEG_THRESHOLD
        
        try:
            timestamps = self._get_speech_timestamps_func(
                wav,
                self._model,
                **kwargs
            )
        except TypeError as e:
            # Some versions don't support neg_threshold, retry without it
            if "neg_threshold" in str(e):
                log_debug("neg_threshold not supported, retrying without it")
                kwargs.pop("neg_threshold", None)
                timestamps = self._get_speech_timestamps_func(
                    wav,
                    self._model,
                    **kwargs
                )
            else:
                raise
        
        segments = []
        for i, ts in enumerate(timestamps):
            start_s = ts['start'] / self.SAMPLE_RATE
            end_s = ts['end'] / self.SAMPLE_RATE
            segments.append(VADSegment(
                segment_id=i,
                speaker_id=speaker_id,
                start_s=start_s,
                end_s=end_s,
            ))
        
        total_speech = sum(s.duration_s for s in segments)
        log_completion(
            f"VAD complete for {speaker_id}: {len(segments)} segments, "
            f"{total_speech:.1f}s total speech"
        )
        
        return segments
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        self._load_model()
        
        if self._read_audio_func is not None:
            try:
                wav = self._read_audio_func(audio_path, sampling_rate=self.SAMPLE_RATE)
                return len(wav) / self.SAMPLE_RATE
            except Exception:
                pass
        
        # Fallback to librosa
        import librosa
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        return len(audio) / self.SAMPLE_RATE
    
    @property
    def config(self) -> dict:
        """Return current configuration as dictionary."""
        return {
            "threshold": self.THRESHOLD,
            "neg_threshold": self.NEG_THRESHOLD,
            "min_speech_duration_ms": self.MIN_SPEECH_DURATION_MS,
            "min_silence_duration_ms": self.MIN_SILENCE_DURATION_MS,
            "window_size_samples": self.WINDOW_SIZE_SAMPLES,
            "speech_pad_ms": self.SPEECH_PAD_MS,
        }
