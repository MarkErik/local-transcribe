#!/usr/bin/env python3
"""
VAD-based audio segmentation using Silero VAD.

This module implements intelligent audio segmentation based on Voice Activity Detection (VAD)
using Silero's lightweight but highly accurate VAD model. The segmentation strategy uses:

1. Neural VAD scoring via Silero's `get_speech_timestamps()` function
2. Configurable thresholds optimized for accuracy
3. Maximum segment duration enforcement with intelligent splitting
4. Optional merging of short adjacent segments

Key advantages of Silero VAD:
- Lightweight (~2MB model)
- No authentication required (MIT license)
- Trained on 6000+ languages
- Highly tunable parameters
- Direct timestamp output (no post-processing needed)

This approach ensures chunks start/end at natural speech boundaries rather than arbitrary time points.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import pathlib
import numpy as np
import torch
from local_transcribe.lib.program_logger import get_logger, log_progress, log_debug, log_completion


@dataclass
class SileroVADSegment:
    """Represents a speech segment with start and end times in seconds."""
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end - self.start
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (start_sec, end_sec) tuple."""
        return (self.start, self.end)


class SileroVADSegmenter:
    """
    Voice Activity Detection based audio segmenter using Silero VAD.
    
    Uses Silero's neural VAD to identify speech regions and segments audio
    intelligently at natural speech boundaries.
    """
    
    def __init__(
        self,
        threshold: float = 0.4,
        neg_threshold: Optional[float] = None,
        min_speech_duration_ms: int = 300,
        min_silence_duration_ms: int = 150,
        speech_pad_ms: int = 50,
        max_segment_duration: float = 45.0,
        merge_threshold: float = 45.0,
        device: Optional[str] = None,
        models_dir: Optional[pathlib.Path] = None
    ):
        """
        Initialize the Silero VAD segmenter.
        
        Args:
            threshold: Speech probability threshold (lower = more sensitive). Default 0.4 for accuracy.
            neg_threshold: Threshold to exit speech state. If None, uses threshold - 0.15.
            min_speech_duration_ms: Minimum speech segment duration in milliseconds.
            min_silence_duration_ms: Minimum silence duration to end speech segment.
            speech_pad_ms: Padding added to each side of speech segments.
            max_segment_duration: Maximum segment duration for ASR in seconds.
            merge_threshold: Maximum duration for merged segments in seconds.
            device: Device to run VAD model on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            models_dir: Directory where models are cached (for consistency with other providers).
        """
        self.logger = get_logger()
        
        # VAD parameters (accuracy-optimized defaults)
        self.threshold = threshold
        self.neg_threshold = neg_threshold if neg_threshold is not None else (threshold - 0.15)
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_segment_duration = max_segment_duration
        self.merge_threshold = merge_threshold
        
        # Device configuration
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "cpu"  # Silero works best on CPU for MPS systems
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.models_dir = models_dir
        
        # Model state
        self._model = None
        self._get_speech_timestamps = None
        self._read_audio = None
    
    def _get_cache_dir(self) -> pathlib.Path:
        """Get the cache directory for VAD models."""
        if self.models_dir is None:
            # Default to user's torch hub cache
            return pathlib.Path(torch.hub.get_dir()) / "silero_vad"
        return self.models_dir / "vad" / "silero"
    
    def _load_model(self):
        """Load the Silero VAD model."""
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
                self._model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
                (self._get_speech_timestamps, _, self._read_audio, _, _) = utils
                
                log_completion("Silero VAD model loaded successfully (via torch.hub)")
                
            except Exception as e:
                self.logger.error(f"Failed to load Silero VAD model: {e}")
                raise
    
    def preload_models(self) -> None:
        """Preload the VAD model to cache."""
        log_progress("Preloading Silero VAD model...")
        
        try:
            # Loading the model will download/cache it
            self._load_model()
            log_completion("Silero VAD model preloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to preload Silero VAD model: {e}")
            raise
    
    def check_models_available_offline(self) -> bool:
        """Check if VAD model is available offline without downloading."""
        try:
            # Try to load the model - Silero is small and typically cached
            # after first use via torch.hub or silero-vad package
            self._load_model()
            return True
        except Exception:
            return False
    
    def _split_long_segment(
        self,
        segment: SileroVADSegment,
        audio: torch.Tensor,
        sample_rate: int
    ) -> List[SileroVADSegment]:
        """
        Split a segment that exceeds max_segment_duration.
        
        Uses VAD probabilities to find the best split point at a silence region.
        
        Args:
            segment: The segment to potentially split
            audio: Full audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            List of segments (original if short enough, or split segments)
        """
        if segment.duration <= self.max_segment_duration:
            return [segment]
        
        # Extract segment audio
        start_sample = int(segment.start * sample_rate)
        end_sample = int(segment.end * sample_rate)
        segment_audio = audio[start_sample:end_sample]
        
        # Get speech probabilities for this segment to find silence points
        # We'll run VAD on the segment with stricter parameters to find internal silences
        try:
            # Get timestamps within the segment
            inner_timestamps = self._get_speech_timestamps(
                segment_audio,
                self._model,
                threshold=self.threshold + 0.1,  # Slightly stricter for internal splits
                sampling_rate=sample_rate,
                min_speech_duration_ms=100,  # Shorter minimum to find more boundaries
                min_silence_duration_ms=50,   # Find shorter silences
                speech_pad_ms=10,
                return_seconds=True
            )
            
            if len(inner_timestamps) <= 1:
                # No good split points found, split at midpoint
                mid_time = segment.start + (segment.duration / 2)
                return [
                    SileroVADSegment(segment.start, mid_time),
                    SileroVADSegment(mid_time, segment.end)
                ]
            
            # Find split points at boundaries between detected speech regions
            result_segments = []
            current_start = segment.start
            accumulated_duration = 0.0
            
            for i, ts in enumerate(inner_timestamps):
                speech_start = segment.start + ts['start']
                speech_end = segment.start + ts['end']
                speech_duration = speech_end - speech_start
                
                if accumulated_duration + speech_duration > self.max_segment_duration and accumulated_duration > 0:
                    # Create segment up to before this speech region
                    result_segments.append(SileroVADSegment(current_start, speech_start))
                    current_start = speech_start
                    accumulated_duration = speech_duration
                else:
                    accumulated_duration += speech_duration
                    if i < len(inner_timestamps) - 1:
                        # Add gap to next speech region
                        next_start = segment.start + inner_timestamps[i + 1]['start']
                        accumulated_duration += (next_start - speech_end)
            
            # Add final segment
            if current_start < segment.end:
                result_segments.append(SileroVADSegment(current_start, segment.end))
            
            # Recursively split any still-too-long segments
            final_segments = []
            for seg in result_segments:
                if seg.duration > self.max_segment_duration:
                    # Simple midpoint split for remaining long segments
                    mid_time = seg.start + (seg.duration / 2)
                    final_segments.append(SileroVADSegment(seg.start, mid_time))
                    final_segments.append(SileroVADSegment(mid_time, seg.end))
                else:
                    final_segments.append(seg)
            
            return final_segments
            
        except Exception as e:
            log_debug(f"Error splitting segment: {e}, using midpoint split")
            # Fallback to simple midpoint split
            mid_time = segment.start + (segment.duration / 2)
            left = SileroVADSegment(segment.start, mid_time)
            right = SileroVADSegment(mid_time, segment.end)
            
            # Recursively handle if still too long
            return self._split_long_segment(left, audio, sample_rate) + \
                   self._split_long_segment(right, audio, sample_rate)
    
    def _merge_short_segments(
        self,
        segments: List[SileroVADSegment]
    ) -> List[SileroVADSegment]:
        """
        Merge adjacent segments if their combined duration is below merge_threshold.
        
        Args:
            segments: List of segments sorted by start time
            
        Returns:
            List of merged segments
        """
        if not segments or len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Calculate combined duration including the gap
            combined_duration = next_seg.end - current.start
            
            if combined_duration <= self.merge_threshold:
                # Merge: extend current to include next
                current = SileroVADSegment(current.start, next_seg.end)
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        
        merge_count = len(segments) - len(merged)
        if merge_count > 0:
            log_debug(f"Merge phase: {len(segments)} -> {len(merged)} segments ({merge_count} merges)")
        
        return merged
    
    def segment_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        """
        Full VAD segmentation pipeline.
        
        Args:
            waveform: Audio waveform as numpy array (mono)
            sample_rate: Sample rate of the audio (must be 8000 or 16000)
            
        Returns:
            List of (start_sec, end_sec) tuples representing speech segments
        """
        log_progress("Running Silero VAD segmentation pipeline...")
        
        # Validate sample rate
        if sample_rate not in [8000, 16000]:
            raise ValueError(f"Silero VAD only supports 8000 or 16000 Hz sample rates, got {sample_rate}")
        
        # Ensure model is loaded
        self._load_model()
        
        # Convert numpy array to torch tensor
        if isinstance(waveform, np.ndarray):
            audio_tensor = torch.from_numpy(waveform).float()
        else:
            audio_tensor = waveform
        
        # Ensure 1D tensor
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        
        audio_duration = len(audio_tensor) / sample_rate
        
        # Get speech timestamps using Silero's built-in function
        try:
            speech_timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=self.threshold,
                neg_threshold=self.neg_threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=True
            )
        except Exception as e:
            self.logger.error(f"Silero VAD failed: {e}")
            # Return entire audio as single segment on failure
            log_debug("VAD failed, returning full audio as single segment")
            return [(0.0, audio_duration)]
        
        if not speech_timestamps:
            # No speech detected - return the whole audio as one segment
            log_debug("No speech segments detected by VAD, using full audio")
            return [(0.0, audio_duration)]
        
        # Convert to SileroVADSegment objects
        segments = [
            SileroVADSegment(ts['start'], ts['end'])
            for ts in speech_timestamps
        ]
        
        log_debug(f"Silero VAD detected {len(segments)} initial speech segments")
        
        # Apply maximum segment duration constraint
        constrained_segments = []
        for seg in segments:
            constrained_segments.extend(
                self._split_long_segment(seg, audio_tensor, sample_rate)
            )
        
        if len(constrained_segments) != len(segments):
            log_debug(f"Split long segments: {len(segments)} -> {len(constrained_segments)} segments")
        
        # Optionally merge short adjacent segments
        if self.merge_threshold > 0:
            constrained_segments = self._merge_short_segments(constrained_segments)
        
        # Convert to time tuples
        time_segments = [seg.to_tuple() for seg in constrained_segments]
        
        # Log summary
        total_speech = sum(end - start for start, end in time_segments)
        log_completion(
            f"Silero VAD segmentation complete: {len(time_segments)} segments, "
            f"{total_speech:.1f}s speech / {audio_duration:.1f}s total"
        )
        
        return time_segments
    
    def segment_audio_from_file(
        self,
        audio_path: str,
        target_sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        """
        Convenience method to segment audio directly from a file path.
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate (audio will be resampled if needed)
            
        Returns:
            List of (start_sec, end_sec) tuples representing speech segments
        """
        import librosa
        
        waveform, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        return self.segment_audio(waveform, sr)


def create_silero_vad_segmenter(
    max_segment_duration: float = 45.0,
    merge_threshold: float = 45.0,
    device: Optional[str] = None,
    accuracy_optimized: bool = True,
    **kwargs
) -> SileroVADSegmenter:
    """
    Factory function to create a SileroVADSegmenter with common defaults.
    
    Args:
        max_segment_duration: Maximum segment duration for ASR
        merge_threshold: Maximum duration for merged segments
        device: Device for VAD model
        accuracy_optimized: If True, use accuracy-optimized parameters
        **kwargs: Additional parameters passed to SileroVADSegmenter
        
    Returns:
        Configured SileroVADSegmenter instance
    """
    if accuracy_optimized:
        # Accuracy-optimized defaults
        defaults = {
            "threshold": 0.4,
            "min_speech_duration_ms": 300,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 50,
        }
        # Override with any provided kwargs
        defaults.update(kwargs)
        kwargs = defaults
    
    return SileroVADSegmenter(
        max_segment_duration=max_segment_duration,
        merge_threshold=merge_threshold,
        device=device,
        **kwargs
    )
