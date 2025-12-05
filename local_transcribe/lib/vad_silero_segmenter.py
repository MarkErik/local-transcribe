#!/usr/bin/env python3
"""
VAD-based audio segmentation using Silero VAD.

This module implements intelligent audio segmentation based on Voice Activity Detection (VAD)
using Silero's lightweight but highly accurate VAD model. The segmentation strategy uses:

1. Neural VAD scoring via Silero's `get_speech_timestamps()` function
2. Maximum segment duration enforcement with intelligent splitting
3. Optional merging of short adjacent segments

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
from datetime import datetime
from local_transcribe.lib.program_logger import get_logger, log_progress, log_debug, log_completion, get_output_context, get_output_context


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
            threshold: Speech probability threshold (lower = more sensitive).
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
        
        # VAD parameters
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
            
            log_debug(f"Split mode: segment {segment.start:.2f}-{segment.end:.2f}s, found {len(inner_timestamps)} internal speech regions")
            
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
        sample_rate: int = 16000,
        debug_file_path: Optional[str] = None
    ) -> List[Tuple[float, float]]:
        """
        Full VAD segmentation pipeline.
        
        Args:
            waveform: Audio waveform as numpy array (mono)
            sample_rate: Sample rate of the audio (must be 8000 or 16000)
            debug_file_path: Optional path to write debug information to a text file
            
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
        
        # Debug logging setup
        debug_enabled = get_output_context().should_log("DEBUG") and debug_file_path is not None
        debug_lines = []
        
        if debug_enabled:
            debug_lines.append("=" * 80)
            debug_lines.append("SILERO VAD SEGMENTATION DEBUG LOG")
            debug_lines.append("=" * 80)
            debug_lines.append(f"Timestamp: {datetime.now().isoformat()}")
            debug_lines.append(f"Audio duration: {audio_duration:.3f} seconds")
            debug_lines.append(f"Sample rate: {sample_rate} Hz")
            debug_lines.append("")
            debug_lines.append("VAD Parameters:")
            debug_lines.append(f"  threshold: {self.threshold}")
            debug_lines.append(f"  neg_threshold: {self.neg_threshold}")
            debug_lines.append(f"  min_speech_duration_ms: {self.min_speech_duration_ms}")
            debug_lines.append(f"  min_silence_duration_ms: {self.min_silence_duration_ms}")
            debug_lines.append(f"  speech_pad_ms: {self.speech_pad_ms}")
            debug_lines.append(f"  max_segment_duration: {self.max_segment_duration}")
            debug_lines.append(f"  merge_threshold: {self.merge_threshold}")
            debug_lines.append("")
        
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
            if debug_enabled:
                debug_lines.append("VAD PROCESSING FAILED")
                debug_lines.append(f"Error: {e}")
                debug_lines.append("")
                debug_lines.append("FALLBACK: Returning full audio as single segment")
                debug_lines.append(f"Full audio segment: 0.000 - {audio_duration:.3f}")
                debug_lines.append("")
                debug_lines.append("NON-SPEECH REGIONS:")
                debug_lines.append("  (No speech detected - entire audio is non-speech)")
                self._write_debug_file(debug_file_path, debug_lines)
            return [(0.0, audio_duration)]
        
        if debug_enabled:
            debug_lines.append("RAW SILERO SPEECH TIMESTAMPS:")
            if speech_timestamps:
                for i, ts in enumerate(speech_timestamps):
                    debug_lines.append(f"  Segment {i+1}: {ts['start']:.3f} - {ts['end']:.3f} ({ts['end']-ts['start']:.3f}s)")
            else:
                debug_lines.append("  No speech timestamps returned")
            debug_lines.append("")
        
        if not speech_timestamps:
            # No speech detected - return the whole audio as one segment
            log_debug("No speech segments detected by VAD, using full audio")
            if debug_enabled:
                debug_lines.append("NO SPEECH DETECTED")
                debug_lines.append("FALLBACK: Returning full audio as single segment")
                debug_lines.append(f"Full audio segment: 0.000 - {audio_duration:.3f}")
                debug_lines.append("")
                debug_lines.append("NON-SPEECH REGIONS:")
                debug_lines.append("  (No speech detected - entire audio is non-speech)")
                self._write_debug_file(debug_file_path, debug_lines)
            return [(0.0, audio_duration)]
        
        # Convert to SileroVADSegment objects
        segments = [
            SileroVADSegment(ts['start'], ts['end'])
            for ts in speech_timestamps
        ]
        
        if debug_enabled:
            debug_lines.append("INITIAL SPEECH SEGMENTS (after Silero VAD):")
            for i, seg in enumerate(segments):
                debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s)")
            debug_lines.append("")
        
        log_debug(f"Silero VAD detected {len(segments)} initial speech segments")
        
        # Apply maximum segment duration constraint
        constrained_segments = []
        for seg in segments:
            constrained_segments.extend(
                self._split_long_segment(seg, audio_tensor, sample_rate)
            )
        
        if debug_enabled:
            debug_lines.append("AFTER SPLITTING LONG SEGMENTS:")
            for i, seg in enumerate(constrained_segments):
                debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s)")
            debug_lines.append("")
        
        if len(constrained_segments) != len(segments):
            log_debug(f"Split long segments: {len(segments)} -> {len(constrained_segments)} segments")
        
        # Optionally merge short adjacent segments
        if self.merge_threshold > 0:
            constrained_segments = self._merge_short_segments(constrained_segments)
        
        if debug_enabled:
            debug_lines.append("AFTER MERGING SHORT SEGMENTS:")
            for i, seg in enumerate(constrained_segments):
                debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s)")
            debug_lines.append("")
        
        # Convert to time tuples
        time_segments = [seg.to_tuple() for seg in constrained_segments]
        
        # Log summary
        total_speech = sum(end - start for start, end in time_segments)
        log_completion(
            f"Silero VAD segmentation complete: {len(time_segments)} segments, "
            f"{total_speech:.1f}s speech / {audio_duration:.1f}s total"
        )
        
        if debug_enabled:
            debug_lines.append("FINAL SPEECH SEGMENTS:")
            for i, (start, end) in enumerate(time_segments):
                debug_lines.append(f"  Segment {i+1}: {start:.3f} - {end:.3f} ({end-start:.3f}s)")
            debug_lines.append("")
            debug_lines.append("NON-SPEECH REGIONS:")
            if time_segments:
                # Before first segment
                if time_segments[0][0] > 0:
                    debug_lines.append(f"  0.000 - {time_segments[0][0]:.3f} ({time_segments[0][0]:.3f}s silence)")
                # Between segments
                for i in range(len(time_segments) - 1):
                    gap_start = time_segments[i][1]
                    gap_end = time_segments[i + 1][0]
                    gap_duration = gap_end - gap_start
                    debug_lines.append(f"  {gap_start:.3f} - {gap_end:.3f} ({gap_duration:.3f}s silence)")
                # After last segment
                if time_segments[-1][1] < audio_duration:
                    debug_lines.append(f"  {time_segments[-1][1]:.3f} - {audio_duration:.3f} ({audio_duration - time_segments[-1][1]:.3f}s silence)")
            else:
                debug_lines.append("  (No speech segments - entire audio is non-speech)")
            debug_lines.append("")
            debug_lines.append("SUMMARY:")
            debug_lines.append(f"  Total audio duration: {audio_duration:.3f}s")
            debug_lines.append(f"  Total speech duration: {total_speech:.3f}s")
            debug_lines.append(f"  Speech percentage: {(total_speech/audio_duration*100):.1f}%" if audio_duration > 0 else "  Speech percentage: N/A")
            debug_lines.append(f"  Number of speech segments: {len(time_segments)}")
            self._write_debug_file(debug_file_path, debug_lines)
        
        return time_segments
    
    def _write_debug_file(self, debug_file_path: str, lines: List[str]) -> None:
        """Write debug information to a text file."""
        try:
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            log_debug(f"Debug information written to {debug_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write debug file {debug_file_path}: {e}")
    
    def segment_audio_from_file(
        self,
        audio_path: str,
        target_sample_rate: int = 16000,
        debug_file_path: Optional[str] = None
    ) -> List[Tuple[float, float]]:
        """
        Convenience method to segment audio directly from a file path.
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate (audio will be resampled if needed)
            debug_file_path: Optional path to write debug information to a text file
            
        Returns:
            List of (start_sec, end_sec) tuples representing speech segments
        """
        import librosa
        
        waveform, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
        return self.segment_audio(waveform, sr, debug_file_path)


def create_silero_vad_segmenter(
    max_segment_duration: float = 45.0,
    merge_threshold: float = 45.0,
    device: Optional[str] = None,
    custom_settings: bool = True,
    **kwargs
) -> SileroVADSegmenter:
    """
    Factory function to create a SileroVADSegmenter with common defaults.
    
    Args:
        max_segment_duration: Maximum segment duration for ASR
        merge_threshold: Maximum duration for merged segments
        device: Device for VAD model
        custom_settings: If True, use custom parameters
        **kwargs: Additional parameters passed to SileroVADSegmenter
        
    Returns:
        Configured SileroVADSegmenter instance
    """
    if custom_settings:
        # custom settings
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
