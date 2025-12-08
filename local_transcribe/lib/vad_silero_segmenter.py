#!/usr/bin/env python3
"""
VAD-based audio segmentation using Silero VAD.

This module implements intelligent audio segmentation based on Voice Activity Detection (VAD)
using the SileroVAD model. The segmentation strategy uses:

1. Neural VAD scoring via Silero's `get_speech_timestamps()` function
2. Maximum segment duration enforcement with intelligent splitting
3. Optional merging of short adjacent segments

This approach ensures chunks start/end at natural speech boundaries rather than arbitrary time points.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Callable
import pathlib
import numpy as np
import torch
from datetime import datetime
from local_transcribe.lib.program_logger import get_logger, log_progress, log_debug, log_completion, get_output_context


@dataclass
class VADSegment:
    """Represents a speech segment with start and end times in seconds."""
    start_time: float
    end_time: float
    duration: float
    original_number: int = 0
    
    def __post_init__(self):
        # Validate duration calculation with tolerance for floating-point precision
        calculated_duration = self.end_time - self.start_time
        if abs(calculated_duration - self.duration) > 0.001:  # 1ms tolerance
            self.duration = calculated_duration
    
    @property
    def start(self) -> float:
        """Backward compatibility."""
        return self.start_time
    
    @property
    def end(self) -> float:
        """Backward compatibility."""
        return self.end_time
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (start_sec, end_sec) tuple."""
        return (self.start_time, self.end_time)


@dataclass
class CombinedSegment:
    """Represents a combined segment containing multiple VAD segments."""
    segments: List[VADSegment]
    
    def __post_init__(self):
        if not self.segments:
            raise ValueError("CombinedSegment must contain at least one VAD segment")
        
        # Calculate combined properties
        self.start_time = min(seg.start_time for seg in self.segments)
        self.end_time = max(seg.end_time for seg in self.segments)
        self.duration = self.end_time - self.start_time
    
    @property
    def start(self) -> float:
        """Backward compatibility."""
        return self.start_time
    
    @property
    def end(self) -> float:
        """Backward compatibility."""
        return self.end_time
    
    def get_segment_count(self) -> int:
        """Get the number of original segments in this combined segment."""
        return len(self.segments)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (start_sec, end_sec) tuple."""
        return (self.start_time, self.end_time)


@dataclass
class SegmentCombinationConfig:
    """Configuration for segment combination and splitting."""
    # Core combination thresholds
    micro_pause_threshold: float = 0.5
    thinking_pause_threshold: float = 2.0
    natural_boundary_threshold: float = 5.0
    
    # Duration constraints
    max_segment_duration: float = 30.0
    min_segment_duration: float = 0.5
    
    # Context-aware parameters
    disfluency_threshold: float = 0.5
    min_disfluency_context: float = 1.0
    
    # Splitting parameters - tiered gap thresholds
    min_gap_for_primary_split: float = 1.0
    min_gap_for_secondary_split: float = 0.7
    min_gap_for_tertiary_split: float = 0.6
    
    # Advanced splitting parameters
    max_splits_per_segment: int = 3
    min_split_segment_duration: float = 5.0
    preferred_split_gap_ratio: float = 0.3
    lookahead_segments: int = 5


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
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 150,
        speech_pad_ms: int = 50,
        device: Optional[str] = None,
        models_dir: Optional[pathlib.Path] = None,
        combination_config: Optional[SegmentCombinationConfig] = None,
        max_segment_duration: float = 30.0
    ):
        """
        Initialize the Silero VAD segmenter.
        
        Args:
            threshold: Speech probability threshold.
            neg_threshold: Threshold to exit speech state. If None, uses threshold - 0.15.
            min_speech_duration_ms: Minimum speech segment duration in milliseconds.
            min_silence_duration_ms: Minimum silence duration to end speech segment.
            speech_pad_ms: Padding added to each side of speech segments.
            device: Device to run VAD model on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            models_dir: Directory where models are cached (for consistency with other providers).
            combination_config: Configuration for segment combination and splitting.
            max_segment_duration: Maximum segment duration before splitting.
        """
        self.logger = get_logger()
        
        # VAD parameters
        self.threshold = threshold
        self.neg_threshold = neg_threshold if neg_threshold is not None else (threshold - 0.15)
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_segment_duration = max_segment_duration
        
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
        self._model: Optional[Any] = None
        self._get_speech_timestamps: Optional[Callable[..., Any]] = None
        self._read_audio: Optional[Callable[..., Any]] = None
        
        # Initialize combination config
        if combination_config is None:
            self.combination_config = SegmentCombinationConfig(
                max_segment_duration=max_segment_duration
            )
        else:
            self.combination_config = combination_config
    
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
    
    def _calculate_split_score(
        self,
        segment: CombinedSegment,
        split_index: int,
        gap: float,
        first_seg_duration: float,
        second_seg_duration: float,
        config: SegmentCombinationConfig
    ) -> float:
        """Calculate score for a potential split point."""
        # Balance Score (40% weight)
        duration_ratio = min(first_seg_duration, second_seg_duration) / max(first_seg_duration, second_seg_duration)
        balance_score = duration_ratio * 0.4
        
        # Gap Score (30% weight)
        gap_score = min(gap / config.min_gap_for_primary_split, 1.0) * 0.3
        
        # Context Score (30% weight)
        context_score = 0.0
        if split_index > 1:
            # Add context from previous gap
            prev_gap = segment.segments[split_index - 1].start_time - segment.segments[split_index - 2].end_time
            context_score += max(0, (gap - prev_gap)) * 0.1
        
        if split_index < len(segment.segments) - 1:
            # Add context from next gap
            next_gap = segment.segments[split_index + 1].start_time - segment.segments[split_index].end_time
            context_score += max(0, (gap - next_gap)) * 0.1
        
        return balance_score + gap_score + context_score
    
    def _find_best_split_points(
        self,
        segment: CombinedSegment,
        config: SegmentCombinationConfig
    ) -> List[int]:
        """Find the best split points for a segment."""
        potential_splits = []
        
        for i in range(1, len(segment.segments)):
            gap = segment.segments[i].start_time - segment.segments[i-1].end_time
            
            # Check gap against tiered threshold system
            if gap >= config.min_gap_for_tertiary_split:
                # Verify split would create meaningful segments
                first_seg_duration = segment.segments[i].start_time - segment.segments[0].start_time
                second_seg_duration = segment.segments[-1].end_time - segment.segments[i].start_time
                
                if first_seg_duration >= config.min_split_segment_duration and second_seg_duration >= config.min_split_segment_duration:
                    score = self._calculate_split_score(segment, i, gap, first_seg_duration, second_seg_duration, config)
                    potential_splits.append((i, gap, score, first_seg_duration, second_seg_duration))
        
        # Sort by score and select best splits
        potential_splits.sort(key=lambda x: x[2], reverse=True)
        
        # Select best split points, ensuring they're not too close to each other
        selected_splits = []
        last_split = -1
        
        for split_index, gap, score, first_duration, second_duration in potential_splits:
            if split_index - last_split >= config.min_split_segment_duration:
                selected_splits.append(split_index)
                last_split = split_index
                
                # Limit number of splits per segment
                if len(selected_splits) >= config.max_splits_per_segment:
                    break
        
        return selected_splits
    
    def _split_long_segments_enhanced(
        self,
        segments: List[CombinedSegment],
        config: SegmentCombinationConfig
    ) -> List[CombinedSegment]:
        """Enhanced splitting logic for long segments."""
        final_segments = []
        
        for segment in segments:
            if segment.duration <= config.max_segment_duration:
                final_segments.append(segment)
                continue
            
            # Find best split points
            split_points = self._find_best_split_points(segment, config)
            
            if not split_points:
                # No good split points found, keep as is
                final_segments.append(segment)
                continue
            
            # Split the segment at identified points
            split_groups = []
            current_group = [segment.segments[0]]
            
            for i in range(1, len(segment.segments)):
                if i in split_points:
                    # Finalize current group and start new one
                    if current_group:
                        split_groups.append(current_group)
                    current_group = [segment.segments[i]]
                else:
                    current_group.append(segment.segments[i])
            
            # Add the last group
            if current_group:
                split_groups.append(current_group)
            
            # Convert groups to CombinedSegments
            for group in split_groups:
                if group:
                    final_segments.append(CombinedSegment(group))
        
        return final_segments
    
    def _split_long_segments_second_pass(
        self,
        segments: List[CombinedSegment],
        config: SegmentCombinationConfig
    ) -> List[CombinedSegment]:
        """Second-pass splitting for very long segments using recursive queue."""
        final_segments = []
        
        for segment in segments:
            if segment.duration <= config.max_segment_duration:
                final_segments.append(segment)
                continue
            
            # Apply enhanced splitting
            enhanced_splits = self._split_long_segments_enhanced([segment], config)
            
            # Handle segments that are still too long with recursive splitting
            for enhanced_segment in enhanced_splits:
                if enhanced_segment.duration <= config.max_segment_duration:
                    final_segments.append(enhanced_segment)
                else:
                    # Use a queue for recursive splitting of very long segments
                    recursively_split_segments = []
                    split_queue = [enhanced_segment]
                    
                    while split_queue:
                        current_segment = split_queue.pop(0)
                        
                        if current_segment.duration <= config.max_segment_duration:
                            recursively_split_segments.append(current_segment)
                            continue
                        
                        # Find best split points for this very long segment
                        split_points = self._find_best_split_points(current_segment, config)
                        
                        if not split_points:
                            # No good split points found, keep as is
                            recursively_split_segments.append(current_segment)
                            continue
                        
                        # Split the segment and add parts back to queue if needed
                        split_groups = []
                        current_group = [current_segment.segments[0]]
                        
                        for i in range(1, len(current_segment.segments)):
                            if i in split_points:
                                # Finalize current group and start new one
                                if current_group:
                                    split_groups.append(current_group)
                                current_group = [current_segment.segments[i]]
                            else:
                                current_group.append(current_segment.segments[i])
                        
                        # Add the last group
                        if current_group:
                            split_groups.append(current_group)
                        
                        # Add split groups back to queue for further processing if needed
                        for group in split_groups:
                            if group:
                                new_segment = CombinedSegment(group)
                                if new_segment.duration > config.max_segment_duration:
                                    split_queue.append(new_segment)
                                else:
                                    recursively_split_segments.append(new_segment)
                    
                    # Add all recursively split segments
                    final_segments.extend(recursively_split_segments)
        
        return final_segments
    
    def _initial_combination(
        self,
        segments: List[VADSegment],
        config: SegmentCombinationConfig
    ) -> List[CombinedSegment]:
        """Initial combination of segments based on gap analysis."""
        if not segments:
            return []
        
        # Sort segments chronologically
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        # Handle overlapping segments
        combined_segments = []
        for new_segment in sorted_segments:
            if combined_segments and new_segment.start_time < combined_segments[-1].end_time:
                # Overlapping segment detected - merge with previous
                log_debug(f"Overlapping segment detected: {new_segment.start_time}-{new_segment.end_time} "
                         f"overlaps with {combined_segments[-1].start_time}-{combined_segments[-1].end_time}")
                
                # Merge overlapping segments by extending the end time
                last_segment = combined_segments[-1]
                last_segment.end_time = max(last_segment.end_time, new_segment.end_time)
                last_segment.duration = last_segment.end_time - last_segment.start_time
            else:
                combined_segments.append(new_segment)
        
        # Combine segments based on gap analysis
        final_combined = []
        current_combined = CombinedSegment([combined_segments[0]])
        
        for next_segment in combined_segments[1:]:
            gap = next_segment.start_time - current_combined.end_time
            
            if gap <= config.micro_pause_threshold:
                # Always combine micro-pauses (0.1-0.5s)
                current_combined.segments.append(next_segment)
            elif gap >= config.natural_boundary_threshold:
                # Never combine topic change boundaries (5s+)
                final_combined.append(current_combined)
                current_combined = CombinedSegment([next_segment])
            else:
                # Handle thinking pauses and disfluencies
                if gap <= config.thinking_pause_threshold:
                    # Check for disfluency handling
                    if (next_segment.duration <= config.disfluency_threshold and
                        current_combined.duration >= config.min_disfluency_context):
                        # Combine short disfluency with sufficient context
                        current_combined.segments.append(next_segment)
                    else:
                        # Regular thinking pause - combine with context evaluation
                        current_combined.segments.append(next_segment)
                else:
                    # Keep separate for larger gaps
                    final_combined.append(current_combined)
                    current_combined = CombinedSegment([next_segment])
        
        final_combined.append(current_combined)
        return final_combined
    
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
            debug_lines.append("")
        
        # Get speech timestamps using Silero's built-in function
        try:
            speech_timestamps = self._get_speech_timestamps(  # type: ignore
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
                if debug_file_path:
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
                if debug_file_path:
                    self._write_debug_file(debug_file_path, debug_lines)
            return [(0.0, audio_duration)]
        
        # Convert to VADSegment objects
        segments = [
            VADSegment(ts['start'], ts['end'], ts['end'] - ts['start'])
            for ts in speech_timestamps
        ]
        
        if debug_enabled:
            debug_lines.append("INITIAL SPEECH SEGMENTS (after Silero VAD):")
            for i, seg in enumerate(segments):
                debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s)")
            debug_lines.append("")
        
        log_debug(f"Silero VAD detected {len(segments)} initial speech segments")
        
        # Apply the new combination and splitting approach
        combined_segments = self._initial_combination(segments, self.combination_config)
        
        if debug_enabled:
            debug_lines.append("AFTER INITIAL COMBINATION:")
            for i, seg in enumerate(combined_segments):
                debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s) [{seg.get_segment_count()} original segments]")
            debug_lines.append("")
        
        # Apply enhanced splitting for long segments
        final_segments = self._split_long_segments_enhanced(combined_segments, self.combination_config)
        
        if debug_enabled:
            debug_lines.append("AFTER ENHANCED SPLITTING:")
            for i, seg in enumerate(final_segments):
                debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s) [{seg.get_segment_count()} original segments]")
            debug_lines.append("")
        
        # Check for remaining long segments and perform second pass if needed
        long_segments = [seg for seg in final_segments if seg.duration > self.combination_config.max_segment_duration]
        if long_segments:
            log_debug(f"Found {len(long_segments)} segments still exceeding max duration, applying second-pass splitting")
            final_segments = self._split_long_segments_second_pass(final_segments, self.combination_config)
            
            if debug_enabled:
                debug_lines.append("AFTER SECOND-PASS SPLITTING:")
                for i, seg in enumerate(final_segments):
                    debug_lines.append(f"  Segment {i+1}: {seg.start:.3f} - {seg.end:.3f} ({seg.duration:.3f}s) [{seg.get_segment_count()} original segments]")
                debug_lines.append("")
        
        # Convert to time tuples
        time_segments = [seg.to_tuple() for seg in final_segments]
        
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
            if debug_file_path:
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
        return self.segment_audio(waveform, int(sr), debug_file_path)


def create_silero_vad_segmenter(
    max_segment_duration: float = 45.0,
    device: Optional[str] = None,
    custom_settings: bool = True,
    **kwargs
) -> SileroVADSegmenter:
    """
    Factory function to create a SileroVADSegmenter with common defaults.
    
    Args:
        max_segment_duration: Maximum segment duration for ASR
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
        device=device,
        **kwargs
    )
