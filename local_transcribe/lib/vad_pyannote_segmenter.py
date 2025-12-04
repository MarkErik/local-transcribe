#!/usr/bin/env python3
"""
VAD-based audio segmentation using pyannote.audio.

This module implements intelligent audio segmentation based on Voice Activity Detection (VAD)
using pyannote's neural VAD model. The segmentation strategy uses:

1. Neural VAD scoring: Frame-wise speech probability scores
2. Binarization with hysteresis: Convert scores to speech/non-speech segments
3. Cut: Split segments exceeding max duration at low-VAD points
4. Merge: Combine short adjacent segments up to merge threshold

This approach ensures chunks start/end at natural speech boundaries rather than arbitrary time points.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchaudio
from local_transcribe.lib.program_logger import get_logger, log_progress, log_debug, log_completion


@dataclass
class VADSegment:
    """Represents a speech segment with frame indices."""
    start_idx: int  # inclusive frame index
    end_idx: int    # exclusive frame index
    
    @property
    def length_frames(self) -> int:
        return self.end_idx - self.start_idx
    
    def duration(self, frame_hop: float) -> float:
        """Get duration in seconds."""
        return self.length_frames * frame_hop
    
    def to_time_tuple(self, frame_hop: float) -> Tuple[float, float]:
        """Convert to (start_sec, end_sec) tuple."""
        return (self.start_idx * frame_hop, self.end_idx * frame_hop)


class VADSegmenter:
    """
    Voice Activity Detection based audio segmenter.
    
    Uses pyannote's neural VAD to identify speech regions and segments audio
    intelligently at natural speech boundaries.
    """
    
    def __init__(
        self,
        onset_threshold: float = 0.767,
        offset_threshold: float = 0.377,
        min_duration_on: float = 0.136,
        min_duration_off: float = 0.067,
        max_segment_duration: float = 45.0,
        merge_threshold: float = 45.0,
        device: Optional[str] = None
    ):
        """
        Initialize the VAD segmenter.
        
        Args:
            onset_threshold: Threshold to start speech (higher = stricter)
            offset_threshold: Threshold to end speech (lower = stickier)
            min_duration_on: Minimum speech segment duration in seconds
            min_duration_off: Minimum silence duration in seconds
            max_segment_duration: Maximum segment duration for ASR in seconds
            merge_threshold: Maximum duration for merged segments in seconds
            device: Device to run VAD model on ('cpu', 'cuda', 'mps')
        """
        self.logger = get_logger()
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_segment_duration = max_segment_duration
        self.merge_threshold = merge_threshold
        self.device = device
        
        # Model state
        self._model = None
        self._inference = None
    
    def _load_model(self):
        """Load the pyannote VAD/segmentation model."""
        if self._model is not None:
            return
        
        log_progress("Loading pyannote VAD model...")
        
        try:
            from pyannote.audio import Model, Inference
            import os
            
            # Use segmentation model which provides speech scores
            model_name = "pyannote/segmentation"
            token = os.getenv("HF_TOKEN")
            
            self._model = Model.from_pretrained(model_name, use_auth_token=token)
            
            if self.device:
                self._model = self._model.to(self.device)
            
            # Create inference object for processing
            self._inference = Inference(
                self._model,
                window="whole",
                device=self.device if self.device else "cpu"
            )
            
            log_completion("Pyannote VAD model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load pyannote VAD model: {e}")
            raise
    
    def get_vad_scores(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[np.ndarray, float]:
        """
        Get frame-wise VAD speech probability scores.
        
        Args:
            waveform: Audio waveform as numpy array (mono)
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (speech_scores, frame_hop) where:
            - speech_scores: numpy array of shape (num_frames,) with values in [0,1]
            - frame_hop: time in seconds between frames
        """
        self._load_model()
        
        # Ensure waveform is the right shape for pyannote (channels, samples)
        if waveform.ndim == 1:
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
        else:
            waveform_tensor = torch.from_numpy(waveform).float()
        
        # Run inference
        scores = self._inference({"waveform": waveform_tensor, "sample_rate": sample_rate})
        
        # scores is a SlidingWindowFeature with shape (num_frames, num_classes)
        # For segmentation model, typically:
        # - Index 0: non-speech
        # - Index 1: speech (or multiple speaker indices)
        # We want the max over speech classes (indices 1+) or just sum them
        
        score_data = scores.data
        frame_hop = scores.sliding_window.step
        
        # Get speech probability (sum of all speaker classes, or max)
        # The segmentation model outputs probabilities for each speaker slot
        if score_data.shape[1] > 1:
            # Sum all non-silence classes (usually indices 1, 2, 3 for speakers)
            # Or take max to get "any speech" probability
            speech_scores = np.max(score_data[:, :], axis=1)  # Max across all classes
        else:
            speech_scores = score_data[:, 0]
        
        log_debug(f"VAD scores computed: {len(speech_scores)} frames at {frame_hop:.4f}s hop")
        
        return speech_scores, frame_hop
    
    def _seconds_to_frames(self, duration_s: float, frame_hop: float) -> int:
        """Convert duration in seconds to frame count."""
        return int(round(duration_s / frame_hop))
    
    def binarize(
        self,
        scores: np.ndarray,
        frame_hop: float
    ) -> List[VADSegment]:
        """
        Convert frame-wise scores to speech segments using hysteresis.
        
        Args:
            scores: Frame-wise speech probability scores
            frame_hop: Time in seconds between frames
            
        Returns:
            List of VADSegment objects representing speech regions
        """
        n_frames = len(scores)
        min_on_frames = self._seconds_to_frames(self.min_duration_on, frame_hop)
        min_off_frames = self._seconds_to_frames(self.min_duration_off, frame_hop)
        
        segments: List[VADSegment] = []
        is_speech = False
        seg_start = None
        last_change_idx = 0
        
        for i in range(n_frames):
            p = scores[i]
            
            if not is_speech:
                # Currently non-speech; check if we should start speech
                if p >= self.onset_threshold:
                    # Ensure we've had at least min_off_frames of silence
                    if i - last_change_idx >= min_off_frames or last_change_idx == 0:
                        is_speech = True
                        seg_start = i
                        last_change_idx = i
            else:
                # Currently speech; check if we should end speech
                if p < self.offset_threshold:
                    # Ensure the speech segment is long enough
                    if i - seg_start >= min_on_frames:
                        segments.append(VADSegment(seg_start, i))
                        is_speech = False
                        seg_start = None
                        last_change_idx = i
                    # If segment too short, keep looking for offset
        
        # If still in speech at end of file
        if is_speech and seg_start is not None:
            if n_frames - seg_start >= min_on_frames:
                segments.append(VADSegment(seg_start, n_frames))
        
        log_debug(f"Binarization produced {len(segments)} raw speech segments")
        
        return segments
    
    def _cut_long_segment(
        self,
        segment: VADSegment,
        scores: np.ndarray,
        frame_hop: float
    ) -> List[VADSegment]:
        """
        Recursively cut a segment so no sub-segment exceeds max_segment_duration.
        
        Cuts are placed at the minimum VAD score within [max/2, max] from start,
        ensuring cuts happen at the most silence-like points.
        """
        max_frames = self._seconds_to_frames(self.max_segment_duration, frame_hop)
        
        # Base case: already short enough
        if segment.length_frames <= max_frames:
            return [segment]
        
        start = segment.start_idx
        end = segment.end_idx
        
        # Define search window for cut (in frames)
        half_max_frames = max_frames // 2
        window_start = start + half_max_frames
        window_end = min(start + max_frames, end)
        
        if window_start >= window_end:
            # Edge case: can't find a proper window; just split at max_frames
            cut_idx = start + max_frames
        else:
            # Find the frame with minimum VAD score in the window
            window_scores = scores[window_start:window_end]
            rel_min_idx = int(np.argmin(window_scores))
            cut_idx = window_start + rel_min_idx
        
        # Split into [start, cut_idx) and [cut_idx, end)
        left = VADSegment(start, cut_idx)
        right = VADSegment(cut_idx, end)
        
        # Recursively cut both sides
        left_cut = self._cut_long_segment(left, scores, frame_hop)
        right_cut = self._cut_long_segment(right, scores, frame_hop)
        
        return left_cut + right_cut
    
    def cut_long_segments(
        self,
        segments: List[VADSegment],
        scores: np.ndarray,
        frame_hop: float
    ) -> List[VADSegment]:
        """
        Apply cutting to all segments that exceed max_segment_duration.
        """
        result: List[VADSegment] = []
        for seg in segments:
            result.extend(self._cut_long_segment(seg, scores, frame_hop))
        
        cut_count = len(result) - len(segments)
        if cut_count > 0:
            log_debug(f"Cut phase: {len(segments)} -> {len(result)} segments ({cut_count} cuts made)")
        
        return result
    
    def merge_segments(
        self,
        segments: List[VADSegment],
        frame_hop: float
    ) -> List[VADSegment]:
        """
        Merge adjacent segments if their combined duration is below merge_threshold.
        
        Assumes segments are sorted by start_idx and non-overlapping.
        """
        if not segments:
            return []
        
        merge_frames = self._seconds_to_frames(self.merge_threshold, frame_hop)
        
        merged: List[VADSegment] = []
        current = segments[0]
        
        for nxt in segments[1:]:
            # Combined duration including the gap between segments
            combined_start = current.start_idx
            combined_end = nxt.end_idx
            combined_length = combined_end - combined_start
            
            if combined_length <= merge_frames:
                # Merge: extend current to include next
                current = VADSegment(combined_start, combined_end)
            else:
                merged.append(current)
                current = nxt
        
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
        Full VAD segmentation pipeline: score -> binarize -> cut -> merge.
        
        Args:
            waveform: Audio waveform as numpy array (mono)
            sample_rate: Sample rate of the audio
            
        Returns:
            List of (start_sec, end_sec) tuples representing speech segments
        """
        log_progress("Running VAD segmentation pipeline...")
        
        # Step 1: Get VAD scores
        scores, frame_hop = self.get_vad_scores(waveform, sample_rate)
        
        # Step 2: Binarization
        segments = self.binarize(scores, frame_hop)
        
        if not segments:
            # No speech detected - return the whole audio as one segment
            duration = len(waveform) / sample_rate
            log_debug("No speech segments detected by VAD, using full audio")
            return [(0.0, duration)]
        
        # Step 3: Cut long segments
        segments = self.cut_long_segments(segments, scores, frame_hop)
        
        # Step 4: Merge short adjacent segments
        segments = self.merge_segments(segments, frame_hop)
        
        # Convert to time tuples
        time_segments = [seg.to_time_tuple(frame_hop) for seg in segments]
        
        # Log summary
        total_speech = sum(end - start for start, end in time_segments)
        audio_duration = len(waveform) / sample_rate
        log_completion(
            f"VAD segmentation complete: {len(time_segments)} segments, "
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


def create_vad_segmenter(
    max_segment_duration: float = 45.0,
    merge_threshold: float = 45.0,
    device: Optional[str] = None,
    **kwargs
) -> VADSegmenter:
    """
    Factory function to create a VADSegmenter with common defaults.
    
    Args:
        max_segment_duration: Maximum segment duration for ASR
        merge_threshold: Maximum duration for merged segments
        device: Device for VAD model
        **kwargs: Additional parameters passed to VADSegmenter
        
    Returns:
        Configured VADSegmenter instance
    """
    return VADSegmenter(
        max_segment_duration=max_segment_duration,
        merge_threshold=merge_threshold,
        device=device,
        **kwargs
    )
