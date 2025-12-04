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
from local_transcribe.lib.environment import *  # Load environment variables


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
    
    # Default model used by this segmenter
    DEFAULT_MODEL = "pyannote/segmentation-3.0"
    
    def __init__(
        self,
        onset_threshold: float = 0.767,
        offset_threshold: float = 0.377,
        min_duration_on: float = 0.136,
        min_duration_off: float = 0.067,
        max_segment_duration: float = 45.0,
        merge_threshold: float = 45.0,
        device: Optional[str] = None,
        models_dir: Optional[pathlib.Path] = None
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
            models_dir: Directory where models are cached
        """
        self.logger = get_logger()
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_segment_duration = max_segment_duration
        self.merge_threshold = merge_threshold
        self.device = device
        self.models_dir = models_dir
        
        # Model state
        self._model = None
        self._inference = None
    
    def _get_cache_dir(self) -> pathlib.Path:
        """Get the cache directory for VAD models."""
        if self.models_dir is None:
            raise ValueError("models_dir must be provided to VADSegmenter")
        return self.models_dir / "vad" / "pyannote"
    
    def _model_name_to_hf_format(self, model: str) -> str:
        """Convert model name to HuggingFace cache directory format."""
        return model.replace("/", "--")
    
    def _load_model(self):
        """Load the pyannote VAD/segmentation model."""
        if self._model is not None:
            return
        
        log_progress("Loading pyannote VAD model...")
        
        # Get token from environment
        token = os.getenv("HF_TOKEN", "")
        
        # Find model snapshot directory
        cache_dir = self._get_cache_dir()
        hf_model_name = self._model_name_to_hf_format(self.DEFAULT_MODEL)
        model_dir = cache_dir / f"models--{hf_model_name}"
        
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
            raise FileNotFoundError(f"VAD model not found at {model_dir}. Please ensure models are downloaded first.")
        
        snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if not snapshot_dirs:
            raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
        
        latest_snapshot_dir = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
        
        try:
            from pyannote.audio import Model
            
            # Load model from the specific snapshot directory
            self._model = Model.from_pretrained(
                str(latest_snapshot_dir),
                use_auth_token=token if token else None
            )
            
            if self.device:
                device = torch.device(self.device)
                self._model = self._model.to(device)
            
            # Create inference object for processing
            device = torch.device(self.device) if self.device else torch.device("cpu")
            self._inference = Inference(
                self._model,
                window="whole",
                device=device
            )
            
            log_completion("Pyannote VAD model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load pyannote VAD model: {e}")
            raise
    
    def preload_models(self) -> None:
        """Preload the VAD model to cache."""
        import sys
        
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        # Force reload of huggingface_hub modules to pick up new environment
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
        
        # Get token from environment once
        token = os.getenv("HF_TOKEN", "")
        
        try:
            from huggingface_hub import snapshot_download
            
            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(self.DEFAULT_MODEL, cache_dir=str(cache_dir), token=token if token else None)
            log_completion(f"VAD model {self.DEFAULT_MODEL} downloaded successfully.")
            
        except Exception as e:
            raise Exception(f"Failed to download VAD model {self.DEFAULT_MODEL}: {e}")
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode
    
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
        
        # For segmentation-3.0, use raw model inference and apply softmax
        with torch.no_grad():
            logits = self._model(waveform_tensor)
            # Apply softmax to get probabilities
            import torch.nn.functional as F
            probabilities = F.softmax(logits, dim=-1)
            
            # For segmentation-3.0: class 0 is non-speech
            # Speech probability = 1 - non-speech probability
            speech_scores = 1.0 - probabilities[0, :, 0].cpu().numpy()
            
            # Frame hop: model processes 10s audio into ~56 frames, so ~10/56 ≈ 0.179s per frame
            # But let's calculate based on audio duration
            audio_duration = len(waveform) / sample_rate
            num_frames = speech_scores.shape[0]
            frame_hop = audio_duration / num_frames
        
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
