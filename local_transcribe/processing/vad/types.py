#!/usr/bin/env python3
"""
Unified data structures for VAD processing.

This module provides the core data classes for representing VAD segments,
blocks (turns), ASR chunks, and configuration objects used across the
VAD-driven pipelines.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class VADSegment:
    """A single VAD-detected speech segment.
    
    Represents a contiguous region of detected speech from Silero VAD
    for a single speaker.
    """
    segment_id: int          # Original segment number from VAD
    speaker_id: str          # Speaker identifier (e.g., "Interviewer", "Participant")
    start_s: float           # Start time in seconds (absolute timeline)
    end_s: float             # End time in seconds (absolute timeline)
    
    @property
    def duration_s(self) -> float:
        """Duration of this segment in seconds."""
        return round(self.end_s - self.start_s, 3)
    
    @property
    def start_time(self) -> float:
        """Alias for start_s for compatibility."""
        return self.start_s
    
    @property
    def end_time(self) -> float:
        """Alias for end_s for compatibility."""
        return self.end_s
    
    @property
    def duration(self) -> float:
        """Alias for duration_s for compatibility."""
        return self.duration_s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "segment_id": self.segment_id,
            "speaker_id": self.speaker_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VADSegment':
        """Create from dictionary."""
        return cls(
            segment_id=data["segment_id"],
            speaker_id=data["speaker_id"],
            start_s=data["start_s"],
            end_s=data["end_s"],
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (start_sec, end_sec) tuple."""
        return (self.start_s, self.end_s)


@dataclass
class CombinedSegment:
    """A combined segment containing multiple VAD segments, ready for ASR.
    
    Represents multiple nearby VAD segments that have been combined
    into a single chunk for ASR processing.
    """
    segments: List[VADSegment]
    
    def __post_init__(self):
        if not self.segments:
            raise ValueError("CombinedSegment must contain at least one VADSegment")
        
        # Calculate combined properties
        self._start_s = min(seg.start_s for seg in self.segments)
        self._end_s = max(seg.end_s for seg in self.segments)
    
    @property
    def start_s(self) -> float:
        """Start time of the combined segment."""
        return self._start_s
    
    @property
    def end_s(self) -> float:
        """End time of the combined segment."""
        return self._end_s
    
    @property
    def duration_s(self) -> float:
        """Duration of the combined segment."""
        return round(self._end_s - self._start_s, 3)
    
    # Aliases for compatibility
    @property
    def start_time(self) -> float:
        return self._start_s
    
    @property
    def end_time(self) -> float:
        return self._end_s
    
    @property
    def duration(self) -> float:
        return self.duration_s
    
    @property
    def start(self) -> float:
        return self._start_s
    
    @property
    def end(self) -> float:
        return self._end_s
    
    def get_segment_count(self) -> int:
        """Get the number of original segments in this combined segment."""
        return len(self.segments)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (start_sec, end_sec) tuple."""
        return (self._start_s, self._end_s)


@dataclass 
class VADBlock:
    """A merged block of contiguous VAD segments for a speaker (a turn).
    
    Represents a speaking turn created by merging nearby VAD segments
    from a single speaker. May contain interjection flags and overlap
    information when interleaved with other speakers.
    
    The source_segments field stores the actual VADSegment objects that
    formed this block, enabling splitting at natural pause boundaries
    when the block exceeds ASR chunk duration limits.
    """
    block_id: int
    speaker_id: str
    start_s: float
    end_s: float
    source_segments: List['VADSegment']  # Original VAD segments that formed this block
    is_interjection: bool = False
    overlap_with: Optional[List[int]] = None  # block_ids this overlaps with
    text: str = ""  # Transcript text (filled after ASR)
    
    @property
    def duration_s(self) -> float:
        """Duration of this block in seconds."""
        return round(self.end_s - self.start_s, 3)
    
    @property
    def source_segment_ids(self) -> List[int]:
        """Get IDs of source segments (for backward compatibility)."""
        return [seg.segment_id for seg in self.source_segments]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "block_id": self.block_id,
            "speaker_id": self.speaker_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "source_segment_ids": self.source_segment_ids,
            "source_segments": [seg.to_dict() for seg in self.source_segments],
            "is_interjection": self.is_interjection,
            "overlap_with": self.overlap_with,
            "text": self.text,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VADBlock':
        """Create from dictionary."""
        # Handle both old format (source_segment_ids) and new format (source_segments)
        if "source_segments" in data:
            source_segments = [VADSegment.from_dict(seg) for seg in data["source_segments"]]
        else:
            # Legacy format: reconstruct minimal VADSegments from IDs
            # Note: This loses timing information, but maintains compatibility
            source_segments = [
                VADSegment(
                    segment_id=seg_id,
                    speaker_id=data["speaker_id"],
                    start_s=data["start_s"],  # Approximate
                    end_s=data["end_s"],  # Approximate
                )
                for seg_id in data.get("source_segment_ids", [])
            ]
        
        return cls(
            block_id=data["block_id"],
            speaker_id=data["speaker_id"],
            start_s=data["start_s"],
            end_s=data["end_s"],
            source_segments=source_segments,
            is_interjection=data.get("is_interjection", False),
            overlap_with=data.get("overlap_with"),
            text=data.get("text", ""),
        )


@dataclass
class VADBlockBuilderConfig:
    """Configuration for merging VAD segments into blocks.
    
    Controls how nearby VAD segments are merged and how interjections
    and overlaps are detected.
    """
    merge_gap_threshold_ms: int = 900   # Merge segments with gap < this
    interjection_max_duration_ms: int = 2000  # Max duration to classify as interjection
    interjection_window_ms: int = 500   # Window after block start for interjection detection
    overlap_threshold_ms: int = 100     # Min overlap to consider segments overlapping
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "merge_gap_threshold_ms": self.merge_gap_threshold_ms,
            "interjection_max_duration_ms": self.interjection_max_duration_ms,
            "interjection_window_ms": self.interjection_window_ms,
            "overlap_threshold_ms": self.overlap_threshold_ms,
        }


@dataclass
class ASRChunk:
    """An audio chunk prepared for ASR.
    
    Represents a segment of audio extracted from a VAD block that
    is ready for transcription. Used for chunking long blocks into
    manageable pieces with overlap for stitching.
    """
    chunk_id: int
    speaker_id: str
    source_block_id: int
    start_s: float              # Absolute start time
    end_s: float                # Absolute end time
    audio_segment: np.ndarray   # Audio samples (16kHz mono)
    overlap_start_s: float      # Start of overlap region (for stitching)
    split_method: str = "vad_boundary"  # How this chunk was split from source
    
    @property
    def duration_s(self) -> float:
        """Duration of this chunk in seconds."""
        return round(self.end_s - self.start_s, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (without audio data)."""
        return {
            "chunk_id": self.chunk_id,
            "speaker_id": self.speaker_id,
            "source_block_id": self.source_block_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "overlap_start_s": self.overlap_start_s,
            "split_method": self.split_method,
        }


@dataclass
class SegmentCombinationConfig:
    """Configuration for segment combination/splitting.
    
    Controls how raw VAD segments are combined into ASR-ready chunks
    and how long segments are split at natural boundaries.
    """
    # Core combination thresholds
    micro_pause_threshold: float = 0.5      # Gaps < this always combined
    thinking_pause_threshold: float = 2.0   # Gaps in this range use context
    natural_boundary_threshold: float = 5.0 # Gaps >= this never combined
    
    # Duration constraints
    max_segment_duration: float = 30.0  # From transcriber capability
    min_segment_duration: float = 0.5
    
    # Disfluency handling
    disfluency_threshold: float = 0.5      # Segments < this are disfluencies
    min_disfluency_context: float = 1.0    # Min context to preserve disfluencies
    
    # Splitting parameters - tiered gap thresholds
    min_gap_for_primary_split: float = 1.0
    min_gap_for_secondary_split: float = 0.7
    min_gap_for_tertiary_split: float = 0.6
    
    # Additional splitting parameters
    max_splits_per_segment: int = 3
    min_split_segment_duration: float = 5.0
    preferred_split_gap_ratio: float = 0.3
    lookahead_segments: int = 5
