#!/usr/bin/env python3
"""
Data structures for VAD-driven split-audio pipeline.

This module defines the core data classes for representing VAD segments,
blocks (turns), and ASR chunks used in the VAD-first pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class VADSegment:
    """
    A single VAD-detected speech segment.
    
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


@dataclass 
class VADBlock:
    """
    A merged block of contiguous VAD segments for a speaker (a turn).
    
    Represents a speaking turn created by merging nearby VAD segments
    from a single speaker. May contain interjection flags and overlap
    information when interleaved with other speakers.
    """
    block_id: int
    speaker_id: str
    start_s: float
    end_s: float
    source_segment_ids: List[int]  # Original VAD segment IDs that formed this block
    is_interjection: bool = False
    overlap_with: Optional[List[int]] = None  # block_ids this overlaps with
    text: str = ""  # Transcript text (filled after ASR)
    
    @property
    def duration_s(self) -> float:
        """Duration of this block in seconds."""
        return round(self.end_s - self.start_s, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "block_id": self.block_id,
            "speaker_id": self.speaker_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "source_segment_ids": self.source_segment_ids,
            "is_interjection": self.is_interjection,
            "overlap_with": self.overlap_with,
            "text": self.text,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VADBlock':
        """Create from dictionary."""
        return cls(
            block_id=data["block_id"],
            speaker_id=data["speaker_id"],
            start_s=data["start_s"],
            end_s=data["end_s"],
            source_segment_ids=data["source_segment_ids"],
            is_interjection=data.get("is_interjection", False),
            overlap_with=data.get("overlap_with"),
            text=data.get("text", ""),
        )


@dataclass
class VADBlockBuilderConfig:
    """
    Configuration for merging VAD segments into blocks.
    
    Controls how nearby VAD segments are merged and how interjections
    and overlaps are detected.
    """
    merge_gap_threshold_ms: int = 600   # Merge segments with gap < this
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
    """
    An audio chunk prepared for ASR.
    
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
        }
