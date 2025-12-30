#!/usr/bin/env python3
"""
Core data types and configuration for chunk stitching.

This module provides the fundamental data structures and configuration
used throughout the chunk stitching system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


@dataclass
class ChunkStitcherConfig:
    """Configuration for the chunk stitcher."""
    
    # Overlap detection thresholds
    min_overlap_ratio: float = 0.6
    """Minimum ratio of overlapping words to consider a valid overlap."""
    
    similarity_threshold: float = 0.7
    """Threshold for word similarity using SequenceMatcher."""
    
    # Sequence alignment parameters
    sequence_alignment_window: int = 20
    """Number of words to compare from each chunk for sequence alignment."""
    
    min_sequence_match_length: int = 3
    """Minimum number of matching words required for sequence alignment."""
    
    # Debug options
    skip_single_chunk_debug: bool = True
    """If True, skip debug output when there's only one chunk (no stitching needed)."""
    
    use_timestamped_debug_dir: bool = True
    """If True, create a timestamped subdirectory under intermediate_dir for debug files."""

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ChunkStitcherConfig":
        """Create a config from keyword arguments, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class OverlapResult:
    """Result from overlap detection."""
    
    overlap_start: int = 0
    """Index in chunk1 where overlap starts."""
    
    overlap_length: int = 0
    """Number of overlapping words in chunk1."""
    
    words_to_skip_in_chunk2: int = 0
    """Number of words to skip from the start of chunk2."""
    
    method: str = "none"
    """Method used to detect the overlap."""
    
    @property
    def found(self) -> bool:
        """Whether an overlap was found."""
        return self.overlap_length > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overlap_found': self.found,
            'overlap_start': self.overlap_start,
            'overlap_length': self.overlap_length,
            'words_to_skip_in_chunk2': self.words_to_skip_in_chunk2,
            'method': self.method,
        }


@dataclass
class ChunkInfo:
    """Information about a chunk for debugging/audit purposes."""
    
    chunk_id: Optional[Any] = None
    word_count: int = 0
    last_10_words: List[str] = field(default_factory=list)
    first_10_words: List[str] = field(default_factory=list)
    time_range: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            'word_count': self.word_count,
        }
        if self.chunk_id is not None:
            result['chunk_id'] = self.chunk_id
        if self.last_10_words:
            result['last_10_words'] = self.last_10_words
        if self.first_10_words:
            result['first_10_words'] = self.first_10_words
        if self.time_range:
            result['time_range'] = self.time_range
        return result


@dataclass
class StitchStepInfo:
    """Information about a single stitching step."""
    
    step_num: int
    chunk1_info: ChunkInfo
    chunk2_info: ChunkInfo
    overlap_info: OverlapResult
    result_word_count: int
    result_time_range: Optional[Dict[str, float]] = None
    words_from_chunk1: int = 0
    words_from_chunk2: int = 0


# Type aliases for clarity
WordDict = Dict[str, Any]
"""A word represented as a dictionary with 'text', 'start', 'end' keys."""

WordType = Union[str, WordDict]
"""A word can be either a string or a dictionary with timestamps."""

WordList = List[WordType]
"""A list of words (either strings or dicts)."""
