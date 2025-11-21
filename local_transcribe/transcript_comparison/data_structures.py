"""
Core data structures for transcript comparison.

This module defines the data structures used throughout the transcript comparison system,
including representations of transcripts, alignment results, and comparison metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class TextSegment:
    """Represents a segment of text with timing and metadata information."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    segment_id: str = ""
    parent_id: Optional[str] = None  # For hierarchical structure
    confidence: Optional[float] = None
    features: Optional[Dict[str, Any]] = None  # Stores CSV line number and other metadata
    
    def __post_init__(self):
        """Initialize features dict if None."""
        if self.features is None:
            self.features = {}


@dataclass
class TranscriptMetadata:
    """Metadata about a transcript."""
    source: str  # Origin of transcript (e.g., "whisper", "human", "csv")
    creation_date: datetime
    duration: float
    word_count: int
    language: str = "en"
    segmentation_type: str = "word"  # "word", "phrase", "turn"
    original_format: Optional[str] = None  # "csv", "json", etc.


@dataclass
class EnhancedTranscript:
    """Enhanced transcript representation with segments and metadata."""
    segments: List[TextSegment]  # Can be words, phrases, or turns
    metadata: TranscriptMetadata
    features: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize features dict if None."""
        if self.features is None:
            self.features = {}


@dataclass
class AlignedPair:
    """Represents a pair of aligned segments from reference and hypothesis transcripts."""
    reference_segment: Optional[TextSegment]
    hypothesis_segment: Optional[TextSegment]
    similarity_score: float
    alignment_type: str  # "exact", "semantic", "partial", "insertion", "deletion"
    timing_difference: float
    confidence: float


@dataclass
class AlignmentResult:
    """Result of transcript alignment process."""
    aligned_pairs: List[AlignedPair]
    unaligned_reference: List[TextSegment]
    unaligned_hypothesis: List[TextSegment]
    alignment_score: float


@dataclass
class Difference:
    """Represents a difference between reference and hypothesis transcripts."""
    type: str  # "substitution", "insertion", "deletion", "timing"
    severity: str  # "minor", "moderate", "major"
    reference_segment: Optional[TextSegment]
    hypothesis_segment: Optional[TextSegment]
    description: str
    context: List[TextSegment]  # Surrounding segments for context
    suggestions: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize suggestions list if None."""
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class QualityMetrics:
    """Quality metrics for transcript comparison."""
    word_error_rate: float
    match_error_rate: float
    semantic_similarity: float
    timing_accuracy: float
    fluency_score: float
    confidence_score: float
    overall_quality: float  # Weighted combination of above metrics


@dataclass
class ComparisonSummary:
    """Summary statistics for transcript comparison."""
    total_segments: int
    matched_segments: int
    substituted_segments: int
    inserted_segments: int
    deleted_segments: int
    timing_differences: int
    overall_similarity: float


@dataclass
class ComparisonResult:
    """Complete result of transcript comparison."""
    alignment_result: AlignmentResult
    quality_metrics: QualityMetrics
    differences: List[Difference]
    summary: ComparisonSummary


@dataclass
class ComparisonConfig:
    """Configuration for transcript comparison."""
    # Similarity thresholds
    semantic_threshold: float = 0.8
    fuzzy_threshold: float = 0.7
    phonetic_threshold: float = 0.6
    
    # Similarity weights
    semantic_weight: float = 0.5
    fuzzy_weight: float = 0.3
    phonetic_weight: float = 0.1
    contextual_weight: float = 0.1
    
    # Timing parameters
    timing_tolerance: float = 2.0  # seconds
    max_timing_difference: float = 10.0  # seconds
    
    # Segmentation parameters
    min_segment_length: float = 0.1  # seconds
    max_segment_length: float = 30.0  # seconds
    
    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Output options
    include_context: bool = True
    context_window_size: int = 2  # segments on each side
    max_differences: int = 100
    
    # Quality metrics weights
    wer_weight: float = 0.3
    semantic_weight: float = 0.4
    timing_weight: float = 0.2
    fluency_weight: float = 0.1
    
    # CSV format specific options
    estimate_timing_from_line_numbers: bool = True
    average_words_per_second: float = 2.5
    
    # Phrase context options
    phrase_length: int = 5  # Number of words to group for context
    
    # Alignment parameters
    gap_penalty: float = 0.5  # Penalty for gaps in sequence alignment