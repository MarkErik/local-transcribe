"""
Transcript Comparison Module

This module provides sophisticated transcript comparison capabilities that go beyond
simple word-by-word matching to include semantic similarity, timing alignment,
and detailed quality metrics.
"""

from .data_structures import (
    EnhancedTranscript,
    TextSegment,
    TranscriptMetadata,
    AlignmentResult,
    AlignedPair,
    ComparisonResult,
    Difference,
    QualityMetrics,
    ComparisonSummary,
    ComparisonConfig
)
from .engine import TranscriptComparisonEngine
from .csv_parser import CSVTranscriptParser, LineNumberAlignmentStrategy

__all__ = [
    "EnhancedTranscript",
    "TextSegment",
    "TranscriptMetadata",
    "AlignmentResult",
    "AlignedPair",
    "ComparisonResult",
    "Difference",
    "QualityMetrics",
    "ComparisonSummary",
    "ComparisonConfig",
    "TranscriptComparisonEngine",
    "CSVTranscriptParser",
    "LineNumberAlignmentStrategy"
]