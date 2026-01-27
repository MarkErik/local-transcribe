#!/usr/bin/env python3
"""
Core utilities and data structures for LLM-based de-identification.

This module contains shared constants, data classes, and utility functions
used across all de-identification modules.
"""

import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DeIdentificationConfig:
    """Configuration for de-identification processing."""
    
    # Chunking settings
    chunk_size: int = 400              # Words per chunk
    overlap_size: int = 60             # Words of overlap between chunks
    min_final_chunk: int = 200         # Min words for final chunk
    
    # LLM settings
    llm_url: str = "http://0.0.0.0:8080"
    llm_timeout: int = 300             # Seconds
    temperature: float = 1.0           # Temperature for LLM
    max_retries: int = 3               # Number of retries on validation failure
    temperature_decay: float = 0.15    # Reduce temperature on each retry
    
    # Response parsing (auto-detected if None)
    parse_harmony: Optional[bool] = None  # None = auto-detect
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            'chunk_size': self.chunk_size,
            'overlap_size': self.overlap_size,
            'min_final_chunk': self.min_final_chunk,
            'llm_url': self.llm_url,
            'llm_timeout': self.llm_timeout,
            'temperature': self.temperature,
            'max_retries': self.max_retries,
            'temperature_decay': self.temperature_decay,
            'parse_harmony': self.parse_harmony,
        }


# Default configuration instance
DEFAULT_CONFIG = DeIdentificationConfig()


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WordReplacement:
    """Record of a single word replacement."""
    word_index: int
    original: str
    timestamp: Optional[float] = None
    speaker: Optional[str] = None
    matched_from_list: Optional[str] = None  # For second pass
    pass_number: int = 1  # 1 or 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'word_index': self.word_index,
            'original': self.original,
            'timestamp': self.timestamp,
            'speaker': self.speaker,
            'pass': self.pass_number,
        }
        if self.matched_from_list:
            result['matched_from_list'] = self.matched_from_list
        return result


@dataclass
class DiscoveredName:
    """Represents a name discovered during de-identification."""
    name: str
    source_speaker: Optional[str] = None
    occurrences: int = 1
    
    def __hash__(self):
        return hash(normalize_text_for_comparison(self.name).lower())
    
    def __eq__(self, other):
        if isinstance(other, DiscoveredName):
            return words_equivalent(self.name, other.name)
        return False


@dataclass
class ValidationResult:
    """Result of validating LLM output."""
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'reason': self.reason,
            'details': self.details,
        }


@dataclass
class ChunkProcessingResult:
    """Result of processing a single chunk."""
    processed_text: str
    response_time_ms: float
    validation: ValidationResult
    raw_response: Optional[str] = None
    attempt_logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Chunk:
    """
    A chunk of text/segments for processing.
    
    Note: segments and words are mutually exclusive:
    - segments is populated when processing WordSegment lists
    - words is populated when processing plain text
    """
    text: str
    start_idx: int
    end_idx: int
    segments: List[Any] = field(default_factory=list)  # WordSegment list if available
    words: List[str] = field(default_factory=list)     # Plain words if text-only mode


@dataclass
class DeIdentificationResult:
    """Complete result from de-identification (both passes)."""
    segments: List[Any]  # WordSegment
    first_pass_replacements: List[WordReplacement] = field(default_factory=list)
    second_pass_replacements: List[WordReplacement] = field(default_factory=list)
    discovered_names: Set[str] = field(default_factory=set)
    
    @property
    def all_replacements(self) -> List[WordReplacement]:
        """Get all replacements from both passes."""
        return self.first_pass_replacements + self.second_pass_replacements
    
    @property
    def total_replacements(self) -> int:
        """Total number of replacements across both passes."""
        return len(self.first_pass_replacements) + len(self.second_pass_replacements)
    
    def get_names_list(self) -> List[str]:
        """Return sorted list of discovered names."""
        return sorted(self.discovered_names)
    
    # Legacy compatibility - convert to list of dicts
    def get_replacements_as_dicts(self) -> List[Dict]:
        """Get all replacements as list of dicts for backward compatibility."""
        return [r.to_dict() for r in self.all_replacements]


# ============================================================================
# Text Normalization Utilities
# ============================================================================

def normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for comparison by handling Unicode variations.
    
    This function addresses the issue where different Unicode representations
    of the same text (e.g., straight quotes vs. curly quotes) are incorrectly
    flagged as mismatches during validation.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text that can be compared reliably
    """
    if not text:
        return text
    
    # Convert to Unicode normalization form NFC (composed characters)
    normalized = unicodedata.normalize('NFC', text)
    
    # Map common Unicode quote variants to ASCII equivalents
    quote_mappings = {
        '\u2018': "'",  # Left single curly quote
        '\u2019': "'",  # Right single curly quote
        '\u201A': "'",  # Single low-9 quote
        '\u201B': "'",  # Single high-reversed-9 quote
        '\u201C': '"',  # Left double curly quote
        '\u201D': '"',  # Right double curly quote
        '\u201E': '"',  # Double low-9 quote
        '\u201F': '"',  # Double high-reversed-9 quote
        '\u00AB': '"',  # Left-pointing double angle quote
        '\u00BB': '"',  # Right-pointing double angle quote
        '\u2039': "'",  # Single left-pointing angle quote
        '\u203A': "'",  # Single right-pointing angle quote
    }
    
    for unicode_char, ascii_char in quote_mappings.items():
        normalized = normalized.replace(unicode_char, ascii_char)
    
    return normalized


def words_equivalent(word1: str, word2: str) -> bool:
    """
    Check if two words are equivalent, handling Unicode variations.
    
    Args:
        word1: First word to compare
        word2: Second word to compare
        
    Returns:
        True if words are equivalent, False otherwise
    """
    norm1 = normalize_text_for_comparison(word1)
    norm2 = normalize_text_for_comparison(word2)
    return norm1.lower() == norm2.lower()


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
