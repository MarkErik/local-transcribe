#!/usr/bin/env python3
"""
Chunk stitching for overlapping transcript chunks from audio transcription.

This package provides functionality to stitch overlapping chunks from transcriber output,
handling cases where words may be cut off at chunk boundaries (e.g. "generational" -> "rational")
and dealing with slight differences in similar-sounding words.

Supports both:
- Simple string words: chunks with "words" as List[str]
- Timestamped words: chunks with "words" as List[Dict] with "text", "start", "end" keys

Usage:
    from local_transcribe.processing.chunk_stitching import (
        stitch_chunks,  # Convenience function
        ChunkStitcher,  # Class for more control
    )
    
    # Simple usage
    result = stitch_chunks(chunks, intermediate_dir=Path("./output"))
    
    # With stitcher instance for more control
    stitcher = ChunkStitcher(
        min_overlap_ratio=0.6,
        similarity_threshold=0.7,
        intermediate_dir=Path("./output")
    )
    result = stitcher.stitch_chunks(chunks)
"""

# Main API
from local_transcribe.processing.chunk_stitching.stitcher import (
    ChunkStitcher,
    stitch_chunks,
)

# Core data structures
from local_transcribe.processing.chunk_stitching.core import (
    ChunkStitcherConfig,
    OverlapResult,
    ChunkInfo,
    StitchStepInfo,
)

# Word utilities
from local_transcribe.processing.chunk_stitching.word_utils import (
    get_word_text,
    get_word_texts,
    has_timestamps,
    words_similar,
    is_fuzzy_match,
    is_partial_word_match,
)

# Overlap detection
from local_transcribe.processing.chunk_stitching.overlap_strategies import (
    OverlapDetector,
)

# Debug utilities
from local_transcribe.processing.chunk_stitching.debug import (
    StitcherDebugWriter,
    capture_chunk_info,
)


__all__ = [
    # Main API
    'ChunkStitcher',
    'stitch_chunks',
    
    # Configuration and data structures
    'ChunkStitcherConfig',
    'OverlapResult',
    'ChunkInfo',
    'StitchStepInfo',
    
    # Word utilities
    'get_word_text',
    'get_word_texts',
    'has_timestamps',
    'words_similar',
    'is_fuzzy_match',
    'is_partial_word_match',
    
    # Overlap detection
    'OverlapDetector',
    
    # Debug utilities
    'StitcherDebugWriter',
    'capture_chunk_info',
]
