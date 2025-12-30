#!/usr/bin/env python3
"""
Text chunking utilities for de-identification.

Provides functions to split text and word segments into overlapping chunks
for LLM processing.
"""

from typing import List, Optional, Any

from local_transcribe.lib.program_logger import log_debug

from .core import Chunk


def chunk_word_segments(
    segments: List[Any],  # List[WordSegment]
    chunk_size: int = 400,
    overlap_size: int = 70,
    min_final_chunk: int = 200
) -> List[Chunk]:
    """
    Chunk word segments for LLM processing with overlap.
    
    Args:
        segments: List of WordSegment objects
        chunk_size: Maximum words per chunk
        overlap_size: Words of overlap between chunks for context
        min_final_chunk: Minimum words for final chunk (merges if smaller)
        
    Returns:
        List of Chunk objects with segments and text
    """
    chunks = []
    i = 0
    
    while i < len(segments):
        end_idx = min(i + chunk_size, len(segments))
        chunk_segments = segments[i:end_idx]
        chunk_text = " ".join(seg.text for seg in chunk_segments)
        
        log_debug(
            f"Creating chunk {len(chunks)+1} with {len(chunk_segments)} segments "
            f"(words: {len(chunk_text.split())})"
        )
        if chunk_text.split():
            log_debug(f"First few words: {chunk_text.split()[:5]}")
        
        chunks.append(Chunk(
            text=chunk_text,
            start_idx=i,
            end_idx=end_idx,
            segments=list(chunk_segments),
            words=[]
        ))
        
        # Check if we're near the end
        remaining = len(segments) - end_idx
        if remaining == 0:
            break
        elif remaining < min_final_chunk:
            # Merge small final chunk into current chunk
            merged_segments = list(segments[i:])
            chunks[-1] = Chunk(
                text=" ".join(seg.text for seg in segments[i:]),
                start_idx=i,
                end_idx=len(segments),
                segments=merged_segments,
                words=[]
            )
            log_debug(f"Merged final chunk with {len(merged_segments)} segments")
            break
        
        # Move forward, accounting for overlap
        i += chunk_size - overlap_size
    
    return chunks


def chunk_plain_text(
    words: List[str],
    chunk_size: int = 400,
    overlap_size: int = 70,
    min_final_chunk: int = 200
) -> List[Chunk]:
    """
    Chunk plain text words for LLM processing with overlap.
    
    Args:
        words: List of word strings
        chunk_size: Maximum words per chunk
        overlap_size: Words of overlap between chunks for context
        min_final_chunk: Minimum words for final chunk (merges if smaller)
        
    Returns:
        List of Chunk objects with words and text
    """
    chunks = []
    i = 0
    
    while i < len(words):
        end_idx = min(i + chunk_size, len(words))
        chunk_words = words[i:end_idx]
        chunk_text = " ".join(chunk_words)
        
        log_debug(
            f"Creating text chunk {len(chunks)+1} with {len(chunk_words)} words"
        )
        if chunk_words:
            log_debug(f"First few words: {chunk_words[:5]}")
        
        chunks.append(Chunk(
            text=chunk_text,
            start_idx=i,
            end_idx=end_idx,
            segments=[],
            words=list(chunk_words)
        ))
        
        # Check if we're near the end
        remaining = len(words) - end_idx
        if remaining == 0:
            break
        elif remaining < min_final_chunk:
            # Merge small final chunk into current chunk
            merged_words = list(words[i:])
            chunks[-1] = Chunk(
                text=" ".join(words[i:]),
                start_idx=i,
                end_idx=len(words),
                segments=[],
                words=merged_words
            )
            log_debug(f"Merged final text chunk with {len(merged_words)} words")
            break
        
        # Move forward, accounting for overlap
        i += chunk_size - overlap_size
    
    return chunks


def merge_processed_text_chunks(
    processed_chunks: List[str],
    overlap_size: int
) -> str:
    """
    Merge processed text chunks, handling overlaps.
    
    For simplicity, we use the first occurrence of overlapping sections.
    
    Args:
        processed_chunks: List of processed text strings
        overlap_size: Number of words that overlap between chunks
        
    Returns:
        Merged text string
    """
    if not processed_chunks:
        return ""
    
    if len(processed_chunks) == 1:
        return processed_chunks[0]
    
    # Start with first chunk
    merged_words = processed_chunks[0].split()
    
    # Add subsequent chunks, skipping overlap
    for i in range(1, len(processed_chunks)):
        chunk_words = processed_chunks[i].split()
        # Skip the overlap region
        merged_words.extend(chunk_words[overlap_size:])
    
    return " ".join(merged_words)
