#!/usr/bin/env python3
"""
Overlap detection strategies for chunk stitching.

This module implements various strategies for detecting overlapping regions
between transcript chunks:
1. Temporal overlap (using timestamps)
2. Exact/fuzzy matching
3. Partial word overlap (handles cut-off words)
4. Sequence alignment (handles insertions/deletions)
"""

from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

from local_transcribe.processing.chunk_stitching.core import OverlapResult
from local_transcribe.processing.chunk_stitching.word_utils import (
    get_word_texts,
    words_similar,
    is_fuzzy_match,
    is_partial_word_match,
)
from local_transcribe.lib.program_logger import log_progress


class OverlapDetector:
    """
    Detects overlapping regions between transcript chunks.
    
    Uses a cascade of strategies to find the best overlap:
    1. Temporal overlap (if timestamps available)
    2. Exact/fuzzy matching
    3. Partial word overlap
    4. Sequence alignment
    """
    
    def __init__(
        self,
        min_overlap_ratio: float = 0.6,
        similarity_threshold: float = 0.7,
        sequence_alignment_window: int = 20,
        min_sequence_match_length: int = 3,
    ):
        """
        Initialize the overlap detector.
        
        Args:
            min_overlap_ratio: Minimum ratio of overlapping words
            similarity_threshold: Threshold for word similarity
            sequence_alignment_window: Number of words to compare for sequence alignment
            min_sequence_match_length: Minimum matching words for sequence alignment
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.sequence_alignment_window = sequence_alignment_window
        self.min_sequence_match_length = min_sequence_match_length
    
    def find_best_overlap(
        self, 
        chunk1: List[Any], 
        chunk2: List[Any], 
        has_timestamps: bool
    ) -> OverlapResult:
        """
        Find the best overlap between two chunks using a cascade of strategies.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            has_timestamps: Whether words have timestamp information
            
        Returns:
            OverlapResult with overlap details
        """
        if not chunk1 or not chunk2:
            return OverlapResult()
        
        # Strategy 1: Use temporal overlap if timestamps available
        if has_timestamps:
            result = self._find_temporal_overlap(chunk1, chunk2)
            if result.found:
                result.method = 'temporal'
                return result
        
        # Get word texts for text-based strategies
        chunk1_texts = get_word_texts(chunk1)
        chunk2_texts = get_word_texts(chunk2)
        
        # Strategy 2: Exact/fuzzy matching
        overlap_start, overlap_length = self._find_exact_or_fuzzy_overlap(
            chunk1_texts, chunk2_texts
        )
        if overlap_length > 0:
            return OverlapResult(
                overlap_start=overlap_start,
                overlap_length=overlap_length,
                words_to_skip_in_chunk2=overlap_length,
                method='exact_or_fuzzy'
            )
        
        # Strategy 3: Partial word overlap
        partial_result = self._find_partial_word_overlap(chunk1_texts, chunk2_texts)
        if partial_result:
            return OverlapResult(
                overlap_start=partial_result[0],
                overlap_length=partial_result[1],
                words_to_skip_in_chunk2=partial_result[1],
                method='partial_word'
            )
        
        # Strategy 4: Sequence alignment
        seq_result = self._find_overlap_with_sequence_alignment(chunk1_texts, chunk2_texts)
        if seq_result:
            return OverlapResult(
                overlap_start=seq_result[0],
                overlap_length=seq_result[1],
                words_to_skip_in_chunk2=seq_result[2],
                method='sequence_alignment'
            )
        
        return OverlapResult()
    
    def _find_temporal_overlap(
        self, 
        chunk1: List[Dict], 
        chunk2: List[Dict]
    ) -> OverlapResult:
        """
        Find overlap using timestamp information.
        
        This method identifies words that fall within the overlapping time region
        and then uses sequence alignment to find the best match within that region.
        
        Args:
            chunk1: First chunk's word list (with timestamps)
            chunk2: Second chunk's word list (with timestamps)
            
        Returns:
            OverlapResult with temporal overlap details
        """
        if not chunk1 or not chunk2:
            return OverlapResult()
        
        chunk1_end_time = chunk1[-1].get('end', 0)
        chunk2_start_time = chunk2[0].get('start', 0)
        
        # If chunk2 starts after chunk1 ends, no temporal overlap
        if chunk2_start_time >= chunk1_end_time:
            return OverlapResult()
        
        # Find words in chunk1 that might overlap temporally with chunk2
        # Use a small buffer for timing imprecision
        time_buffer = 0.5  # 500ms buffer
        overlap_start_time = chunk2_start_time - time_buffer
        
        # Find first word in chunk1 that's in the overlap region
        chunk1_overlap_start = None
        for i, word in enumerate(chunk1):
            word_start = word.get('start', 0)
            if word_start >= overlap_start_time:
                chunk1_overlap_start = i
                break
        
        if chunk1_overlap_start is None:
            # Check if last few words overlap temporally
            for i in range(max(0, len(chunk1) - 10), len(chunk1)):
                if chunk1[i].get('start', 0) >= overlap_start_time:
                    chunk1_overlap_start = i
                    break
        
        if chunk1_overlap_start is None:
            return OverlapResult()
        
        # Find words in chunk2 that fall within the overlap time region
        chunk2_overlap_end = None
        for i, word in enumerate(chunk2):
            word_end = word.get('end', 0)
            if word_end > chunk1_end_time + time_buffer:
                chunk2_overlap_end = i
                break
        
        if chunk2_overlap_end is None:
            chunk2_overlap_end = len(chunk2)
        
        # Now perform sequence alignment on the overlapping regions
        chunk1_region = chunk1[chunk1_overlap_start:]
        chunk2_region = chunk2[:chunk2_overlap_end]
        
        if not chunk1_region or not chunk2_region:
            return OverlapResult()
        
        chunk1_texts = [w['text'].lower() for w in chunk1_region]
        chunk2_texts = [w['text'].lower() for w in chunk2_region]
        
        # Use sequence alignment to find best match
        matcher = SequenceMatcher(None, chunk1_texts, chunk2_texts)
        matching_blocks = matcher.get_matching_blocks()
        
        # Find the best match that reaches the end of chunk1_region
        best_match = None
        for block in matching_blocks:
            a_start, b_start, length = block
            if length < 2:  # Require at least 2 matching words
                continue
            
            a_end = a_start + length
            # Prefer matches that extend to the end of chunk1's region
            if a_end >= len(chunk1_texts) - 1:
                if best_match is None or length > best_match[2]:
                    best_match = (a_start, b_start, length)
        
        if best_match is None:
            return OverlapResult()
        
        a_start, b_start, length = best_match
        
        # Calculate positions in full chunks
        overlap_start_in_chunk1 = chunk1_overlap_start + a_start
        words_to_skip_in_chunk2 = b_start + length
        overlap_len_in_chunk1 = len(chunk1) - overlap_start_in_chunk1
        
        log_progress(
            f"Temporal overlap: matched {length} words in time region "
            f"[{overlap_start_time:.2f}s - {chunk1_end_time:.2f}s], "
            f"skipping {words_to_skip_in_chunk2} words from chunk2"
        )
        
        return OverlapResult(
            overlap_start=overlap_start_in_chunk1,
            overlap_length=overlap_len_in_chunk1,
            words_to_skip_in_chunk2=words_to_skip_in_chunk2,
            method='temporal'
        )
    
    def _find_exact_or_fuzzy_overlap(
        self, 
        chunk1_texts: List[str], 
        chunk2_texts: List[str]
    ) -> Tuple[int, int]:
        """
        Find overlap using exact or fuzzy matching.
        
        Args:
            chunk1_texts: Word texts from first chunk
            chunk2_texts: Word texts from second chunk
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length)
        """
        max_possible_overlap = min(len(chunk1_texts), len(chunk2_texts))
        
        # Check for overlaps from largest to smallest
        for overlap_size in range(max_possible_overlap, 0, -1):
            chunk1_end = chunk1_texts[-overlap_size:]
            chunk2_start = chunk2_texts[:overlap_size]
            
            # Check for exact match
            if chunk1_end == chunk2_start:
                return len(chunk1_texts) - overlap_size, overlap_size
            
            # Check for fuzzy match
            if is_fuzzy_match(
                chunk1_end, 
                chunk2_start,
                self.min_overlap_ratio,
                self.similarity_threshold
            ):
                return len(chunk1_texts) - overlap_size, overlap_size
        
        return 0, 0
    
    def _find_partial_word_overlap(
        self, 
        chunk1_texts: List[str], 
        chunk2_texts: List[str]
    ) -> Optional[Tuple[int, int]]:
        """
        Check for partial word overlaps at chunk boundaries.
        
        Args:
            chunk1_texts: Word texts from first chunk
            chunk2_texts: Word texts from second chunk
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length) if found, else None
        """
        if not chunk1_texts or not chunk2_texts:
            return None
        
        last_word = chunk1_texts[-1]
        first_word = chunk2_texts[0]
        
        if is_partial_word_match(last_word, first_word):
            log_progress(f"Partial word overlap: '{last_word}' ~ '{first_word}'")
            return len(chunk1_texts) - 1, 1
        
        # Check two-word partial matches
        if len(chunk1_texts) >= 2 and len(chunk2_texts) >= 2:
            if (is_partial_word_match(chunk1_texts[-2], chunk2_texts[0]) and
                words_similar(chunk1_texts[-1], chunk2_texts[1], self.similarity_threshold)):
                log_progress(f"Two-word partial overlap detected")
                return len(chunk1_texts) - 2, 2
        
        return None
    
    def _find_overlap_with_sequence_alignment(
        self, 
        chunk1_texts: List[str], 
        chunk2_texts: List[str]
    ) -> Optional[Tuple[int, int, int]]:
        """
        Find overlap using sequence alignment, handling insertions/deletions.
        
        This method can detect overlaps even when one chunk has extra words
        (insertions) or missing words (deletions) compared to the other.
        
        Args:
            chunk1_texts: Word texts from first chunk
            chunk2_texts: Word texts from second chunk
            
        Returns:
            Tuple of (overlap_start_in_chunk1, overlap_len_in_chunk1, words_to_skip_in_chunk2)
            or None if no valid overlap found
        """
        if not chunk1_texts or not chunk2_texts:
            return None
        
        window_size = self.sequence_alignment_window
        chunk1_window = chunk1_texts[-window_size:] if len(chunk1_texts) > window_size else chunk1_texts
        chunk2_window = chunk2_texts[:window_size] if len(chunk2_texts) > window_size else chunk2_texts
        
        chunk1_lower = [w.lower() for w in chunk1_window]
        chunk2_lower = [w.lower() for w in chunk2_window]
        
        matcher = SequenceMatcher(None, chunk1_lower, chunk2_lower)
        matching_blocks = matcher.get_matching_blocks()
        
        # Adaptive minimum match length based on window size
        # For short sequences, allow shorter matches
        effective_min_match = min(
            self.min_sequence_match_length, 
            max(2, len(chunk1_lower) // 3)
        )
        
        # Strategy 1: Find a single block that reaches the end
        best_match = None
        for block in matching_blocks:
            a_start, b_start, length = block
            if length < effective_min_match:
                continue
            
            a_end = a_start + length
            # Must reach near the end of chunk1_window
            if a_end >= len(chunk1_lower) - 1:
                if best_match is None:
                    best_match = (a_start, b_start, length)
                elif a_end > best_match[0] + best_match[2]:
                    best_match = (a_start, b_start, length)
                elif a_end == best_match[0] + best_match[2] and length > best_match[2]:
                    best_match = (a_start, b_start, length)
        
        # Strategy 2: If no single block works, try to find combined blocks that span the overlap
        # This handles cases where insertions break up the match into multiple blocks
        if best_match is None:
            # Filter to significant blocks (at least 1 word)
            significant_blocks = [(a, b, l) for a, b, l in matching_blocks if l >= 1]
            
            # Sort by position in chunk1
            significant_blocks.sort(key=lambda x: x[0])
            
            # Look for a sequence of blocks that collectively reach the end
            for i, (a_start, b_start, length) in enumerate(significant_blocks):
                # Check if this block or subsequent blocks reach the end
                total_matched = length
                last_a_end = a_start + length
                last_b_end = b_start + length
                
                for j in range(i + 1, len(significant_blocks)):
                    next_a, next_b, next_len = significant_blocks[j]
                    # Check if blocks are adjacent or near-adjacent in chunk1
                    if next_a <= last_a_end + 2:  # Allow small gap
                        total_matched += next_len
                        last_a_end = next_a + next_len
                        last_b_end = next_b + next_len
                
                # If we've reached the end of chunk1 with enough total matches
                if last_a_end >= len(chunk1_lower) - 1 and total_matched >= effective_min_match:
                    # Use the first block's start position
                    words_to_skip = last_b_end
                    
                    chunk1_offset = len(chunk1_texts) - len(chunk1_window)
                    overlap_start_in_chunk1 = chunk1_offset + a_start
                    overlap_len_in_chunk1 = len(chunk1_texts) - overlap_start_in_chunk1
                    
                    log_progress(
                        f"Sequence alignment (combined): matched {total_matched} words across blocks, "
                        f"keeping chunk1[:{overlap_start_in_chunk1}] ({overlap_start_in_chunk1} words), "
                        f"skipping chunk2[:{words_to_skip}] ({words_to_skip} words)"
                    )
                    
                    return (overlap_start_in_chunk1, overlap_len_in_chunk1, words_to_skip)
        
        if best_match is None:
            return None
        
        a_start, b_start, length = best_match
        
        # Calculate actual positions in full chunks
        chunk1_offset = len(chunk1_texts) - len(chunk1_window)
        overlap_start_in_chunk1 = chunk1_offset + a_start
        overlap_len_in_chunk1 = len(chunk1_texts) - overlap_start_in_chunk1
        words_to_skip_in_chunk2 = b_start + length
        
        # Log if there's a difference (insertion/deletion detected)
        if overlap_len_in_chunk1 != words_to_skip_in_chunk2:
            diff = words_to_skip_in_chunk2 - overlap_len_in_chunk1
            if diff > 0:
                log_progress(f"Sequence alignment: detected {diff} extra word(s) in chunk2 overlap region")
            else:
                log_progress(f"Sequence alignment: detected {-diff} extra word(s) in chunk1 overlap region")
        
        log_progress(
            f"Sequence alignment: matched {length} words, "
            f"keeping chunk1[:{overlap_start_in_chunk1}] ({overlap_start_in_chunk1} words), "
            f"skipping chunk2[:{words_to_skip_in_chunk2}] ({words_to_skip_in_chunk2} words)"
        )
        
        return (overlap_start_in_chunk1, overlap_len_in_chunk1, words_to_skip_in_chunk2)
