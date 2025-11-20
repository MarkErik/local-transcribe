#!/usr/bin/env python3
"""
Chunk merger for intelligently merging overlapping transcript chunks from audio transcription.

This module provides functionality to merge overlapping chunks from transcriber output,
handling cases where words may be cut off at chunk boundaries (e.g. "generational" -> "rational")
and dealing with slight differences in similar-sounding words.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher


class ChunkMerger:
    """
    A class for merging overlapping transcript chunks with intelligent overlap detection.
    """
    
    def __init__(self, min_overlap_ratio: float = 0.6, similarity_threshold: float = 0.7, verbose: bool = False):
        """
        Initialize the chunk merger with configurable thresholds.
        
        Args:
            min_overlap_ratio: Minimum ratio of overlapping words to consider a valid overlap
            similarity_threshold: Threshold for word similarity using SequenceMatcher
            verbose: Enable verbose output for debugging and progress monitoring
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.verbose = verbose
    
    def merge_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Merge a list of transcript chunks into a single transcript.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'words' keys
            
        Returns:
            Merged transcript as a string
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return " ".join(chunks[0]["words"])
        
        # Start with the first chunk
        merged_words = chunks[0]["words"].copy()
        
        if self.verbose:
            print(f"[ChunkMerger] Processing {len(chunks)} chunks")
        
        # Iteratively merge each subsequent chunk
        for i in range(1, len(chunks)):
            if self.verbose:
                print(f"[ChunkMerger] Processing chunk {i + 1} of {len(chunks)}")
            current_chunk_words = chunks[i]["words"]
            merged_words = self._merge_two_chunks(merged_words, current_chunk_words)
        
        if self.verbose:
            print(f"[ChunkMerger] Merge complete: {len(merged_words)} words total")
        
        return " ".join(merged_words)
    
    def _merge_two_chunks(self, chunk1: List[str], chunk2: List[str]) -> List[str]:
        """
        Merge two chunks, handling overlaps intelligently.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            
        Returns:
            Merged word list
        """
        # Find the overlap region
        overlap_start, overlap_length = self._find_overlap(chunk1, chunk2)
        
        if overlap_length == 0:
            # No overlap found, simply concatenate
            if self.verbose:
                print(f"[ChunkMerger] No overlap detected between chunks; concatenating directly")
            return chunk1 + chunk2
        
        if self.verbose:
            overlapping_words = chunk1[overlap_start:overlap_start + overlap_length]
            print(f"[ChunkMerger] Overlap found: start={overlap_start}, length={overlap_length}, words={overlapping_words}")
        
        # Handle the overlap
        if overlap_length == len(chunk2):
            # Second chunk is entirely contained in first
            return chunk1
        
        # Take the non-overlapping part of chunk1 and all of chunk2
        # This handles cases where chunk2 might have better transcription at the boundary
        return chunk1[:overlap_start] + chunk2
    
    def _find_overlap(self, chunk1: List[str], chunk2: List[str]) -> Tuple[int, int]:
        """
        Find the overlap between two chunks.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length)
        """
        # Maximum possible overlap length
        max_possible_overlap = min(len(chunk1), len(chunk2))
        
        # Check for overlaps from largest to smallest
        for overlap_size in range(max_possible_overlap, 0, -1):
            # Try to find this size overlap at the end of chunk1 and start of chunk2
            chunk1_end = chunk1[-overlap_size:]
            chunk2_start = chunk2[:overlap_size]
            
            # Check for exact match
            if chunk1_end == chunk2_start:
                return len(chunk1) - overlap_size, overlap_size
            
            # Check for fuzzy match (similar words)
            if self._is_fuzzy_match(chunk1_end, chunk2_start):
                return len(chunk1) - overlap_size, overlap_size
        
        # Check for partial word overlaps (e.g., "generational" -> "rational")
        # This handles cases where words might be cut off at boundaries
        partial_overlap = self._find_partial_word_overlap(chunk1, chunk2)
        if partial_overlap:
            return partial_overlap
        
        # No overlap found
        return 0, 0
    
    def _is_fuzzy_match(self, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two word lists are fuzzy matches (similar words).
        
        Args:
            words1: First list of words
            words2: Second list of words
            
        Returns:
            True if the lists are fuzzy matches
        """
        if len(words1) != len(words2):
            return False
        
        # Check each corresponding word pair
        matches = 0
        fuzzy_pairs = []
        for w1, w2 in zip(words1, words2):
            if self._words_similar(w1, w2):
                matches += 1
                if w1 != w2 and self.verbose:
                    ratio = SequenceMatcher(None, w1.lower(), w2.lower()).ratio()
                    fuzzy_pairs.append(f"'{w1}' ~ '{w2}' (ratio={ratio:.2f})")
        
        # Require at least min_overlap_ratio of words to be similar
        is_match = matches / len(words1) >= self.min_overlap_ratio
        if is_match and fuzzy_pairs and self.verbose:
            print(f"[ChunkMerger] Fuzzy matches in overlap: {', '.join(fuzzy_pairs)}")
        
        return is_match
    
    def _words_similar(self, word1: str, word2: str) -> bool:
        """
        Check if two words are similar using SequenceMatcher.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if words are similar
        """
        # Direct comparison
        if word1 == word2:
            return True
        
        # Case-insensitive comparison
        if word1.lower() == word2.lower():
            return True
        
        # Similarity ratio
        ratio = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        return ratio >= self.similarity_threshold
    
    def _find_partial_word_overlap(self, chunk1: List[str], chunk2: List[str]) -> Optional[Tuple[int, int]]:
        """
        Check for partial word overlaps at chunk boundaries.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length) if found, else None
        """
        # Check if the last word of chunk1 could be a partial match for the first word of chunk2
        if not chunk1 or not chunk2:
            return None
        
        last_word_chunk1 = chunk1[-1]
        first_word_chunk2 = chunk2[0]
        
        # Check if one word could be a continuation of the other
        if self._is_partial_word_match(last_word_chunk1, first_word_chunk2):
            # Replace the partial word in chunk1 with the complete word from chunk2
            if self.verbose:
                print(f"[ChunkMerger] Partial word overlap: replacing '{last_word_chunk1}' with '{first_word_chunk2}'")
            return len(chunk1) - 1, 1
        
        # Check for two-word partial matches
        if len(chunk1) >= 2 and len(chunk2) >= 2:
            last_two_chunk1 = chunk1[-2:]
            first_two_chunk2 = chunk2[:2]
            
            if (self._is_partial_word_match(last_two_chunk1[0], first_two_chunk2[0]) and
                self._words_similar(last_two_chunk1[1], first_two_chunk2[1])):
                if self.verbose:
                    print(f"[ChunkMerger] Partial word overlap: replacing '{last_two_chunk1}' with '{first_two_chunk2}'")
                return len(chunk1) - 2, 2
        
        return None
    
    def _is_partial_word_match(self, word1: str, word2: str) -> bool:
        """
        Check if one word could be a partial match for another (e.g. "generational" -> "rational").
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if one word could be a partial match for the other
        """
        # If one word is much longer, check if it contains the other
        if len(word1) > len(word2) * 1.5 and word2.lower() in word1.lower():
            return True
        
        if len(word2) > len(word1) * 1.5 and word1.lower() in word2.lower():
            return True
        
        # Check for common suffixes/prefixes that might indicate partial transcription
        # For example, "generational" and "rational" share the "-ational" suffix
        min_len = min(len(word1), len(word2))
        
        # Check for significant overlap at the end
        for i in range(min_len // 2, min_len):
            if word1.lower()[-i:] == word2.lower()[-i:]:
                return True
        
        # Check for significant overlap at the beginning
        for i in range(min_len // 2, min_len):
            if word1.lower()[:i] == word2.lower()[:i]:
                return True
        
        return False


def merge_chunks(chunks: List[Dict[str, Any]], **kwargs) -> str:
    """
    Convenience function to merge transcript chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_id' and 'words' keys
        **kwargs: Additional arguments (min_overlap_ratio, similarity_threshold, verbose)
        
    Returns:
        Merged transcript as a string
    """
    merger = ChunkMerger(
        min_overlap_ratio=kwargs.get('min_overlap_ratio', 0.6),
        similarity_threshold=kwargs.get('similarity_threshold', 0.7),
        verbose=kwargs.get('verbose', False)
    )
    return merger.merge_chunks(chunks)