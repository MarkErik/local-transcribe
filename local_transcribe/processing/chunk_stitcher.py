#!/usr/bin/env python3
"""
Chunk stitcher for stitching overlapping transcript chunks from audio transcription.

This module provides functionality to stitch overlapping chunks from transcriber output,
handling cases where words may be cut off at chunk boundaries (e.g. "generational" -> "rational")
and dealing with slight differences in similar-sounding words.

Supports both:
- Simple string words: chunks with "words" as List[str]
- Timestamped words: chunks with "words" as List[Dict] with "text", "start", "end" keys
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from local_transcribe.lib.program_logger import log_progress, log_debug, get_output_context
from difflib import SequenceMatcher
from local_transcribe.framework.plugin_interfaces import WordSegment


class ChunkStitcher:
    """
    A class for stitching overlapping transcript chunks with  overlap detection.
    """
    
    def __init__(self, min_overlap_ratio: float = 0.6, similarity_threshold: float = 0.7,
                 intermediate_dir: Optional[Path] = None,
                 sequence_alignment_window: int = 20, min_sequence_match_length: int = 3,
                 skip_single_chunk_debug: bool = True,
                 use_timestamped_debug_dir: bool = True):
        """
        Initialize the chunk stitcher with configurable thresholds.
        
        Args:
            min_overlap_ratio: Minimum ratio of overlapping words to consider a valid overlap
            similarity_threshold: Threshold for word similarity using SequenceMatcher
            intermediate_dir: Optional path for saving debug files when DEBUG logging is enabled.
                              If use_timestamped_debug_dir is False, debug files are saved directly here.
            sequence_alignment_window: Number of words to compare from each chunk for sequence alignment
            min_sequence_match_length: Minimum number of matching words required for sequence alignment
            skip_single_chunk_debug: If True, skip debug output when there's only one chunk (no stitching needed)
            use_timestamped_debug_dir: If True, create a timestamped subdirectory under intermediate_dir.
                                       If False, use intermediate_dir directly for debug files.
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.intermediate_dir = intermediate_dir
        self.sequence_alignment_window = sequence_alignment_window
        self.min_sequence_match_length = min_sequence_match_length
        self.skip_single_chunk_debug = skip_single_chunk_debug
        self.use_timestamped_debug_dir = use_timestamped_debug_dir
        
        # Setup debug directory if DEBUG logging is enabled
        self.debug_dir: Optional[Path] = None
        self._debug_enabled = get_output_context().should_log("DEBUG")
        
        if self._debug_enabled and intermediate_dir:
            if use_timestamped_debug_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.debug_dir = Path(intermediate_dir) / "chunk_stitching" / "stitcher_debug" / timestamp
            else:
                # Use intermediate_dir directly as the debug directory
                self.debug_dir = Path(intermediate_dir)
            
            # Create the debug directory
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Chunk stitcher debug output: {self.debug_dir}")

    # =========================================================================
    # Helper methods for unified word handling
    # =========================================================================
    
    def _get_word_text(self, word: Union[str, Dict[str, Any]]) -> str:
        """Extract text from a word (handles both string and dict formats)."""
        if isinstance(word, str):
            return word
        return word.get("text", "")
    
    def _get_word_texts(self, words: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Extract text from a list of words."""
        return [self._get_word_text(w) for w in words]
    
    def _has_timestamps(self, words: List[Any]) -> bool:
        """Check if words have timestamp information."""
        if not words:
            return False
        first_word = words[0]
        return isinstance(first_word, dict) and "text" in first_word and "start" in first_word
    
    def _should_save_debug(self, chunk_count: int) -> bool:
        """
        Determine if debug files should be saved.
        
        Args:
            chunk_count: Number of chunks to process
            
        Returns:
            True if debug files should be saved
        """
        if not self.debug_dir:
            return False
        
        # Skip debug for single chunks if configured
        if self.skip_single_chunk_debug and chunk_count <= 1:
            return False
        
        return True

    # =========================================================================
    # Main entry point
    # =========================================================================
    
    def stitch_chunks(self, chunks: List[Dict[str, Any]]) -> Union[str, List[WordSegment]]:
        """
        Stitch a list of transcript chunks into a single transcript.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'words' keys
                   Words can be either List[str] or List[Dict] with timestamps
            
        Returns:
            If words are strings: Stitched transcript as a string
            If words have timestamps: List[WordSegment] with preserved timestamps
        """
        if not chunks:
            return ""
        
        # Detect format: check if words have timestamps
        has_timestamps = False
        for chunk in chunks:
            if chunk["words"]:
                has_timestamps = self._has_timestamps(chunk["words"])
                break
        
        # Check if we should save debug output
        save_debug = self._should_save_debug(len(chunks))
        
        # Save debug input (only if we should)
        if save_debug:
            self._save_debug_input(chunks, has_timestamps)
        
        if len(chunks) == 1:
            return self._finalize_result(chunks[0]["words"], chunks, has_timestamps, 0, save_debug)
        
        # Start with the first chunk
        stitched_words = list(chunks[0]["words"])
        
        log_progress(f"Processing {len(chunks)} chunks ({'timestamped' if has_timestamps else 'string'} words)")
        
        # Iteratively stitch each subsequent chunk
        for i in range(1, len(chunks)):
            log_progress(f"Processing chunk {i + 1} of {len(chunks)}")
            current_chunk_words = chunks[i]["words"]
            chunk_id = chunks[i].get('chunk_id', i + 1)
            
            stitched_words = self._stitch_two_chunks(
                stitched_words, 
                current_chunk_words,
                has_timestamps=has_timestamps,
                step_num=i,
                chunk_id=chunk_id,
                save_debug=save_debug
            )
        
        log_progress(f"Stitch complete: {len(stitched_words)} words total")
        
        return self._finalize_result(stitched_words, chunks, has_timestamps, len(chunks) - 1, save_debug)
    
    def _finalize_result(self, words: List[Any], chunks: List[Dict], 
                         has_timestamps: bool, stitch_steps: int,
                         save_debug: bool = True) -> Union[str, List[WordSegment]]:
        """Convert final word list to appropriate output format."""
        if has_timestamps:
            result = [
                WordSegment(text=w["text"], start=w["start"], end=w["end"], speaker=w.get("speaker"))
                for w in words
            ]
            if save_debug:
                self._save_debug_output(result, has_timestamps=True)
        else:
            result = " ".join(words)
            if save_debug:
                self._save_debug_output(result, has_timestamps=False)
        
        if save_debug:
            self._save_debug_session_summary(chunks, has_timestamps, stitch_steps, len(words))
        return result

    # =========================================================================
    # Core stitching logic (unified for both formats)
    # =========================================================================
    
    def _stitch_two_chunks(self, chunk1: List[Any], chunk2: List[Any], 
                           has_timestamps: bool, step_num: int = 0, 
                           chunk_id: Any = None, save_debug: bool = True) -> List[Any]:
        """
        Stitch two chunks, handling overlaps.
        
        Uses a cascade of overlap detection strategies:
        1. Temporal overlap (if timestamps available)
        2. Exact/fuzzy matching
        3. Sequence alignment (handles insertions/deletions)
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            has_timestamps: Whether words have timestamp information
            step_num: Step number for debug output
            chunk_id: Chunk ID for debug output
            save_debug: Whether to save debug output
            
        Returns:
            Stitched word list
        """
        # Capture debug info before stitching
        chunk1_info, chunk2_info = self._capture_chunk_info(chunk1, chunk2, chunk_id, has_timestamps)
        
        # Try to find overlap using cascade of strategies
        overlap_result = self._find_best_overlap(chunk1, chunk2, has_timestamps)
        
        overlap_start = overlap_result['overlap_start']
        words_to_skip_in_chunk2 = overlap_result['words_to_skip_in_chunk2']
        overlap_length = overlap_result['overlap_length']
        
        # Build result
        # The overlap region is at the END of chunk1 and the START of chunk2
        # We keep chunk1 completely and skip the overlapping portion from chunk2
        if overlap_length == 0:
            log_progress("No overlap detected between chunks; concatenating directly")
            result = chunk1 + chunk2
        elif words_to_skip_in_chunk2 >= len(chunk2):
            # Second chunk is entirely contained in first (all overlap)
            result = list(chunk1)
        else:
            overlapping_words = self._get_word_texts(chunk1[overlap_start:overlap_start + overlap_length])
            log_progress(f"Overlap found: start={overlap_start}, length={overlap_length}, words={overlapping_words}")
            # Keep all of chunk1, skip the overlapping words from chunk2
            result = list(chunk1) + chunk2[words_to_skip_in_chunk2:]
        
        # Capture and save debug info (only if enabled)
        if save_debug:
            overlap_info = {
                'overlap_found': overlap_length > 0,
                'overlap_start': overlap_start,
                'overlap_length': overlap_length,
                'overlapping_words': self._get_word_texts(chunk1[overlap_start:overlap_start + overlap_length]) if overlap_length > 0 else [],
                'method': overlap_result.get('method', 'none'),
                'words_to_skip_in_chunk2': words_to_skip_in_chunk2
            }
            
            result_info = {
                'word_count': len(result),
                'words_from_chunk1': len(chunk1) if overlap_length > 0 else len(chunk1),
                'words_from_chunk2': len(chunk2) - words_to_skip_in_chunk2 if overlap_length > 0 else len(chunk2)
            }
            
            if has_timestamps and result:
                result_info['time_range'] = {
                    'start': result[0].get('start', 0),
                    'end': result[-1].get('end', 0)
                }
            
            self._save_debug_stitch_step(step_num, chunk1_info, chunk2_info, overlap_info, result_info)
        
        return result
    
    def _find_best_overlap(self, chunk1: List[Any], chunk2: List[Any], 
                           has_timestamps: bool) -> Dict[str, Any]:
        """
        Find the best overlap between two chunks using a cascade of strategies.
        
        Returns:
            Dict with keys: overlap_start, overlap_length, words_to_skip_in_chunk2, method
        """
        no_overlap = {
            'overlap_start': 0, 
            'overlap_length': 0, 
            'words_to_skip_in_chunk2': 0,
            'method': 'none'
        }
        
        if not chunk1 or not chunk2:
            return no_overlap
        
        # Strategy 1: Use temporal overlap if timestamps available
        if has_timestamps:
            temporal_result = self._find_temporal_overlap(chunk1, chunk2)
            if temporal_result:
                temporal_result['method'] = 'temporal'
                return temporal_result
        
        # Strategy 2: Exact/fuzzy matching
        chunk1_texts = self._get_word_texts(chunk1)
        chunk2_texts = self._get_word_texts(chunk2)
        
        overlap_start, overlap_length = self._find_exact_or_fuzzy_overlap(chunk1_texts, chunk2_texts)
        if overlap_length > 0:
            return {
                'overlap_start': overlap_start,
                'overlap_length': overlap_length,
                'words_to_skip_in_chunk2': overlap_length,
                'method': 'exact_or_fuzzy'
            }
        
        # Strategy 3: Partial word overlap
        partial_result = self._find_partial_word_overlap(chunk1_texts, chunk2_texts)
        if partial_result:
            return {
                'overlap_start': partial_result[0],
                'overlap_length': partial_result[1],
                'words_to_skip_in_chunk2': partial_result[1],
                'method': 'partial_word'
            }
        
        # Strategy 4: Sequence alignment (handles insertions/deletions)
        seq_result = self._find_overlap_with_sequence_alignment(chunk1_texts, chunk2_texts)
        if seq_result:
            return {
                'overlap_start': seq_result[0],
                'overlap_length': seq_result[1],
                'words_to_skip_in_chunk2': seq_result[2],
                'method': 'sequence_alignment'
            }
        
        return no_overlap

    # =========================================================================
    # Overlap detection strategies
    # =========================================================================
    
    def _find_temporal_overlap(self, chunk1: List[Dict], chunk2: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Find overlap using timestamp information.
        
        This method identifies words that fall within the overlapping time region
        and then uses sequence alignment to find the best match within that region.
        """
        if not chunk1 or not chunk2:
            return None
        
        chunk1_end_time = chunk1[-1].get('end', 0)
        chunk2_start_time = chunk2[0].get('start', 0)
        
        # If chunk2 starts after chunk1 ends, no temporal overlap
        if chunk2_start_time >= chunk1_end_time:
            return None
        
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
            return None
        
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
            return None
        
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
            return None
        
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
        
        return {
            'overlap_start': overlap_start_in_chunk1,
            'overlap_length': overlap_len_in_chunk1,
            'words_to_skip_in_chunk2': words_to_skip_in_chunk2
        }
    
    def _find_exact_or_fuzzy_overlap(self, chunk1_texts: List[str], 
                                      chunk2_texts: List[str]) -> Tuple[int, int]:
        """
        Find overlap using exact or fuzzy matching.
        
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
            if self._is_fuzzy_match(chunk1_end, chunk2_start):
                return len(chunk1_texts) - overlap_size, overlap_size
        
        return 0, 0
    
    def _find_partial_word_overlap(self, chunk1_texts: List[str], 
                                    chunk2_texts: List[str]) -> Optional[Tuple[int, int]]:
        """
        Check for partial word overlaps at chunk boundaries.
        
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length) if found, else None
        """
        if not chunk1_texts or not chunk2_texts:
            return None
        
        last_word = chunk1_texts[-1]
        first_word = chunk2_texts[0]
        
        if self._is_partial_word_match(last_word, first_word):
            log_progress(f"Partial word overlap: '{last_word}' ~ '{first_word}'")
            return len(chunk1_texts) - 1, 1
        
        # Check two-word partial matches
        if len(chunk1_texts) >= 2 and len(chunk2_texts) >= 2:
            if (self._is_partial_word_match(chunk1_texts[-2], chunk2_texts[0]) and
                self._words_similar(chunk1_texts[-1], chunk2_texts[1])):
                log_progress(f"Two-word partial overlap detected")
                return len(chunk1_texts) - 2, 2
        
        return None
    
    def _find_overlap_with_sequence_alignment(self, chunk1_texts: List[str], 
                                               chunk2_texts: List[str]) -> Optional[Tuple[int, int, int]]:
        """
        Find overlap using sequence alignment, handling insertions/deletions.
        
        This method can detect overlaps even when one chunk has extra words
        (insertions) or missing words (deletions) compared to the other.
        
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
        effective_min_match = min(self.min_sequence_match_length, max(2, len(chunk1_lower) // 3))
        
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
                    best_match = (a_start, b_start, len(chunk1_lower) - a_start)
                    # Words to skip in chunk2 is the end position in chunk2
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

    # =========================================================================
    # Word comparison utilities
    # =========================================================================
    
    def _is_fuzzy_match(self, words1: List[str], words2: List[str]) -> bool:
        """Check if two word lists are fuzzy matches."""
        if len(words1) != len(words2):
            return False
        
        matches = 0
        for w1, w2 in zip(words1, words2):
            if self._words_similar(w1, w2):
                matches += 1
        
        return matches / len(words1) >= self.min_overlap_ratio
    
    def _words_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar."""
        if word1 == word2:
            return True
        if word1.lower() == word2.lower():
            return True
        
        ratio = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        return ratio >= self.similarity_threshold
    
    def _is_partial_word_match(self, word1: str, word2: str) -> bool:
        """
        Check if one word could be a partial match for another.
        E.g., "generational" -> "rational"
        """
        w1, w2 = word1.lower(), word2.lower()
        
        # One word contains the other
        if len(w1) > len(w2) * 1.5 and w2 in w1:
            return True
        if len(w2) > len(w1) * 1.5 and w1 in w2:
            return True
        
        # Check for significant suffix/prefix overlap
        min_len = min(len(w1), len(w2))
        for i in range(min_len // 2, min_len):
            if w1[-i:] == w2[-i:] or w1[:i] == w2[:i]:
                return True
        
        return False

    # =========================================================================
    # Debug output methods
    # =========================================================================
    
    def _capture_chunk_info(self, chunk1: List[Any], chunk2: List[Any], 
                            chunk_id: Any, has_timestamps: bool) -> Tuple[Dict, Dict]:
        """Capture info about chunks for debug output."""
        chunk1_info = {'word_count': len(chunk1)}
        chunk2_info = {'chunk_id': chunk_id, 'word_count': len(chunk2)}
        
        if has_timestamps:
            last_10 = chunk1[-10:] if len(chunk1) >= 10 else chunk1
            first_10 = chunk2[:10] if len(chunk2) >= 10 else chunk2
            chunk1_info['last_10_words'] = [w.get('text', '') for w in last_10]
            chunk2_info['first_10_words'] = [w.get('text', '') for w in first_10]
            if chunk1:
                chunk1_info['time_range'] = {'start': chunk1[0].get('start', 0), 'end': chunk1[-1].get('end', 0)}
            if chunk2:
                chunk2_info['time_range'] = {'start': chunk2[0].get('start', 0), 'end': chunk2[-1].get('end', 0)}
        else:
            chunk1_info['last_10_words'] = chunk1[-10:] if len(chunk1) >= 10 else chunk1
            chunk2_info['first_10_words'] = chunk2[:10] if len(chunk2) >= 10 else chunk2
        
        return chunk1_info, chunk2_info
    
    def _save_debug_input(self, chunks: List[Dict[str, Any]], has_timestamps: bool) -> None:
        """Save input chunks for debugging."""
        if not self.debug_dir:
            return
        
        json_data = {
            'total_chunks': len(chunks),
            'has_timestamps': has_timestamps,
            'chunks': []
        }
        
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.get('chunk_id'),
                'word_count': len(chunk['words'])
            }
            if has_timestamps and chunk['words']:
                chunk_data['words'] = chunk['words']
                chunk_data['time_range'] = {
                    'start': chunk['words'][0].get('start', 0),
                    'end': chunk['words'][-1].get('end', 0)
                }
            else:
                chunk_data['words'] = chunk['words']
            json_data['chunks'].append(chunk_data)
        
        with open(self.debug_dir / "00_input_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_debug_stitch_step(self, step_num: int, chunk1_info: Dict, chunk2_info: Dict,
                                 overlap_info: Dict, result_info: Dict) -> None:
        """Save debug info for a single stitching step."""
        if not self.debug_dir:
            return
        
        json_data = {
            'step': step_num,
            'chunk1': chunk1_info,
            'chunk2': chunk2_info,
            'overlap_detection': overlap_info,
            'result': result_info
        }
        
        with open(self.debug_dir / f"step_{step_num:03d}_stitch.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_debug_output(self, result: Union[str, List[WordSegment]], has_timestamps: bool) -> None:
        """Save final stitched output for debugging."""
        if not self.debug_dir:
            return
        
        if has_timestamps:
            words_list = [{'text': w.text, 'start': w.start, 'end': w.end, 'speaker': w.speaker} for w in result]
            json_data = {'total_words': len(words_list), 'has_timestamps': True, 'words': words_list}
        else:
            words = result.split() if isinstance(result, str) else result
            json_data = {'total_words': len(words), 'has_timestamps': False, 'text': result}
        
        with open(self.debug_dir / "99_final_output.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_debug_session_summary(self, chunks: List[Dict[str, Any]], has_timestamps: bool,
                                     total_stitch_steps: int, final_word_count: int) -> None:
        """Save session summary for debugging."""
        if not self.debug_dir:
            return
        
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'min_overlap_ratio': self.min_overlap_ratio,
                'similarity_threshold': self.similarity_threshold,
                'sequence_alignment_window': self.sequence_alignment_window,
                'min_sequence_match_length': self.min_sequence_match_length
            },
            'input': {
                'total_chunks': len(chunks),
                'has_timestamps': has_timestamps,
                'total_input_words': sum(len(c['words']) for c in chunks)
            },
            'processing': {'stitch_steps': total_stitch_steps},
            'output': {'final_word_count': final_word_count}
        }
        
        with open(self.debug_dir / "session_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def stitch_chunks(chunks: List[Dict[str, Any]], **kwargs) -> Union[str, List[WordSegment]]:
    """
    Convenience function to stitch transcript chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_id' and 'words' keys
               Words can be either List[str] or List[Dict] with timestamps
        **kwargs: Additional arguments:
            - min_overlap_ratio: Minimum ratio of overlapping words (default 0.6)
            - similarity_threshold: Threshold for word similarity (default 0.7)
            - intermediate_dir: Path for debug files
            - sequence_alignment_window: Words to compare for sequence alignment (default 20)
            - min_sequence_match_length: Minimum matching words for alignment (default 3)
            - skip_single_chunk_debug: Skip debug for single chunks (default True)
            - use_timestamped_debug_dir: Create timestamped subdirectory (default True)
        
    Returns:
        If words are strings: Stitched transcript as a string
        If words have timestamps: List[WordSegment] with preserved timestamps
    """
    stitcher = ChunkStitcher(
        min_overlap_ratio=kwargs.get('min_overlap_ratio', 0.6),
        similarity_threshold=kwargs.get('similarity_threshold', 0.7),
        intermediate_dir=kwargs.get('intermediate_dir'),
        sequence_alignment_window=kwargs.get('sequence_alignment_window', 20),
        min_sequence_match_length=kwargs.get('min_sequence_match_length', 3),
        skip_single_chunk_debug=kwargs.get('skip_single_chunk_debug', True),
        use_timestamped_debug_dir=kwargs.get('use_timestamped_debug_dir', True)
    )
    return stitcher.stitch_chunks(chunks)
