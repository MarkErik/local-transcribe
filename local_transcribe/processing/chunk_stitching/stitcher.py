#!/usr/bin/env python3
"""
Main chunk stitcher class for stitching overlapping transcript chunks.

This module provides the ChunkStitcher class which handles the core logic
of stitching overlapping chunks from audio transcription.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from local_transcribe.processing.chunk_stitching.core import (
    ChunkStitcherConfig,
    OverlapResult,
)
from local_transcribe.processing.chunk_stitching.word_utils import (
    get_word_texts,
    has_timestamps,
)
from local_transcribe.processing.chunk_stitching.overlap_strategies import OverlapDetector
from local_transcribe.processing.chunk_stitching.debug import (
    StitcherDebugWriter,
    capture_chunk_info,
)
from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_progress, get_output_context


class ChunkStitcher:
    """
    A class for stitching overlapping transcript chunks with overlap detection.
    
    Supports both:
    - Simple string words: chunks with "words" as List[str]
    - Timestamped words: chunks with "words" as List[Dict] with "text", "start", "end" keys
    """
    
    def __init__(
        self,
        min_overlap_ratio: float = 0.6,
        similarity_threshold: float = 0.7,
        intermediate_dir: Optional[Path] = None,
        sequence_alignment_window: int = 20,
        min_sequence_match_length: int = 3,
        skip_single_chunk_debug: bool = True,
        use_timestamped_debug_dir: bool = True,
    ):
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
        self.config = ChunkStitcherConfig(
            min_overlap_ratio=min_overlap_ratio,
            similarity_threshold=similarity_threshold,
            sequence_alignment_window=sequence_alignment_window,
            min_sequence_match_length=min_sequence_match_length,
            skip_single_chunk_debug=skip_single_chunk_debug,
            use_timestamped_debug_dir=use_timestamped_debug_dir,
        )
        
        # Initialize overlap detector
        self.overlap_detector = OverlapDetector(
            min_overlap_ratio=min_overlap_ratio,
            similarity_threshold=similarity_threshold,
            sequence_alignment_window=sequence_alignment_window,
            min_sequence_match_length=min_sequence_match_length,
        )
        
        # Setup debug writer if DEBUG logging is enabled
        self._debug_enabled = get_output_context().should_log("DEBUG")
        self.debug_writer: Optional[StitcherDebugWriter] = None
        
        if self._debug_enabled and intermediate_dir:
            self.debug_writer = StitcherDebugWriter(
                intermediate_dir=intermediate_dir,
                config=self.config,
                use_timestamped_subdir=use_timestamped_debug_dir,
            )
        
        # Expose intermediate_dir and debug_dir for backward compatibility
        self.intermediate_dir = intermediate_dir
        self.debug_dir = self.debug_writer.debug_dir if self.debug_writer else None
    
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
        words_have_timestamps = False
        for chunk in chunks:
            if chunk["words"]:
                words_have_timestamps = has_timestamps(chunk["words"])
                break
        
        # Check if we should save debug output
        save_debug = bool(self.debug_writer and self.debug_writer.should_save(len(chunks)))
        
        # Save debug input (only if we should)
        if save_debug and self.debug_writer:
            self.debug_writer.save_input(chunks, words_have_timestamps)
        
        if len(chunks) == 1:
            return self._finalize_result(
                chunks[0]["words"], chunks, words_have_timestamps, 0, save_debug
            )
        
        # Start with the first chunk
        stitched_words = list(chunks[0]["words"])
        
        log_progress(
            f"Processing {len(chunks)} chunks "
            f"({'timestamped' if words_have_timestamps else 'string'} words)"
        )
        
        # Iteratively stitch each subsequent chunk
        for i in range(1, len(chunks)):
            log_progress(f"Processing chunk {i + 1} of {len(chunks)}")
            current_chunk_words = chunks[i]["words"]
            chunk_id = chunks[i].get('chunk_id', i + 1)
            
            stitched_words = self._stitch_two_chunks(
                stitched_words, 
                current_chunk_words,
                words_have_timestamps=words_have_timestamps,
                step_num=i,
                chunk_id=chunk_id,
                save_debug=save_debug
            )
        
        log_progress(f"Stitch complete: {len(stitched_words)} words total")
        
        return self._finalize_result(
            stitched_words, chunks, words_have_timestamps, len(chunks) - 1, save_debug
        )
    
    def _stitch_two_chunks(
        self,
        chunk1: List[Any],
        chunk2: List[Any],
        words_have_timestamps: bool,
        step_num: int = 0,
        chunk_id: Any = None,
        save_debug: bool = True,
    ) -> List[Any]:
        """
        Stitch two chunks, handling overlaps.
        
        Uses a cascade of overlap detection strategies:
        1. Temporal overlap (if timestamps available)
        2. Exact/fuzzy matching
        3. Sequence alignment (handles insertions/deletions)
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            words_have_timestamps: Whether words have timestamp information
            step_num: Step number for debug output
            chunk_id: Chunk ID for debug output
            save_debug: Whether to save debug output
            
        Returns:
            Stitched word list
        """
        # Capture debug info before stitching
        if save_debug:
            chunk1_info = capture_chunk_info(chunk1, None, words_have_timestamps, is_first_chunk=True)
            chunk2_info = capture_chunk_info(chunk2, chunk_id, words_have_timestamps, is_first_chunk=False)
        
        # Try to find overlap using cascade of strategies
        overlap_result = self.overlap_detector.find_best_overlap(
            chunk1, chunk2, words_have_timestamps
        )
        
        # Build result
        # The overlap region is at the END of chunk1 and the START of chunk2
        # We keep chunk1 completely and skip the overlapping portion from chunk2
        if not overlap_result.found:
            log_progress("No overlap detected between chunks; concatenating directly")
            result = chunk1 + chunk2
            words_from_chunk1 = len(chunk1)
            words_from_chunk2 = len(chunk2)
        elif overlap_result.words_to_skip_in_chunk2 >= len(chunk2):
            # Second chunk is entirely contained in first (all overlap)
            result = list(chunk1)
            words_from_chunk1 = len(chunk1)
            words_from_chunk2 = 0
        else:
            overlapping_words = get_word_texts(
                chunk1[overlap_result.overlap_start:overlap_result.overlap_start + overlap_result.overlap_length]
            )
            log_progress(
                f"Overlap found: start={overlap_result.overlap_start}, "
                f"length={overlap_result.overlap_length}, words={overlapping_words}"
            )
            # Keep all of chunk1, skip the overlapping words from chunk2
            result = list(chunk1) + chunk2[overlap_result.words_to_skip_in_chunk2:]
            words_from_chunk1 = len(chunk1)
            words_from_chunk2 = len(chunk2) - overlap_result.words_to_skip_in_chunk2
        
        # Capture and save debug info (only if enabled)
        if save_debug and self.debug_writer:
            overlapping_words = (
                get_word_texts(
                    chunk1[overlap_result.overlap_start:overlap_result.overlap_start + overlap_result.overlap_length]
                )
                if overlap_result.found else []
            )
            
            result_time_range = None
            if words_have_timestamps and result:
                result_time_range = {
                    'start': result[0].get('start', 0),
                    'end': result[-1].get('end', 0)
                }
            
            self.debug_writer.save_stitch_step(
                step_num=step_num,
                chunk1_info=chunk1_info,
                chunk2_info=chunk2_info,
                overlap_result=overlap_result,
                result_word_count=len(result),
                words_from_chunk1=words_from_chunk1,
                words_from_chunk2=words_from_chunk2,
                overlapping_words=overlapping_words,
                result_time_range=result_time_range,
            )
        
        return result
    
    def _finalize_result(
        self,
        words: List[Any],
        chunks: List[Dict],
        words_have_timestamps: bool,
        stitch_steps: int,
        save_debug: bool = True,
    ) -> Union[str, List[WordSegment]]:
        """
        Convert final word list to appropriate output format.
        
        Args:
            words: Final word list
            chunks: Original chunks (for debug summary)
            words_have_timestamps: Whether words have timestamps
            stitch_steps: Number of stitch steps performed
            save_debug: Whether to save debug output
            
        Returns:
            String transcript or List[WordSegment] depending on input format
        """
        if words_have_timestamps:
            result = [
                WordSegment(
                    text=w["text"], 
                    start=w["start"], 
                    end=w["end"], 
                    speaker=w.get("speaker")
                )
                for w in words
            ]
            if save_debug and self.debug_writer:
                self.debug_writer.save_output(result, words_have_timestamps=True)
        else:
            result = " ".join(words)
            if save_debug and self.debug_writer:
                self.debug_writer.save_output(result, words_have_timestamps=False)
        
        if save_debug and self.debug_writer:
            self.debug_writer.save_session_summary(
                chunks, words_have_timestamps, stitch_steps, len(words)
            )
        
        return result

    # =========================================================================
    # Backward compatibility: expose internal methods used by original API
    # =========================================================================
    
    def _get_word_text(self, word):
        """Extract text from a word (backward compatibility)."""
        from local_transcribe.processing.chunk_stitching.word_utils import get_word_text
        return get_word_text(word)
    
    def _get_word_texts(self, words):
        """Extract text from a list of words (backward compatibility)."""
        return get_word_texts(words)
    
    def _has_timestamps(self, words):
        """Check if words have timestamp information (backward compatibility)."""
        return has_timestamps(words)


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
