"""
Word replacement utilities for MFA alignment.

Handles replacing MFA-normalized words with original transcript words
while preserving timing information from the alignment.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from local_transcribe.providers.common.mfa_alignment.sequence_alignment import SequenceAligner
from local_transcribe.providers.common.mfa_alignment.fallback_alignment import FallbackAligner


class WordReplacer:
    """
    Replaces aligned words with original transcript text while preserving timestamps.
    
    Uses sequence alignment to match MFA words to original words and handles
    cases where word counts differ due to MFA splitting or merging.
    """
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None,
        sequence_aligner: Optional[SequenceAligner] = None,
        fallback_aligner: Optional[FallbackAligner] = None,
        min_word_duration: float = 0.02
    ):
        """
        Initialize WordReplacer.
        
        Args:
            logger: Logger instance for debug output
            sequence_aligner: SequenceAligner instance (optional)
            fallback_aligner: FallbackAligner instance (optional)
            min_word_duration: Minimum word duration in seconds
        """
        self.logger = logger or logging.getLogger(__name__)
        self.sequence_aligner = sequence_aligner or SequenceAligner()
        self.fallback_aligner = fallback_aligner or FallbackAligner()
        self.min_word_duration = min_word_duration
    
    def replace_words_with_original_text(
        self, 
        word_dicts: List[Dict[str, Any]],
        original_transcript: str,
        segment_start_time: float,
        segment_end_time: float,
        speaker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Replace alignment engine words with original text, using alignment timestamps.
        
        Handles three cases:
        1. Word counts match - direct positional replacement
        2. Word counts differ - uses sequence alignment
        3. Alignment fails - falls back to simple distribution
        
        Args:
            word_dicts: List of word dictionaries from MFA
            original_transcript: Original transcript text
            segment_start_time: Start time of segment
            segment_end_time: End time of segment
            speaker: Speaker identifier (optional)
            
        Returns:
            List of word dictionaries with original text and MFA timestamps
        """
        self.logger.info("Replacing alignment words with original text")
        
        original_words = original_transcript.split()
        mfa_count = len(word_dicts)
        original_count = len(original_words)
        
        self.logger.debug(f"Original word count: {original_count}, Alignment word count: {mfa_count}")
        
        # Fast path: word counts match - direct positional replacement
        if mfa_count == original_count:
            self.logger.debug("Word counts match - using direct positional replacement")
            return self._create_direct_replacement(word_dicts, original_words, speaker)
        
        # Complex case: counts differ - need sequence alignment
        self.logger.debug(f"Word counts differ ({original_count} vs {mfa_count}) - using sequence alignment")
        alignment = self.sequence_aligner.align_word_sequences(original_words, word_dicts)
        
        self.logger.debug(f"Alignment computed with {len(alignment)} entries")
        
        # Process alignment to build result
        return self._process_alignment_result(
            alignment, word_dicts, original_words,
            segment_start_time, segment_end_time, speaker
        )
    
    def _create_direct_replacement(
        self, 
        word_dicts: List[Dict[str, Any]],
        original_words: List[str],
        speaker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create direct positional replacement when word counts match.
        
        Args:
            word_dicts: List of MFA word dictionaries
            original_words: List of original words
            speaker: Speaker identifier
            
        Returns:
            List of word dictionaries with original text
        """
        result = []
        for i, word_dict in enumerate(word_dicts):
            result.append({
                "text": original_words[i],
                "start": word_dict["start"],
                "end": word_dict["end"],
                "speaker": word_dict.get("speaker", speaker)
            })
        return result
    
    def _process_alignment_result(
        self, 
        alignment: List[Tuple[Optional[int], Optional[int]]], 
        word_dicts: List[Dict[str, Any]],
        original_words: List[str], 
        segment_start_time: float,
        segment_end_time: float, 
        speaker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process alignment result to handle gaps and merges.
        
        Handles:
        - Matched pairs: use MFA timing with original text
        - Source gaps: interpolate timestamps for missing words
        - Target gaps: merge MFA words into single original word
        
        Args:
            alignment: List of (source_idx, target_idx) pairs
            word_dicts: MFA word dictionaries
            original_words: Original transcript words
            segment_start_time: Start time of segment
            segment_end_time: End time of segment
            speaker: Speaker identifier
            
        Returns:
            List of word dictionaries
        """
        result = []
        pending_original_words = []  # Words waiting for timestamps
        last_end_time = segment_start_time
        
        i = 0
        while i < len(alignment):
            orig_idx, mfa_idx = alignment[i]
            
            if orig_idx is not None and mfa_idx is not None:
                # Matched pair - first handle any pending words
                if pending_original_words:
                    current_start = word_dicts[mfa_idx]["start"]
                    interpolated = self.fallback_aligner.interpolate_timestamps(
                        last_end_time, current_start,
                        len(pending_original_words), pending_original_words, 
                        speaker, self.min_word_duration
                    )
                    result.extend(interpolated)
                    pending_original_words = []
                
                # Check if next alignments are target gaps that should merge
                mfa_start_idx = mfa_idx
                mfa_end_idx = mfa_idx + 1
                
                j = i + 1
                while j < len(alignment):
                    next_orig, next_mfa = alignment[j]
                    if next_orig is None and next_mfa is not None:
                        # Target extra word - merge it
                        mfa_end_idx = next_mfa + 1
                        j += 1
                    else:
                        break
                
                # Get merged timestamps
                start_time, end_time = self._merge_mfa_timestamps(
                    word_dicts, mfa_start_idx, mfa_end_idx
                )
                
                result.append({
                    "text": original_words[orig_idx],
                    "start": start_time,
                    "end": end_time,
                    "speaker": word_dicts[mfa_idx].get("speaker", speaker)
                })
                
                last_end_time = end_time
                i = j  # Skip merged target words
                
            elif orig_idx is not None and mfa_idx is None:
                # Original word with no target match - queue for interpolation
                pending_original_words.append(original_words[orig_idx])
                i += 1
                
            elif orig_idx is None and mfa_idx is not None:
                # Target extra word with no original match
                self.logger.debug(
                    f"Unexpected target extra word at position {mfa_idx}: "
                    f"'{word_dicts[mfa_idx]['text']}'"
                )
                i += 1
                
            else:
                # Both None - shouldn't happen
                i += 1
        
        # Handle any remaining pending words at the end
        if pending_original_words:
            interpolated = self.fallback_aligner.interpolate_timestamps(
                last_end_time, segment_end_time,
                len(pending_original_words), pending_original_words, 
                speaker, self.min_word_duration
            )
            result.extend(interpolated)
        
        # Final validation
        if len(result) != len(original_words):
            self.logger.warning(
                f"Word count mismatch after alignment: expected {len(original_words)}, "
                f"got {len(result)}. Falling back to simple distribution."
            )
            duration = segment_end_time - segment_start_time
            return self.fallback_aligner.create_simple_alignment(
                " ".join(original_words), segment_start_time, duration, speaker, None
            )
        
        return result
    
    def _merge_mfa_timestamps(
        self, 
        mfa_words: List[Dict[str, Any]], 
        start_idx: int, 
        end_idx: int
    ) -> Tuple[float, float]:
        """
        Get merged start/end times from a range of MFA words.
        
        Args:
            mfa_words: List of MFA word dictionaries
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            Tuple of (start_time, end_time)
        """
        if start_idx >= end_idx or start_idx >= len(mfa_words):
            return (0.0, 0.0)
        
        start_time = mfa_words[start_idx]["start"]
        end_time = mfa_words[min(end_idx - 1, len(mfa_words) - 1)]["end"]
        
        return (start_time, end_time)
    
    def replace_unk_with_original(
        self, 
        word_dicts: List[Dict[str, Any]], 
        original_transcript: str
    ) -> None:
        """
        Replace <unk> tokens in word dicts with words from the original transcript.
        
        Uses two-pointer alignment to match <unk> tokens with their
        corresponding original words. Modifies word_dicts in place.
        
        Args:
            word_dicts: List of word dictionaries (modified in place)
            original_transcript: Original transcript text
        """
        if not word_dicts:
            self.logger.debug("No word dicts to process for UNK replacement")
            return
        
        aligned_texts = [wd["text"] for wd in word_dicts]
        original_words = original_transcript.split()
        
        self.logger.debug(f"[UNK REPLACE] Starting <unk> replacement")
        self.logger.debug(f"[UNK REPLACE] Original transcript word count: {len(original_words)}")
        self.logger.debug(f"[UNK REPLACE] Aligned word count before replacement: {len(aligned_texts)}")
        
        if len(aligned_texts) > 20:
            self.logger.debug(
                f"[UNK REPLACE] Aligned texts ({len(aligned_texts)} words): "
                f"{' '.join(aligned_texts[:10])} ... {' '.join(aligned_texts[-10:])}"
            )
        else:
            self.logger.debug(f"[UNK REPLACE] Aligned texts: {' '.join(aligned_texts)}")
        
        if len(original_words) > 20:
            self.logger.debug(
                f"[UNK REPLACE] Original words ({len(original_words)} words): "
                f"{' '.join(original_words[:10])} ... {' '.join(original_words[-10:])}"
            )
        else:
            self.logger.debug(f"[UNK REPLACE] Original words: {' '.join(original_words)}")
        
        # Two-pointer alignment: ptr tracks position in original_words
        ptr = 0
        replacements_made = 0
        
        for i, word_dict in enumerate(word_dicts):
            if word_dict["text"] == "<unk>":
                # Debug: show context around the <unk> token
                start_idx = max(0, i - 5)
                end_idx = min(len(aligned_texts), i + 6)
                aligned_context = aligned_texts[start_idx:end_idx]
                
                orig_start = max(0, ptr - 5)
                orig_end = min(len(original_words), ptr + 6)
                original_context = original_words[orig_start:orig_end]
                
                self.logger.debug(
                    f"[UNK REPLACE] Replacing <unk> at position {i}: "
                    f"Aligned context: {' '.join(aligned_context)} | "
                    f"Original context around ptr {ptr}: {' '.join(original_context)}"
                )
                
                if ptr < len(original_words):
                    replacement = original_words[ptr]
                    word_dict["text"] = replacement
                    self.logger.debug(f"[UNK REPLACE] Replaced with: '{replacement}'")
                    ptr += 1
                    replacements_made += 1
                else:
                    self.logger.debug(
                        f"[UNK REPLACE] No more original words available, leaving as <unk>"
                    )
            else:
                # Check if current aligned word matches the current position in original
                if ptr < len(original_words):
                    aligned_normalized = word_dict["text"].lower()
                    original_normalized = original_words[ptr].lower()
                    
                    if aligned_normalized == original_normalized:
                        self.logger.debug(
                            f"[UNK REPLACE] Matched '{word_dict['text']}' with "
                            f"original '{original_words[ptr]}', advancing ptr to {ptr+1}"
                        )
                        ptr += 1
                    else:
                        self.logger.debug(
                            f"[UNK REPLACE] No match for '{word_dict['text']}' at ptr {ptr} "
                            f"(expected '{original_words[ptr]}'), not advancing ptr"
                        )
        
        # Final logging
        final_texts = [wd["text"] for wd in word_dicts]
        self.logger.debug(f"[UNK REPLACE] Completed: made {replacements_made} replacements")
        self.logger.debug(f"[UNK REPLACE] Aligned word count after replacement: {len(final_texts)}")
        
        if len(final_texts) > 20:
            self.logger.debug(
                f"[UNK REPLACE] Final aligned texts: "
                f"{' '.join(final_texts[:10])} ... {' '.join(final_texts[-10:])}"
            )
        else:
            self.logger.debug(f"[UNK REPLACE] Final aligned texts: {' '.join(final_texts)}")
