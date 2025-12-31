"""
TextGrid parsing for MFA output files.

Handles reading and parsing Praat TextGrid files produced by Montreal Forced Aligner,
extracting word-level timing information.
"""

from typing import Optional, List, Dict, Any, Tuple
import logging
import pathlib

from local_transcribe.providers.common.mfa_alignment.word_similarity import WordSimilarity
from local_transcribe.providers.common.mfa_alignment.fallback_alignment import FallbackAligner


class TextGridParser:
    """
    Parser for Praat TextGrid files produced by MFA.
    
    Extracts word-level timing information from TextGrid files,
    handling various edge cases and providing fallback alignment.
    """
    
    # Default tokens to filter out from TextGrid
    DEFAULT_SILENCE_TOKENS = ["", "<eps>", "sil", "sp", "spn"]
    DEFAULT_MIN_WORD_DURATION = 0.02  # 20ms minimum
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None,
        min_word_duration: float = DEFAULT_MIN_WORD_DURATION,
        silence_tokens: Optional[List[str]] = None
    ):
        """
        Initialize TextGridParser.
        
        Args:
            logger: Logger instance for debug output
            min_word_duration: Minimum word duration in seconds
            silence_tokens: List of tokens to filter out
        """
        self.logger = logger or logging.getLogger(__name__)
        self.min_word_duration = min_word_duration
        self.silence_tokens = silence_tokens or self.DEFAULT_SILENCE_TOKENS
        
        # Components
        self.word_similarity = WordSimilarity()
        self.fallback_aligner = FallbackAligner()
    
    def parse_textgrid_to_word_dicts(
        self, 
        textgrid_path: pathlib.Path, 
        original_transcript: str,
        segment_start_time: float = 0.0, 
        segment_end_time: float = 0.0,
        speaker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse MFA TextGrid and return list of word dicts with timestamps.
        
        Reads a TextGrid file, extracts the word tier, and creates word
        dictionaries with timing information. Falls back to simple alignment
        if parsing fails.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript text for fallback
            segment_start_time: Start time offset for the segment
            segment_end_time: End time of the segment
            speaker: Speaker identifier (optional)
            
        Returns:
            List of word dictionaries with text, start, end, speaker
        """
        try:
            # Read TextGrid file
            lines = self._read_textgrid_file(textgrid_path)
            
            if not lines:
                self.logger.warning("TextGrid file is empty")
                return self._create_fallback_alignment(
                    original_transcript, segment_start_time, segment_end_time, speaker
                )
            
            # Build word mapping for normalization
            original_words = original_transcript.split()
            normalized_to_original = self._build_word_mapping(original_words)
            
            # Find word tier bounds
            word_tier_bounds = self._find_word_tier(lines)
            if not word_tier_bounds:
                self.logger.warning("Could not find word tier in TextGrid")
                return self._create_fallback_alignment(
                    original_transcript, segment_start_time, segment_end_time, speaker
                )
            
            # Parse intervals
            word_dicts = self._parse_textgrid_intervals(
                lines, word_tier_bounds, normalized_to_original, segment_start_time, speaker
            )
            
            # Validate results
            if not word_dicts:
                self.logger.warning("No valid words extracted from TextGrid")
                return self._create_fallback_alignment(
                    original_transcript, segment_start_time, segment_end_time, speaker
                )
            
            # Replace <unk> tokens with words from original transcript
            self._replace_unk_tokens(word_dicts, original_transcript)
            
            return word_dicts
            
        except Exception as e:
            self.logger.error(f"Failed to parse TextGrid {textgrid_path}: {e}")
            return self._create_fallback_alignment(
                original_transcript, segment_start_time, segment_end_time, speaker
            )
    
    def _read_textgrid_file(self, textgrid_path: pathlib.Path) -> List[str]:
        """
        Read TextGrid file with UTF-8 encoding.
        
        Args:
            textgrid_path: Path to the TextGrid file
            
        Returns:
            List of lines from the file
        """
        with open(textgrid_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    
    def _build_word_mapping(self, original_words: List[str]) -> Dict[str, str]:
        """
        Build mapping of normalized words to original words.
        
        Creates a lookup table for matching MFA-normalized words back
        to their original forms with punctuation.
        
        Args:
            original_words: List of original words from transcript
            
        Returns:
            Dictionary mapping normalized forms to original words
        """
        normalized_to_original = {}
        
        for word in original_words:
            normalized = self.word_similarity.normalize_word_for_matching(word)
            if normalized:
                # Store the first occurrence of each normalized word
                if normalized not in normalized_to_original:
                    normalized_to_original[normalized] = word
        
        return normalized_to_original
    
    def _find_word_tier(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        """
        Find the word tier in TextGrid lines.
        
        Scans the file for the "words" tier section and returns its bounds.
        
        Args:
            lines: List of lines from TextGrid file
            
        Returns:
            Tuple of (start_line, end_line) for the word tier, or None if not found
        """
        word_tier_start = None
        word_tier_end = None
        
        for i, line in enumerate(lines):
            # Look for word tier declaration: name = "words"
            if 'name = "words"' in line:
                word_tier_start = i
            # Look for phones tier (marks end of word tier)
            elif word_tier_start is not None and 'name = "phones"' in line:
                word_tier_end = i
                break
        
        # If we found word tier start but not phones tier, use end of file
        if word_tier_start is not None and word_tier_end is None:
            word_tier_end = len(lines)
        
        # Return None if we didn't find the word tier
        if word_tier_start is None:
            return None
            
        return (word_tier_start, word_tier_end)
    
    def _parse_textgrid_intervals(
        self, 
        lines: List[str], 
        word_tier_bounds: Tuple[int, int],
        normalized_to_original: Dict[str, str],
        segment_start_time: float,
        speaker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse intervals from TextGrid word tier.
        
        Looks for the standard TextGrid interval format:
        - intervals [N]:
        - xmin = <time>
        - xmax = <time>
        - text = "<word>"
        
        Args:
            lines: List of lines from TextGrid file
            word_tier_bounds: Tuple of (start_line, end_line)
            normalized_to_original: Word mapping dictionary
            segment_start_time: Time offset for the segment
            speaker: Speaker identifier
            
        Returns:
            List of word dictionaries
        """
        word_dicts = []
        start_line, end_line = word_tier_bounds
        
        i = start_line
        while i < end_line:
            line = lines[i].strip()
            
            # Look for interval start: "intervals [N]:"
            if line.startswith('intervals ['):
                # Move to next lines for xmin, xmax, text
                i += 1
                if i >= end_line:
                    break
                xmin_line = lines[i].strip()
                
                i += 1
                if i >= end_line:
                    break
                xmax_line = lines[i].strip()
                
                i += 1
                if i >= end_line:
                    break
                text_line = lines[i].strip()
                
                try:
                    # Parse timing
                    start = float(xmin_line.split('=')[1].strip())
                    end = float(xmax_line.split('=')[1].strip())
                    
                    # Parse text, removing quotes
                    mfa_text = text_line.split('=')[1].strip().strip('"')
                    
                    # Skip empty words and silence tokens
                    if mfa_text and mfa_text not in self.silence_tokens:
                        # Validate duration
                        if end - start >= self.min_word_duration:
                            word_dicts.append({
                                "text": mfa_text,
                                "start": round(start + segment_start_time, 2),
                                "end": round(end + segment_start_time, 2),
                                "speaker": speaker
                            })
                except (ValueError, IndexError) as e:
                    self.logger.debug(f"Error parsing interval at line {i}: {e}")
            
            i += 1
        
        return word_dicts
    
    def _replace_unk_tokens(
        self, 
        word_dicts: List[Dict[str, Any]], 
        original_transcript: str
    ) -> None:
        """
        Replace <unk> tokens in word dicts with words from the original transcript.
        
        Uses two-pointer alignment to match <unk> tokens with their
        corresponding original words.
        
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
        self.logger.debug(f"[UNK REPLACE] Original word count: {len(original_words)}, Aligned count: {len(aligned_texts)}")
        
        # Two-pointer alignment
        ptr = 0
        replacements_made = 0
        
        for i, word_dict in enumerate(word_dicts):
            if word_dict["text"] == "<unk>":
                if ptr < len(original_words):
                    replacement = original_words[ptr]
                    word_dict["text"] = replacement
                    self.logger.debug(f"[UNK REPLACE] Replaced <unk> at {i} with: '{replacement}'")
                    ptr += 1
                    replacements_made += 1
            else:
                # Check if current aligned word matches the current position in original
                if ptr < len(original_words):
                    aligned_normalized = word_dict["text"].lower()
                    original_normalized = original_words[ptr].lower()
                    
                    if aligned_normalized == original_normalized:
                        ptr += 1
        
        self.logger.debug(f"[UNK REPLACE] Completed: made {replacements_made} replacements")
    
    def _create_fallback_alignment(
        self, 
        original_transcript: str, 
        segment_start_time: float,
        segment_end_time: float, 
        speaker: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Create fallback alignment using simple even distribution.
        
        Args:
            original_transcript: Original transcript text
            segment_start_time: Start time of segment
            segment_end_time: End time of segment
            speaker: Speaker identifier
            
        Returns:
            List of word dictionaries with evenly distributed timestamps
        """
        duration = segment_end_time - segment_start_time
        return self.fallback_aligner.create_simple_alignment(
            original_transcript, segment_start_time, duration, speaker, None
        )
