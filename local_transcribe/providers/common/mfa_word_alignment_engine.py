"""
MFA-specific Word Alignment Engine for aligning transcribed text with audio segments.

This module provides an MFA-specific subclass of WordAlignmentEngine with enhanced
functionality for MFA-specific word validation, cleanup, and configuration.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import pathlib
from .word_alignment_engine import WordAlignmentEngine


class MFAWordAlignmentEngine(WordAlignmentEngine):
    """
    MFA-specific subclass of WordAlignmentEngine with enhanced functionality.
    
    This class provides MFA-specific configuration and word processing capabilities
    including silence token filtering, duration validation, and enhanced TextGrid parsing.
    """
    
    # MFA-specific configuration constants
    DEFAULT_MIN_DURATION = 0.01  # 10ms minimum word duration for MFA
    DEFAULT_MAX_GAP = 0.1  # 100ms maximum gap between words
    DEFAULT_SILENCE_TOKENS = ['<eps>', 'sil', 'sp', 'spn', 'SIL', 'SP', 'SPN']
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MFAWordAlignmentEngine.
        
        Args:
            logger: Optional logger instance for logging messages
            config: Optional configuration dictionary for customizing alignment behavior.
                    Supported keys:
                    - min_duration: Minimum duration for words in seconds (default: 0.01)
                    - max_gap: Maximum gap between words in seconds (default: 0.1)
                    - silence_tokens: List of silence tokens to filter (default: ['<eps>', 'sil', 'sp', 'spn'])
                    - gap_penalty: Penalty for gaps in alignment (default: -0.5)
                    - min_similarity_threshold: Minimum similarity for word matching (default: 0.6)
                    - min_word_duration: Minimum duration for words in seconds (default: 0.02)
                    - max_fallback_similarity: Maximum similarity for fallback alignment (default: 0.8)
        """
        super().__init__(logger, config)
        
        # Apply MFA-specific configuration with defaults
        self.min_duration = self.config.get('min_duration', self.DEFAULT_MIN_DURATION)
        self.max_gap = self.config.get('max_gap', self.DEFAULT_MAX_GAP)
        self.silence_tokens = self.config.get('silence_tokens', self.DEFAULT_SILENCE_TOKENS)
        
        self.logger.info(f"MFAWordAlignmentEngine initialized with MFA-specific config: "
                        f"min_duration={self.min_duration}, max_gap={self.max_gap}, "
                        f"silence_tokens={self.silence_tokens}")
    
    def parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str,
                                    segment_start_time: float = 0.0, segment_end_time: float = 0.0,
                                    speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse MFA TextGrid and return list of word dicts with timestamps.
        
        Enhanced version with MFA-specific validation and cleanup.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript text for word mapping
            segment_start_time: Start time of the audio segment
            segment_end_time: End time of the audio segment
            speaker: Optional speaker identifier
            
        Returns:
            List of word dictionaries with timestamps and text
        """
        # Call parent method to get basic parsing
        word_dicts = super().parse_textgrid_to_word_dicts(
            textgrid_path, original_transcript, segment_start_time, segment_end_time, speaker
        )
        
        # Apply MFA-specific post-processing
        if word_dicts:
            word_dicts = self._post_process_mfa_words(word_dicts, segment_start_time, segment_end_time)
        
        return word_dicts
    
    def _post_process_mfa_words(self, word_dicts: List[Dict[str, Any]], 
                               segment_start_time: float, segment_end_time: float) -> List[Dict[str, Any]]:
        """
        Apply MFA-specific word validation and cleanup.
        
        This method filters out silence tokens, validates word durations,
        ensures timestamps are within segment bounds, and handles gaps.
        
        Args:
            word_dicts: List of word dictionaries from TextGrid parsing
            segment_start_time: Start time of the audio segment
            segment_end_time: End time of the audio segment
            
        Returns:
            List of cleaned and validated word dictionaries
        """
        if not word_dicts:
            return []
        
        filtered_words = []
        prev_end_time = segment_start_time
        
        for word_dict in word_dicts:
            # Skip silence tokens
            if word_dict.get('text', '').lower() in [token.lower() for token in self.silence_tokens]:
                self.logger.debug(f"Filtered out silence token: '{word_dict['text']}'")
                continue
            
            # Validate word duration
            duration = word_dict.get('end', 0.0) - word_dict.get('start', 0.0)
            if duration < self.min_duration:
                self.logger.debug(f"Filtered out word with duration {duration:.3f}s below threshold: '{word_dict['text']}'")
                continue
            
            # Ensure timestamps are within segment bounds
            start_time = max(word_dict.get('start', 0.0), segment_start_time)
            end_time = min(word_dict.get('end', 0.0), segment_end_time)
            
            # Skip if word is completely outside segment bounds
            if start_time >= end_time:
                self.logger.debug(f"Filtered out word outside segment bounds: '{word_dict['text']}'")
                continue
            
            # Check for excessive gaps between words
            gap = start_time - prev_end_time
            if gap > self.max_gap:
                self.logger.debug(f"Large gap detected: {gap:.3f}s between words")
                # Note: We don't filter out words due to gaps, but log this for analysis
            
            # Update word dictionary with validated timestamps
            validated_word = {
                "text": word_dict.get('text', ''),
                "start": start_time,
                "end": end_time,
                "speaker": word_dict.get('speaker')
            }
            
            filtered_words.append(validated_word)
            prev_end_time = end_time
        
        # Log processing results
        self.logger.info(f"MFA word processing: {len(word_dicts)} -> {len(filtered_words)} words "
                        f"(filtered {len(word_dicts) - len(filtered_words)} invalid words)")
        
        return filtered_words
    
    def align_words(self, audio_segments: List[Dict[str, Any]], transcript: str) -> List[Dict[str, Any]]:
        """
        Align words in transcript with audio segments using MFA-specific logic.
        
        Args:
            audio_segments: List of audio segment dictionaries with timing information
            transcript: The transcribed text to align
            
        Returns:
            List of aligned word dictionaries with timing and confidence information
        """
        # For MFA, we typically rely on the TextGrid parsing rather than implementing
        # custom alignment logic here. This method can be extended if needed.
        self.logger.info("MFAWordAlignmentEngine.align_words called - using TextGrid-based alignment")
        
        # Create simple alignment as fallback
        if not audio_segments:
            self.logger.warning("No audio segments provided for alignment")
            return []
        
        # Use the first segment for alignment
        segment = audio_segments[0]
        return self.create_simple_alignment(
            transcript, 
            segment.get('start', 0.0), 
            segment.get('end', 0.0) - segment.get('start', 0.0),
            segment.get('speaker')
        )