"""
MFA-specific Word Alignment Engine for aligning transcribed text with audio segments.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import pathlib
from local_transcribe.providers.common.word_alignment_engine import WordAlignmentEngine


class MFAWordAlignmentEngine(WordAlignmentEngine):
    """
    MFA-specific subclass of WordAlignmentEngine.
    """
    
    # MFA-specific configuration constants
    DEFAULT_MIN_DURATION = 0.01  # 10ms minimum word duration for MFA
    DEFAULT_SILENCE_TOKENS = ['<eps>', 'sil', 'sp', 'spn', 'SIL', 'SP', 'SPN']
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MFAWordAlignmentEngine.
        """
        super().__init__(logger, config)
        
        # Apply MFA-specific configuration with defaults
        self.min_duration = self.config.get('min_duration', self.DEFAULT_MIN_DURATION)
        self.silence_tokens = self.config.get('silence_tokens', self.DEFAULT_SILENCE_TOKENS)
        
        self.logger.info(f"MFAWordAlignmentEngine initialized with MFA-specific config: "
                        f"min_duration={self.min_duration}, "
                        f"silence_tokens={self.silence_tokens}")
    
    def parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str,
                                    segment_start_time: float = 0.0, segment_end_time: float = 0.0,
                                    speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse MFA TextGrid and return list of word dicts with timestamps.
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
    