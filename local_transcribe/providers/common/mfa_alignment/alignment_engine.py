"""
MFA Alignment Engine - Main orchestrator for MFA word alignment operations.

This is the primary interface for MFA alignment functionality, composing
the specialized modules for TextGrid parsing, sequence alignment, and
word replacement.
"""

from typing import Optional, Dict, Any, List
import logging
import pathlib

from local_transcribe.providers.common.mfa_alignment.textgrid_parser import TextGridParser
from local_transcribe.providers.common.mfa_alignment.word_similarity import WordSimilarity
from local_transcribe.providers.common.mfa_alignment.sequence_alignment import SequenceAligner
from local_transcribe.providers.common.mfa_alignment.word_replacement import WordReplacer
from local_transcribe.providers.common.mfa_alignment.fallback_alignment import FallbackAligner


class MFAAlignmentEngine:
    """
    Main MFA alignment engine that orchestrates all alignment operations.
    
    This class provides a unified interface for MFA word alignment while
    delegating to specialized components for specific tasks:
    - TextGridParser: Parses MFA TextGrid output files
    - WordSimilarity: Calculates word similarity for alignment
    - SequenceAligner: Dynamic programming sequence alignment
    - WordReplacer: Replaces MFA words with original text
    - FallbackAligner: Simple alignment for fallback cases
    
    Usage:
        engine = MFAAlignmentEngine(logger)
        
        # Parse TextGrid directly
        word_dicts = engine.parse_textgrid_to_word_dicts(
            textgrid_path, transcript, start_time, end_time, speaker
        )
        
        # Replace words with original text
        result = engine.replace_words_with_original_text(
            word_dicts, transcript, start_time, end_time, speaker
        )
    """
    
    # Configuration constants
    DEFAULT_GAP_PENALTY = -0.5
    DEFAULT_MIN_SIMILARITY_THRESHOLD = 0.6
    DEFAULT_MIN_WORD_DURATION = 0.02  # 20ms
    DEFAULT_MAX_GAP = 0.1  # 100ms
    DEFAULT_SILENCE_TOKENS = ['<eps>', 'sil', 'sp', 'spn', 'SIL', 'SP', 'SPN']
    
    def __init__(
        self, 
        logger: Optional[logging.Logger] = None, 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MFAAlignmentEngine.
        
        Args:
            logger: Logger instance for debug output
            config: Configuration dictionary with optional keys:
                - gap_penalty: Penalty for gaps in alignment (default: -0.5)
                - min_similarity_threshold: Minimum word similarity (default: 0.6)
                - min_word_duration: Minimum word duration in seconds (default: 0.02)
                - max_gap: Maximum gap between words (default: 0.1)
                - silence_tokens: List of silence tokens to filter
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Extract configuration with defaults
        self.gap_penalty = self.config.get('gap_penalty', self.DEFAULT_GAP_PENALTY)
        self.min_similarity_threshold = self.config.get(
            'min_similarity_threshold', self.DEFAULT_MIN_SIMILARITY_THRESHOLD
        )
        self.min_word_duration = self.config.get('min_word_duration', self.DEFAULT_MIN_WORD_DURATION)
        self.max_gap = self.config.get('max_gap', self.DEFAULT_MAX_GAP)
        self.silence_tokens = self.config.get('silence_tokens', self.DEFAULT_SILENCE_TOKENS)
        
        # Initialize components
        self.word_similarity = WordSimilarity(
            min_similarity_threshold=self.min_similarity_threshold
        )
        self.fallback_aligner = FallbackAligner()
        self.sequence_aligner = SequenceAligner(
            gap_penalty=self.gap_penalty,
            word_similarity=self.word_similarity
        )
        self.textgrid_parser = TextGridParser(
            logger=self.logger,
            min_word_duration=self.min_word_duration,
            silence_tokens=self.silence_tokens
        )
        self.word_replacer = WordReplacer(
            logger=self.logger,
            sequence_aligner=self.sequence_aligner,
            fallback_aligner=self.fallback_aligner,
            min_word_duration=self.min_word_duration
        )
        
        self.logger.info(
            f"MFAAlignmentEngine initialized with config: gap_penalty={self.gap_penalty}, "
            f"min_similarity_threshold={self.min_similarity_threshold}, "
            f"min_word_duration={self.min_word_duration}, max_gap={self.max_gap}"
        )
    
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
        
        Reads a TextGrid file produced by MFA and extracts word-level
        timing information.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript text for fallback
            segment_start_time: Start time offset for the segment
            segment_end_time: End time of the segment
            speaker: Speaker identifier (optional)
            
        Returns:
            List of word dictionaries with text, start, end, speaker
        """
        word_dicts = self.textgrid_parser.parse_textgrid_to_word_dicts(
            textgrid_path, original_transcript, segment_start_time, segment_end_time, speaker
        )
        
        # Apply MFA-specific post-processing
        if word_dicts:
            word_dicts = self._post_process_mfa_words(
                word_dicts, segment_start_time, segment_end_time
            )
        
        return word_dicts
    
    def _post_process_mfa_words(
        self, 
        word_dicts: List[Dict[str, Any]], 
        segment_start_time: float, 
        segment_end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Apply MFA-specific word validation and cleanup.
        
        Filters out silence tokens, validates durations, and ensures
        timestamps are within segment bounds.
        
        Args:
            word_dicts: List of word dictionaries from TextGrid parser
            segment_start_time: Start time of segment
            segment_end_time: End time of segment
            
        Returns:
            Filtered list of word dictionaries
        """
        if not word_dicts:
            return []
        
        filtered_words = []
        prev_end_time = segment_start_time
        
        for word_dict in word_dicts:
            text = word_dict.get('text', '')
            
            # Skip silence tokens (case-insensitive check)
            if text.lower() in [token.lower() for token in self.silence_tokens]:
                self.logger.debug(f"Filtered out silence token: '{text}'")
                continue
            
            # Validate word duration
            duration = word_dict.get('end', 0.0) - word_dict.get('start', 0.0)
            if duration < self.min_word_duration:
                self.logger.debug(
                    f"Filtered out word with duration {duration:.3f}s below threshold: '{text}'"
                )
                continue
            
            # Ensure timestamps are within segment bounds
            start_time = max(word_dict.get('start', 0.0), segment_start_time)
            end_time = min(word_dict.get('end', 0.0), segment_end_time)
            
            # Skip if word is completely outside segment bounds
            if start_time >= end_time:
                self.logger.debug(f"Filtered out word outside segment bounds: '{text}'")
                continue
            
            # Log large gaps (but don't filter)
            gap = start_time - prev_end_time
            if gap > self.max_gap:
                self.logger.debug(f"Large gap detected: {gap:.3f}s between words")
            
            # Update word dictionary with validated timestamps
            validated_word = {
                "text": text,
                "start": start_time,
                "end": end_time,
                "speaker": word_dict.get('speaker')
            }
            
            filtered_words.append(validated_word)
            prev_end_time = end_time
        
        self.logger.info(
            f"MFA word processing: {len(word_dicts)} -> {len(filtered_words)} words "
            f"(filtered {len(word_dicts) - len(filtered_words)} invalid words)"
        )
        
        return filtered_words
    
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
        
        Args:
            word_dicts: List of word dictionaries from MFA
            original_transcript: Original transcript text
            segment_start_time: Start time of segment
            segment_end_time: End time of segment
            speaker: Speaker identifier (optional)
            
        Returns:
            List of word dictionaries with original text and MFA timestamps
        """
        return self.word_replacer.replace_words_with_original_text(
            word_dicts, original_transcript, segment_start_time, segment_end_time, speaker
        )
    
    def replace_unk_with_original(
        self, 
        word_dicts: List[Dict[str, Any]], 
        original_transcript: str
    ) -> None:
        """
        Replace <unk> tokens in word dicts with words from the original transcript.
        
        Modifies word_dicts in place.
        
        Args:
            word_dicts: List of word dictionaries (modified in place)
            original_transcript: Original transcript text
        """
        self.word_replacer.replace_unk_with_original(word_dicts, original_transcript)
    
    def create_simple_alignment(
        self, 
        transcript: str, 
        segment_start_time: float = 0.0,
        segment_duration: Optional[float] = None,
        speaker: Optional[str] = None,
        audio_wav: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Create simple even-distribution alignment as fallback.
        
        Args:
            transcript: The transcript text to align
            segment_start_time: Start time of the segment
            segment_duration: Duration of the segment (optional)
            speaker: Speaker identifier (optional)
            audio_wav: Audio waveform for duration calculation (optional)
            
        Returns:
            List of word dictionaries with evenly distributed timestamps
        """
        return self.fallback_aligner.create_simple_alignment(
            transcript, segment_start_time, segment_duration, speaker, audio_wav
        )
    
    def align_word_sequences(
        self, 
        source_words: List[str], 
        target_words: List[Dict[str, Any]]
    ) -> List[tuple]:
        """
        Align source and target word sequences using dynamic programming.
        
        Args:
            source_words: List of words from original transcript
            target_words: List of word dictionaries from MFA
            
        Returns:
            List of alignment pairs (source_idx, target_idx)
        """
        return self.sequence_aligner.align_word_sequences(source_words, target_words)
    
    def normalize_word_for_matching(self, word: str) -> str:
        """
        Normalize word for comparison during alignment.
        
        Args:
            word: The word to normalize
            
        Returns:
            Normalized word string
        """
        return self.word_similarity.normalize_word_for_matching(word)
    
    def calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words for alignment purposes.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        return self.word_similarity.calculate_similarity(word1, word2)
    
    def interpolate_timestamps(
        self, 
        prev_end: float, 
        next_start: float, 
        num_words: int,
        words: List[str], 
        speaker: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create word dicts with interpolated timestamps for missed words.
        
        Args:
            prev_end: End time of previous word
            next_start: Start time of next word
            num_words: Number of words to interpolate
            words: List of word texts
            speaker: Speaker identifier (optional)
            
        Returns:
            List of word dictionaries with interpolated timestamps
        """
        return self.fallback_aligner.interpolate_timestamps(
            prev_end, next_start, num_words, words, speaker, self.min_word_duration
        )
    
    def align_words(
        self, 
        audio_segments: List[Dict[str, Any]], 
        transcript: str
    ) -> List[Dict[str, Any]]:
        """
        Align words in transcript with audio segments.
        
        For MFA, this typically uses TextGrid parsing. This method provides
        a simple fallback when no TextGrid is available.
        
        Args:
            audio_segments: List of audio segment dictionaries
            transcript: Transcript text
            
        Returns:
            List of word dictionaries
        """
        self.logger.info("MFAAlignmentEngine.align_words - using simple fallback alignment")
        
        if not audio_segments:
            self.logger.warning("No audio segments provided for alignment")
            return []
        
        # Use the first segment for alignment
        segment = audio_segments[0]
        duration = segment.get('end', 0.0) - segment.get('start', 0.0)
        
        return self.create_simple_alignment(
            transcript, 
            segment.get('start', 0.0), 
            duration,
            segment.get('speaker')
        )


# Backward compatibility alias
MFAWordAlignmentEngine = MFAAlignmentEngine
