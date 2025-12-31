"""
Fallback alignment strategies when MFA alignment is unavailable.

Provides simple even-distribution alignment as a fallback mechanism.
"""

from typing import Optional, List, Dict, Any


class FallbackAligner:
    """
    Fallback alignment provider for when MFA alignment fails or is unavailable.
    
    Creates simple even-distribution alignments based on word count and duration.
    """
    
    DEFAULT_SAMPLE_RATE = 16000  # 16kHz
    DEFAULT_WORDS_PER_SECOND = 2.0  # Average speaking rate
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Initialize FallbackAligner.
        
        Args:
            sample_rate: Audio sample rate for duration calculations
        """
        self.sample_rate = sample_rate
    
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
        
        Distributes time evenly among all words in the transcript.
        
        Args:
            transcript: The transcript text to align
            segment_start_time: Start time of the segment in seconds
            segment_duration: Duration of the segment in seconds (optional)
            speaker: Speaker identifier (optional)
            audio_wav: Audio waveform array for duration calculation (optional)
            
        Returns:
            List of word dictionaries with text, start, end, and speaker
        """
        words = transcript.split()
        if not words:
            return []
        
        # Calculate duration from available information
        duration = self._calculate_duration(words, segment_duration, audio_wav)
        
        # Distribute time evenly among words
        word_duration = duration / len(words)
        
        word_dicts = []
        current_time = segment_start_time
        
        for word in words:
            word_end = current_time + word_duration
            word_dicts.append({
                "text": word,
                "start": round(current_time, 2),
                "end": round(word_end, 2),
                "speaker": speaker
            })
            current_time = word_end
        
        return word_dicts
    
    def _calculate_duration(
        self, 
        words: List[str], 
        segment_duration: Optional[float],
        audio_wav: Optional[Any]
    ) -> float:
        """
        Calculate segment duration from available information.
        
        Args:
            words: List of words in the segment
            segment_duration: Explicit duration if provided
            audio_wav: Audio waveform array
            
        Returns:
            Duration in seconds
        """
        if segment_duration is not None:
            return segment_duration
        elif audio_wav is not None:
            return len(audio_wav) / self.sample_rate
        else:
            # Estimate based on word count (average speaking rate)
            return len(words) / self.DEFAULT_WORDS_PER_SECOND
    
    def interpolate_timestamps(
        self, 
        prev_end: float, 
        next_start: float, 
        num_words: int,
        words: List[str], 
        speaker: Optional[str] = None,
        min_word_duration: float = 0.02
    ) -> List[Dict[str, Any]]:
        """
        Create word dicts with interpolated timestamps for words that were missed.
        
        Used when alignment produces gaps that need to be filled with words
        from the original transcript.
        
        Args:
            prev_end: End time of the previous word
            next_start: Start time of the next word
            num_words: Number of words to interpolate
            words: List of word texts
            speaker: Speaker identifier (optional)
            min_word_duration: Minimum duration per word
            
        Returns:
            List of word dictionaries with interpolated timestamps
        """
        if num_words == 0 or not words:
            return []
        
        # Calculate available time window
        available_duration = next_start - prev_end
        if available_duration <= 0:
            # No time available, use minimum duration per word
            available_duration = max(min_word_duration * num_words, min_word_duration)
        
        # Distribute time evenly among words
        word_duration = available_duration / num_words
        
        result = []
        current_time = prev_end
        
        for word in words:
            word_end = current_time + word_duration
            result.append({
                "text": word,
                "start": round(current_time, 2),
                "end": round(word_end, 2),
                "speaker": speaker
            })
            current_time = word_end
        
        return result
