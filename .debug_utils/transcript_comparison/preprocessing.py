"""
Transcript preprocessing module.

This module handles normalization and preparation of transcripts for comparison,
including text normalization, segmentation adjustments, and feature extraction.
"""

import re
from typing import Dict, List, Optional, Tuple

from .data_structures import EnhancedTranscript, TextSegment, TranscriptMetadata


class TranscriptPreprocessor:
    """Handles normalization and preparation of transcripts for comparison."""
    
    def __init__(self, config):
        """Initialize the preprocessor with configuration."""
        self.config = config
        self.compiled_patterns = self._compile_regex_patterns()
    
    def _compile_regex_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for text normalization."""
        return {
            'punctuation': re.compile(r'[^\w\s]'),
            'extra_spaces': re.compile(r'\s+'),
            'leading_trailing_spaces': re.compile(r'^\s+|\s+$'),
            'numbers': re.compile(r'\d+'),
            'abbreviations': re.compile(r'\b(?:mr|mrs|ms|dr|prof|etc)\b', re.IGNORECASE),
        }
    
    def normalize_transcript(self, transcript: EnhancedTranscript) -> EnhancedTranscript:
        """
        Apply text normalization to a transcript.
        
        Args:
            transcript: The transcript to normalize
            
        Returns:
            A new EnhancedTranscript with normalized segments
        """
        normalized_segments = []
        
        for segment in transcript.segments:
            normalized_text = self._normalize_text(segment.text)
            
            # Create a new segment with normalized text
            normalized_segment = TextSegment(
                text=normalized_text,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                segment_id=f"{segment.segment_id}_normalized" if segment.segment_id else "",
                parent_id=segment.segment_id,
                confidence=segment.confidence,
                features=segment.features.copy() if segment.features else {}
            )
            
            # Add normalization flag to features
            normalized_segment.features['normalized'] = True
            
            normalized_segments.append(normalized_segment)
        
        # Create a new transcript with normalized segments
        normalized_transcript = EnhancedTranscript(
            segments=normalized_segments,
            metadata=transcript.metadata,
            features=transcript.features.copy() if transcript.features else {}
        )
        
        # Add normalization flag to transcript features
        normalized_transcript.features['normalized'] = True
        
        return normalized_transcript
    
    def _normalize_text(self, text: str) -> str:
        """
        Apply text normalization transformations.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove punctuation (optional, based on config)
        if getattr(self.config, 'remove_punctuation', True):
            normalized = self.compiled_patterns['punctuation'].sub(' ', normalized)
        
        # Remove extra whitespace
        normalized = self.compiled_patterns['extra_spaces'].sub(' ', normalized)
        normalized = self.compiled_patterns['leading_trailing_spaces'].sub('', normalized)
        
        # Normalize numbers (optional, based on config)
        if getattr(self.config, 'normalize_numbers', True):
            normalized = self.compiled_patterns['numbers'].sub('NUM', normalized)
        
        # Normalize common abbreviations (optional, based on config)
        if getattr(self.config, 'normalize_abbreviations', True):
            normalized = self._normalize_abbreviations(normalized)
        
        return normalized.strip()
    
    def _normalize_abbreviations(self, text: str) -> str:
        """Normalize common abbreviations."""
        abbrev_map = {
            'mr': 'mister',
            'mrs': 'misses',
            'ms': 'miss',
            'dr': 'doctor',
            'prof': 'professor',
            'etc': 'et cetera'
        }
        
        def replace_abbrev(match):
            abbrev = match.group(0).lower()
            return abbrev_map.get(abbrev, abbrev)
        
        return self.compiled_patterns['abbreviations'].sub(replace_abbrev, text)
    
    def adjust_segmentation(self, transcript: EnhancedTranscript) -> EnhancedTranscript:
        """
        Adjust segmentation boundaries for better alignment.
        
        Args:
            transcript: The transcript to adjust
            
        Returns:
            A new EnhancedTranscript with adjusted segmentation
        """
        # Merge very short segments with adjacent segments
        merged_segments = self._merge_short_segments(transcript.segments)
        
        # Split very long segments at natural boundaries
        split_segments = self._split_long_segments(merged_segments)
        
        # Create a new transcript with adjusted segments
        adjusted_transcript = EnhancedTranscript(
            segments=split_segments,
            metadata=transcript.metadata,
            features=transcript.features.copy() if transcript.features else {}
        )
        
        # Add segmentation adjustment flag
        adjusted_transcript.features['segmentation_adjusted'] = True
        
        return adjusted_transcript
    
    def _merge_short_segments(self, segments: List[TextSegment]) -> List[TextSegment]:
        """Merge segments that are shorter than the minimum length."""
        if not segments:
            return []
        
        merged = []
        current_group = [segments[0]]
        
        for segment in segments[1:]:
            # Check if current group is too short
            group_duration = sum(seg.end - seg.start for seg in current_group)
            
            if (group_duration < self.config.min_segment_length and 
                len(current_group) < 3):  # Don't merge too many segments
                current_group.append(segment)
            else:
                # Finalize current group
                merged_segment = self._create_merged_segment(current_group)
                merged.append(merged_segment)
                current_group = [segment]
        
        # Add the last group
        if current_group:
            merged_segment = self._create_merged_segment(current_group)
            merged.append(merged_segment)
        
        return merged
    
    def _split_long_segments(self, segments: List[TextSegment]) -> List[TextSegment]:
        """Split segments that are longer than the maximum length."""
        split = []
        
        for segment in segments:
            duration = segment.end - segment.start
            
            if duration > self.config.max_segment_length:
                # Split at natural word boundaries
                sub_segments = self._split_at_word_boundaries(segment)
                split.extend(sub_segments)
            else:
                split.append(segment)
        
        return split
    
    def _create_merged_segment(self, segments: List[TextSegment]) -> TextSegment:
        """Create a single segment from multiple segments."""
        if not segments:
            raise ValueError("Cannot create merged segment from empty list")
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start)
        
        # Combine text
        combined_text = " ".join(seg.text for seg in sorted_segments)
        
        # Create merged segment
        merged = TextSegment(
            text=combined_text,
            start=sorted_segments[0].start,
            end=sorted_segments[-1].end,
            speaker=sorted_segments[0].speaker,
            segment_id=f"merged_{sorted_segments[0].segment_id}",
            parent_id=sorted_segments[0].segment_id,
            confidence=min((seg.confidence for seg in sorted_segments if seg.confidence is not None), default=1.0),
            features={
                'merged_from': [seg.segment_id for seg in sorted_segments],
                'original_count': len(sorted_segments)
            }
        )
        
        return merged
    
    def _split_at_word_boundaries(self, segment: TextSegment) -> List[TextSegment]:
        """Split a segment at word boundaries."""
        words = segment.text.split()
        
        if len(words) <= 1:
            return [segment]
        
        # Calculate approximate duration per word
        total_duration = segment.end - segment.start
        duration_per_word = total_duration / len(words)
        
        split_segments = []
        current_time = segment.start
        
        for i, word in enumerate(words):
            word_start = current_time
            word_end = current_time + duration_per_word
            
            word_segment = TextSegment(
                text=word,
                start=word_start,
                end=word_end,
                speaker=segment.speaker,
                segment_id=f"{segment.segment_id}_word_{i}",
                parent_id=segment.segment_id,
                confidence=segment.confidence,
                features={
                    'split_from': segment.segment_id,
                    'word_index': i
                }
            )
            
            split_segments.append(word_segment)
            current_time = word_end
        
        return split_segments
    
    def extract_text_features(self, transcript: EnhancedTranscript) -> Dict[str, any]:
        """
        Extract linguistic features from a transcript.
        
        Args:
            transcript: The transcript to analyze
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'word_count': len(transcript.segments),
            'unique_words': len(set(seg.text.lower() for seg in transcript.segments)),
            'avg_word_length': self._calculate_avg_word_length(transcript),
            'avg_segment_duration': self._calculate_avg_segment_duration(transcript),
            'total_duration': transcript.metadata.duration,
            'speech_rate': transcript.metadata.word_count / max(transcript.metadata.duration, 0.1),
            'pause_frequency': self._calculate_pause_frequency(transcript),
            'vocabulary_richness': self._calculate_vocabulary_richness(transcript)
        }
        
        return features
    
    def _calculate_avg_word_length(self, transcript: EnhancedTranscript) -> float:
        """Calculate average word length in characters."""
        if not transcript.segments:
            return 0.0
        
        total_chars = sum(len(seg.text) for seg in transcript.segments)
        return total_chars / len(transcript.segments)
    
    def _calculate_avg_segment_duration(self, transcript: EnhancedTranscript) -> float:
        """Calculate average segment duration in seconds."""
        if not transcript.segments:
            return 0.0
        
        total_duration = sum(seg.end - seg.start for seg in transcript.segments)
        return total_duration / len(transcript.segments)
    
    def _calculate_pause_frequency(self, transcript: EnhancedTranscript) -> float:
        """Calculate frequency of pauses between segments."""
        if len(transcript.segments) < 2:
            return 0.0
        
        # Sort segments by start time
        sorted_segments = sorted(transcript.segments, key=lambda s: s.start)
        
        # Calculate gaps between segments
        gaps = []
        for i in range(1, len(sorted_segments)):
            gap = sorted_segments[i].start - sorted_segments[i-1].end
            if gap > 0.1:  # Only count significant gaps
                gaps.append(gap)
        
        if not gaps:
            return 0.0
        
        # Return average pause duration
        return sum(gaps) / len(gaps)
    
    def _calculate_vocabulary_richness(self, transcript: EnhancedTranscript) -> float:
        """Calculate vocabulary richness using type-token ratio."""
        if not transcript.segments:
            return 0.0
        
        word_count = len(transcript.segments)
        unique_words = len(set(seg.text.lower() for seg in transcript.segments))
        
        return unique_words / word_count