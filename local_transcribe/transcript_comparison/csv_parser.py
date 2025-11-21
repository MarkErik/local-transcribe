"""
CSV parser module for transcript comparison.

This module handles parsing of CSV files with Line and Word columns,
and implements line number alignment strategies that don't depend on exact line matches.
"""

import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .data_structures import (
    EnhancedTranscript, TextSegment, TranscriptMetadata
)


class CSVTranscriptParser:
    """Parses CSV transcript files with Line and Word columns."""
    
    def __init__(self, config=None):
        """Initialize the parser with configuration."""
        self.config = config
    
    def parse_csv_file(self, file_path: str, source_name: str = None) -> EnhancedTranscript:
        """
        Parse a CSV transcript file.
        
        Args:
            file_path: Path to the CSV file
            source_name: Name of the transcript source (for metadata)
            
        Returns:
            EnhancedTranscript object with parsed segments
        """
        segments = []
        line_numbers = []
        words = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        line_num = int(row['Line'])
                        word = row['Word']
                        
                        line_numbers.append(line_num)
                        words.append(word)
                        
                        # Create a text segment for each word
                        # Estimate timing based on line number if configured
                        start_time, end_time = self._estimate_timing(line_num, len(segments))
                        
                        segment = TextSegment(
                            text=word,
                            start=start_time,
                            end=end_time,
                            segment_id=f"{source_name or 'csv'}_line_{line_num}",
                            features={
                                "csv_line_number": line_num,
                                "source": source_name or "csv",
                                "original_index": len(segments)
                            }
                        )
                        
                        segments.append(segment)
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping malformed row {row}: {e}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file '{file_path}': {e}")
        
        # Create metadata
        duration = max((seg.end for seg in segments), default=0.0)
        metadata = TranscriptMetadata(
            source=source_name or "csv",
            creation_date=datetime.now(),
            duration=duration,
            word_count=len(segments),
            original_format="csv"
        )
        
        return EnhancedTranscript(
            segments=segments,
            metadata=metadata,
            features={
                "file_path": file_path,
                "line_numbers": line_numbers,
                "raw_words": words
            }
        )
    
    def _estimate_timing(self, line_number: int, segment_index: int) -> Tuple[float, float]:
        """
        Estimate timing information based on line number and segment index.
        
        Args:
            line_number: Original line number from CSV
            segment_index: Index of this segment in the transcript
            
        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        if not getattr(self.config, 'estimate_timing_from_line_numbers', True):
            # Simple sequential timing
            avg_duration = getattr(self.config, 'average_words_per_second', 2.5)
            start_time = segment_index / avg_duration
            end_time = start_time + (1.0 / avg_duration)
            return start_time, end_time
        
        # Estimate timing based on line number
        # Assume line numbers are roughly sequential and correspond to time
        avg_words_per_second = getattr(self.config, 'average_words_per_second', 2.5)
        
        # Use line number as a proxy for time position
        # This assumes line numbers are roughly sequential and start from 1
        if line_number <= 0:
            line_number = 1
            
        # Estimate start time based on line number
        # Divide by words per second to get time
        start_time = (line_number - 1) / avg_words_per_second
        
        # Each word takes approximately 1/avg_words_per_second seconds
        end_time = start_time + (1.0 / avg_words_per_second)
        
        return start_time, end_time


class LineNumberAlignmentStrategy:
    """
    Implements line number alignment strategy that doesn't depend on exact line matches.
    
    This strategy groups words into phrases based on line number proximity and
    aligns these phrases between transcripts, allowing for differences in
    segmentation while preserving context.
    """
    
    def __init__(self, config=None):
        """Initialize the alignment strategy with configuration."""
        self.config = config
        if config is None:
            # Use default values if no config provided
            self.phrase_length = 5
            self.line_tolerance = 2
        else:
            # Use values from config
            self.phrase_length = getattr(config, 'phrase_length', 5)
            self.line_tolerance = getattr(config, 'line_tolerance', 2)
    
    def group_into_phrases(self, transcript: EnhancedTranscript) -> EnhancedTranscript:
        """
        Group word segments into phrases based on line numbers.
        
        Args:
            transcript: Transcript with word-level segments
            
        Returns:
            New transcript with phrase-level segments
        """
        if not transcript.segments:
            return transcript
        
        # Sort segments by line number
        sorted_segments = sorted(
            transcript.segments, 
            key=lambda s: s.features.get('csv_line_number', 0)
        )
        
        phrases = []
        current_phrase_segments = []
        last_line_number = None
        
        for segment in sorted_segments:
            line_number = segment.features.get('csv_line_number', 0)
            
            # Start a new phrase if:
            # 1. This is the first segment
            # 2. We've reached the phrase length limit
            # 3. There's a gap in line numbers larger than tolerance
            if (last_line_number is None or 
                len(current_phrase_segments) >= self.phrase_length or
                (last_line_number is not None and 
                 line_number - last_line_number > self.line_tolerance)):
                
                # Finalize current phrase if it exists
                if current_phrase_segments:
                    phrase = self._create_phrase_segment(current_phrase_segments)
                    phrases.append(phrase)
                
                # Start new phrase
                current_phrase_segments = [segment]
            else:
                # Add to current phrase
                current_phrase_segments.append(segment)
            
            last_line_number = line_number
        
        # Add the last phrase
        if current_phrase_segments:
            phrase = self._create_phrase_segment(current_phrase_segments)
            phrases.append(phrase)
        
        # Create new transcript with phrases
        phrase_transcript = EnhancedTranscript(
            segments=phrases,
            metadata=transcript.metadata,
            features=transcript.features.copy() if transcript.features else {}
        )
        
        # Add phrase grouping flag
        phrase_transcript.features['phrase_grouped'] = True
        
        return phrase_transcript
    
    def _create_phrase_segment(self, segments: List[TextSegment]) -> TextSegment:
        """Create a phrase segment from multiple word segments."""
        if not segments:
            raise ValueError("Cannot create phrase from empty segment list")
        
        # Sort by line number
        sorted_segments = sorted(
            segments, 
            key=lambda s: s.features.get('csv_line_number', 0)
        )
        
        # Combine text
        phrase_text = " ".join(seg.text for seg in sorted_segments)
        
        # Calculate timing
        start_time = min(seg.start for seg in sorted_segments)
        end_time = max(seg.end for seg in sorted_segments)
        
        # Collect line numbers
        line_numbers = [seg.features.get('csv_line_number', 0) for seg in sorted_segments]
        
        # Create phrase segment
        phrase = TextSegment(
            text=phrase_text,
            start=start_time,
            end=end_time,
            segment_id=f"phrase_{min(line_numbers)}_to_{max(line_numbers)}",
            features={
                "constituent_words": [seg.text for seg in sorted_segments],
                "line_numbers": line_numbers,
                "word_count": len(sorted_segments),
                "min_line": min(line_numbers),
                "max_line": max(line_numbers)
            }
        )
        
        return phrase
    
    def align_by_line_proximity(self, reference: EnhancedTranscript, 
                               hypothesis: EnhancedTranscript) -> Dict[int, int]:
        """
        Create alignment between transcripts based on line number proximity.
        
        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript
            
        Returns:
            Dictionary mapping reference line numbers to hypothesis line numbers
        """
        # Group both transcripts into phrases
        ref_phrases = self.group_into_phrases(reference)
        hyp_phrases = self.group_into_phrases(hypothesis)
        
        # Create mapping based on overlapping line number ranges
        alignment = {}
        
        for ref_phrase in ref_phrases.segments:
            ref_min_line = ref_phrase.features.get('min_line', 0)
            ref_max_line = ref_phrase.features.get('max_line', 0)
            
            # Find hypothesis phrase with maximum overlap
            best_overlap = 0
            best_hyp_phrase = None
            
            for hyp_phrase in hyp_phrases.segments:
                hyp_min_line = hyp_phrase.features.get('min_line', 0)
                hyp_max_line = hyp_phrase.features.get('max_line', 0)
                
                # Calculate overlap
                overlap_start = max(ref_min_line, hyp_min_line)
                overlap_end = min(ref_max_line, hyp_max_line)
                
                if overlap_start <= overlap_end:
                    overlap = overlap_end - overlap_start + 1
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_hyp_phrase = hyp_phrase
            
            # If we found a good match, create alignment
            if best_hyp_phrase and best_overlap > 0:
                # Map each line in the reference phrase to the corresponding
                # line in the hypothesis phrase based on relative position
                ref_lines = sorted(ref_phrase.features.get('line_numbers', []))
                hyp_lines = sorted(best_hyp_phrase.features.get('line_numbers', []))
                
                # Map lines based on relative position
                for i, ref_line in enumerate(ref_lines):
                    if i < len(hyp_lines):
                        alignment[ref_line] = hyp_lines[i]
        
        return alignment