"""
Transcript comparison engine.

This module provides the main TranscriptComparisonEngine class that orchestrates
all components of the transcript comparison system, including preprocessing,
similarity calculation, alignment, metrics computation, and output formatting.
"""

from typing import Dict, List, Optional, Union

from .data_structures import (
    EnhancedTranscript, ComparisonResult, ComparisonConfig, AlignmentResult,
    QualityMetrics, ComparisonSummary, Difference
)
from .preprocessing import TranscriptPreprocessor
from .similarity import SimilarityCalculator
from .alignment import TranscriptAlignmentEngine, HierarchicalAlignmentEngine
from .metrics import MetricsCalculator
from .output_formatting import ComparisonResultFormatter
from .csv_parser import CSVTranscriptParser, LineNumberAlignmentStrategy


class TranscriptComparisonEngine:
    """
    Main engine for transcript comparison that orchestrates all components.
    
    This class provides a simple interface for comparing two transcripts,
    handling the complete comparison workflow from parsing to output.
    """
    
    def __init__(self, config: ComparisonConfig = None):
        """
        Initialize the transcript comparison engine.
        
        Args:
            config: Configuration for comparison parameters
        """
        self.config = config or ComparisonConfig()
        
        # Initialize all components
        self.preprocessor = TranscriptPreprocessor(self.config)
        self.similarity_calculator = SimilarityCalculator(self.config)
        self.alignment_engine = TranscriptAlignmentEngine(self.config)
        self.hierarchical_alignment_engine = HierarchicalAlignmentEngine(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        self.result_formatter = ComparisonResultFormatter(self.config)
        
        # Initialize CSV-specific components
        self.csv_parser = CSVTranscriptParser(self.config)
        self.line_alignment_strategy = LineNumberAlignmentStrategy(self.config)
    
    def compare_csv_files(self, file_path1: str, file_path2: str, 
                          source1: str = None, source2: str = None) -> ComparisonResult:
        """
        Compare two CSV transcript files.
        
        Args:
            file_path1: Path to first CSV file
            file_path2: Path to second CSV file
            source1: Name of first transcript source
            source2: Name of second transcript source
            
        Returns:
            ComparisonResult with detailed comparison results
        """
        # Parse CSV files
        transcript1 = self.csv_parser.parse_csv_file(file_path1, source1 or "file1")
        transcript2 = self.csv_parser.parse_csv_file(file_path2, source2 or "file2")
        
        # Compare the parsed transcripts
        return self.compare_transcripts(transcript1, transcript2)
    
    def compare_transcripts(self, reference: EnhancedTranscript, 
                           hypothesis: EnhancedTranscript) -> ComparisonResult:
        """
        Compare two transcripts.
        
        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript to compare against reference
            
        Returns:
            ComparisonResult with detailed comparison results
        """
        # Step 1: Preprocess transcripts
        normalized_reference = self.preprocessor.normalize_transcript(reference)
        normalized_hypothesis = self.preprocessor.normalize_transcript(hypothesis)
        
        # Step 2: Adjust segmentation if needed
        adjusted_reference = self.preprocessor.adjust_segmentation(normalized_reference)
        adjusted_hypothesis = self.preprocessor.adjust_segmentation(normalized_hypothesis)
        
        # Step 3: Perform alignment
        if getattr(self.config, 'use_hierarchical_alignment', False):
            alignment_result = self.hierarchical_alignment_engine.align_hierarchically(
                adjusted_reference, adjusted_hypothesis
            )
        else:
            alignment_result = self.alignment_engine.align_transcripts(
                adjusted_reference, adjusted_hypothesis
            )
        
        # Step 4: Calculate metrics
        quality_metrics = self.metrics_calculator.calculate_metrics(alignment_result)
        
        # Step 5: Identify differences
        differences = self._identify_differences(alignment_result)
        
        # Step 6: Create summary
        summary = self._create_summary(alignment_result, quality_metrics)
        
        # Step 7: Create final comparison result
        comparison_result = ComparisonResult(
            alignment_result=alignment_result,
            quality_metrics=quality_metrics,
            differences=differences,
            summary=summary
        )
        
        return comparison_result
    
    def _identify_differences(self, alignment_result: AlignmentResult) -> List[Difference]:
        """
        Identify and categorize differences between transcripts.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            List of Difference objects
        """
        differences = []
        
        # Process aligned pairs to find substitutions and timing differences
        for pair in alignment_result.aligned_pairs:
            if pair.alignment_type == "substitution":
                # Create substitution difference
                diff = Difference(
                    type="substitution",
                    severity=self._calculate_severity(pair.similarity_score),
                    reference_segment=pair.reference_segment,
                    hypothesis_segment=pair.hypothesis_segment,
                    description=f"Word substituted: '{pair.reference_segment.text}' → '{pair.hypothesis_segment.text}'",
                    context=self._get_context(pair, alignment_result)
                )
                differences.append(diff)
            
            elif pair.alignment_type in ["exact", "semantic"] and pair.timing_difference > self.config.timing_tolerance:
                # Create timing difference
                diff = Difference(
                    type="timing",
                    severity=self._calculate_timing_severity(pair.timing_difference),
                    reference_segment=pair.reference_segment,
                    hypothesis_segment=pair.hypothesis_segment,
                    description=f"Timing difference: {pair.timing_difference:.2f}s",
                    context=self._get_context(pair, alignment_result)
                )
                differences.append(diff)
        
        # Process unaligned segments (insertions and deletions)
        for segment in alignment_result.unaligned_reference:
            # Create deletion difference
            diff = Difference(
                type="deletion",
                severity="moderate",  # Default severity for deletions
                reference_segment=segment,
                hypothesis_segment=None,
                description=f"Word missing from hypothesis: '{segment.text}'",
                context=self._get_segment_context(segment, alignment_result, is_reference=True)
            )
            differences.append(diff)
        
        for segment in alignment_result.unaligned_hypothesis:
            # Create insertion difference
            diff = Difference(
                type="insertion",
                severity="minor",  # Default severity for insertions
                reference_segment=None,
                hypothesis_segment=segment,
                description=f"Extra word in hypothesis: '{segment.text}'",
                context=self._get_segment_context(segment, alignment_result, is_reference=False)
            )
            differences.append(diff)
        
        # Sort differences by severity and position
        differences.sort(key=lambda d: (
            self._severity_order(d.severity),
            d.reference_segment.start if d.reference_segment else d.hypothesis_segment.start
        ))
        
        # Limit number of differences
        return differences[:self.config.max_differences]
    
    def _calculate_severity(self, similarity_score: float) -> str:
        """Calculate severity based on similarity score."""
        if similarity_score >= 0.8:
            return "minor"
        elif similarity_score >= 0.5:
            return "moderate"
        else:
            return "major"
    
    def _calculate_timing_severity(self, timing_difference: float) -> str:
        """Calculate severity based on timing difference."""
        if timing_difference <= self.config.timing_tolerance:
            return "minor"
        elif timing_difference <= self.config.max_timing_difference / 2:
            return "moderate"
        else:
            return "major"
    
    def _severity_order(self, severity: str) -> int:
        """Get numeric order for severity sorting."""
        return {"major": 0, "moderate": 1, "minor": 2}.get(severity, 3)
    
    def _get_context(self, aligned_pair, alignment_result: AlignmentResult) -> List:
        """Get context segments for an aligned pair."""
        if not aligned_pair.reference_segment:
            return []
        
        # Find the index of this segment in the aligned pairs
        target_index = -1
        for i, pair in enumerate(alignment_result.aligned_pairs):
            if pair == aligned_pair:
                target_index = i
                break
        
        if target_index == -1:
            return []
        
        # Get surrounding segments
        context_segments = []
        start_idx = max(0, target_index - self.config.context_window_size)
        end_idx = min(len(alignment_result.aligned_pairs) - 1, 
                      target_index + self.config.context_window_size)
        
        for i in range(start_idx, end_idx + 1):
            if i != target_index:  # Skip the current segment
                pair = alignment_result.aligned_pairs[i]
                if pair.reference_segment:
                    context_segments.append(pair.reference_segment)
        
        return context_segments
    
    def _get_segment_context(self, segment, alignment_result: AlignmentResult, 
                            is_reference: bool = True) -> List:
        """Get context segments for an unaligned segment."""
        context_segments = []
        
        # Add nearby aligned segments as context
        for pair in alignment_result.aligned_pairs[:self.config.context_window_size * 2]:
            if is_reference and pair.reference_segment:
                context_segments.append(pair.reference_segment)
            elif not is_reference and pair.hypothesis_segment:
                context_segments.append(pair.hypothesis_segment)
        
        return context_segments
    
    def _create_summary(self, alignment_result: AlignmentResult, 
                        quality_metrics: QualityMetrics) -> ComparisonSummary:
        """Create summary statistics for the comparison."""
        # Count different types of alignments
        total_segments = len(alignment_result.aligned_pairs)
        exact_matches = sum(1 for pair in alignment_result.aligned_pairs 
                           if pair.alignment_type == "exact")
        semantic_matches = sum(1 for pair in alignment_result.aligned_pairs 
                             if pair.alignment_type == "semantic")
        substitutions = sum(1 for pair in alignment_result.aligned_pairs 
                          if pair.alignment_type == "substitution")
        insertions = sum(1 for pair in alignment_result.aligned_pairs 
                        if pair.alignment_type == "insertion")
        deletions = sum(1 for pair in alignment_result.aligned_pairs 
                      if pair.alignment_type == "deletion")
        
        # Count timing differences
        timing_differences = sum(1 for pair in alignment_result.aligned_pairs 
                               if (pair.alignment_type in ["exact", "semantic"] and 
                                   pair.timing_difference > self.config.timing_tolerance))
        
        # Calculate overall similarity
        matched_segments = exact_matches + semantic_matches
        overall_similarity = (matched_segments / max(total_segments, 1)) * quality_metrics.semantic_similarity
        
        return ComparisonSummary(
            total_segments=total_segments,
            matched_segments=matched_segments,
            substituted_segments=substitutions,
            inserted_segments=insertions,
            deleted_segments=deletions,
            timing_differences=timing_differences,
            overall_similarity=overall_similarity
        )
    
    def format_results(self, comparison_result: ComparisonResult, 
                      output_format: str = "text") -> str:
        """
        Format comparison results for output.
        
        Args:
            comparison_result: Result of transcript comparison
            output_format: Output format ("json", "text", "html", "csv")
            
        Returns:
            Formatted output as a string
        """
        return self.result_formatter.format_results(comparison_result, output_format)
    
    def save_results(self, comparison_result: ComparisonResult, 
                    output_path: str, output_format: str = "text") -> None:
        """
        Save comparison results to a file.
        
        Args:
            comparison_result: Result of transcript comparison
            output_path: Path to output file
            output_format: Output format ("json", "text", "html", "csv")
        """
        self.result_formatter.write_to_file(comparison_result, output_path, output_format)
    
    def compare_with_line_alignment(self, file_path1: str, file_path2: str,
                                   source1: str = None, source2: str = None) -> ComparisonResult:
        """
        Compare two CSV files using line number alignment strategy.
        
        This method uses a specialized alignment strategy that groups words into
        phrases based on line number proximity, allowing for differences in
        segmentation while preserving context.
        
        Args:
            file_path1: Path to first CSV file
            file_path2: Path to second CSV file
            source1: Name of first transcript source
            source2: Name of second transcript source
            
        Returns:
            ComparisonResult with detailed comparison results
        """
        # Parse CSV files
        transcript1 = self.csv_parser.parse_csv_file(file_path1, source1 or "file1")
        transcript2 = self.csv_parser.parse_csv_file(file_path2, source2 or "file2")
        
        # Group into phrases based on line numbers
        phrase_transcript1 = self.line_alignment_strategy.group_into_phrases(transcript1)
        phrase_transcript2 = self.line_alignment_strategy.group_into_phrases(transcript2)
        
        # Compare the phrase-level transcripts
        return self.compare_transcripts(phrase_transcript1, phrase_transcript2)