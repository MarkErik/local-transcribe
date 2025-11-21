"""
Metrics calculation module.

This module computes various quality metrics based on transcript comparison results,
including word error rate, semantic similarity, timing accuracy, and fluency scores.
"""

from typing import Dict, List, Optional

from .data_structures import (
    AlignmentResult, AlignedPair, QualityMetrics, ComparisonConfig
)


class MetricsCalculator:
    """Computes various quality metrics based on comparison results."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize the metrics calculator with configuration."""
        self.config = config
    
    def calculate_metrics(self, alignment_result: AlignmentResult) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics from alignment result.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            QualityMetrics object with calculated metrics
        """
        # Calculate individual metrics
        word_error_rate = self._calculate_word_error_rate(alignment_result)
        match_error_rate = self._calculate_match_error_rate(alignment_result)
        semantic_similarity = self._calculate_semantic_similarity(alignment_result)
        timing_accuracy = self._calculate_timing_accuracy(alignment_result)
        fluency_score = self._calculate_fluency_score(alignment_result)
        confidence_score = self._calculate_confidence_score(alignment_result)
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(
            word_error_rate, match_error_rate, semantic_similarity, 
            timing_accuracy, fluency_score, confidence_score
        )
        
        return QualityMetrics(
            word_error_rate=word_error_rate,
            match_error_rate=match_error_rate,
            semantic_similarity=semantic_similarity,
            timing_accuracy=timing_accuracy,
            fluency_score=fluency_score,
            confidence_score=confidence_score,
            overall_quality=overall_quality
        )
    
    def _calculate_word_error_rate(self, alignment_result: AlignmentResult) -> float:
        """
        Calculate Word Error Rate (WER).
        
        WER = (S + D + I) / N
        where S = substitutions, D = deletions, I = insertions, N = reference words
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Word Error Rate as a float between 0.0 and 1.0
        """
        substitutions = sum(1 for pair in alignment_result.aligned_pairs 
                          if pair.alignment_type == "substitution")
        deletions = sum(1 for pair in alignment_result.aligned_pairs 
                       if pair.alignment_type == "deletion")
        insertions = sum(1 for pair in alignment_result.aligned_pairs 
                        if pair.alignment_type == "insertion")
        
        # Count reference words
        reference_words = sum(1 for pair in alignment_result.aligned_pairs 
                            if pair.reference_segment)
        reference_words += len(alignment_result.unaligned_reference)
        
        if reference_words == 0:
            return 0.0
        
        return (substitutions + deletions + insertions) / reference_words
    
    def _calculate_match_error_rate(self, alignment_result: AlignmentResult) -> float:
        """
        Calculate Match Error Rate (MER).
        
        MER is similar to WER but considers the minimum number of edits required
        and is more robust to segmentation differences.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Match Error Rate as a float between 0.0 and 1.0
        """
        # Count errors
        errors = sum(1 for pair in alignment_result.aligned_pairs 
                    if pair.alignment_type in ["substitution", "deletion", "insertion"])
        errors += len(alignment_result.unaligned_reference)
        errors += len(alignment_result.unaligned_hypothesis)
        
        # Count total words
        total_words = sum(1 for pair in alignment_result.aligned_pairs 
                         if pair.reference_segment or pair.hypothesis_segment)
        total_words += len(alignment_result.unaligned_reference)
        total_words += len(alignment_result.unaligned_hypothesis)
        
        if total_words == 0:
            return 0.0
        
        return errors / total_words
    
    def _calculate_semantic_similarity(self, alignment_result: AlignmentResult) -> float:
        """
        Calculate average semantic similarity of aligned segments.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Semantic similarity score as a float between 0.0 and 1.0
        """
        total_similarity = 0.0
        total_weight = 0.0
        
        for pair in alignment_result.aligned_pairs:
            if pair.alignment_type in ["exact", "semantic", "substitution"] and pair.reference_segment:
                # Weight by segment length (number of words)
                weight = len(pair.reference_segment.text.split())
                total_similarity += pair.similarity_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_similarity / total_weight
    
    def _calculate_timing_accuracy(self, alignment_result: AlignmentResult) -> float:
        """
        Calculate timing accuracy between aligned segments.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Timing accuracy score as a float between 0.0 and 1.0
        """
        timing_differences = []
        
        for pair in alignment_result.aligned_pairs:
            if (pair.alignment_type in ["exact", "semantic"] and 
                pair.reference_segment and pair.hypothesis_segment):
                
                # Calculate midpoint timing difference
                ref_mid = (pair.reference_segment.start + pair.reference_segment.end) / 2
                hyp_mid = (pair.hypothesis_segment.start + pair.hypothesis_segment.end) / 2
                timing_diff = abs(ref_mid - hyp_mid)
                
                # Only consider differences within tolerance
                if timing_diff <= self.config.max_timing_difference:
                    timing_differences.append(timing_diff)
        
        if not timing_differences:
            return 0.0
        
        # Calculate average timing difference
        avg_difference = sum(timing_differences) / len(timing_differences)
        
        # Convert to accuracy score (lower difference = higher accuracy)
        # Normalize by timing tolerance
        accuracy = max(0.0, 1.0 - avg_difference / self.config.timing_tolerance)
        
        return accuracy
    
    def _calculate_fluency_score(self, alignment_result: AlignmentResult) -> float:
        """
        Calculate fluency score based on various factors.
        
        This is a simplified implementation. In a more sophisticated version,
        we would analyze pause patterns, punctuation, grammatical structure, etc.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Fluency score as a float between 0.0 and 1.0
        """
        # For now, use a simple heuristic based on alignment types
        # Higher proportion of exact matches indicates better fluency
        
        total_pairs = len(alignment_result.aligned_pairs)
        if total_pairs == 0:
            return 0.0
        
        exact_matches = sum(1 for pair in alignment_result.aligned_pairs 
                          if pair.alignment_type == "exact")
        
        # Base score on proportion of exact matches
        base_score = exact_matches / total_pairs
        
        # Adjust for unaligned segments (penalize heavily)
        unaligned_penalty = (len(alignment_result.unaligned_reference) + 
                           len(alignment_result.unaligned_hypothesis)) / max(total_pairs, 1)
        unaligned_penalty = min(0.5, unaligned_penalty * 0.1)  # Cap penalty at 0.5
        
        fluency_score = max(0.0, base_score - unaligned_penalty)
        
        return fluency_score
    
    def _calculate_confidence_score(self, alignment_result: AlignmentResult) -> float:
        """
        Calculate average confidence score of aligned segments.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Confidence score as a float between 0.0 and 1.0
        """
        total_confidence = 0.0
        count = 0
        
        for pair in alignment_result.aligned_pairs:
            if pair.reference_segment and pair.reference_segment.confidence is not None:
                total_confidence += pair.reference_segment.confidence
                count += 1
            
            if pair.hypothesis_segment and pair.hypothesis_segment.confidence is not None:
                total_confidence += pair.hypothesis_segment.confidence
                count += 1
        
        if count == 0:
            return 0.5  # Default confidence if no confidence values available
        
        return total_confidence / count
    
    def _calculate_overall_quality(self, word_error_rate: float, match_error_rate: float,
                                  semantic_similarity: float, timing_accuracy: float,
                                  fluency_score: float, confidence_score: float) -> float:
        """
        Calculate overall quality score as weighted combination of metrics.
        
        Args:
            word_error_rate: Word Error Rate (lower is better)
            match_error_rate: Match Error Rate (lower is better)
            semantic_similarity: Semantic similarity score (higher is better)
            timing_accuracy: Timing accuracy score (higher is better)
            fluency_score: Fluency score (higher is better)
            confidence_score: Confidence score (higher is better)
            
        Returns:
            Overall quality score as a float between 0.0 and 1.0
        """
        # Normalize error rates (convert to "goodness" scores where higher is better)
        normalized_wer = 1.0 - min(1.0, word_error_rate)
        normalized_mer = 1.0 - min(1.0, match_error_rate)
        
        # Calculate weighted sum
        overall = (
            self.config.wer_weight * normalized_wer +
            (1.0 - self.config.wer_weight) * normalized_mer +  # Use MER for the rest of the weight
            self.config.semantic_weight * semantic_similarity +
            self.config.timing_weight * timing_accuracy +
            self.config.fluency_weight * fluency_score +
            0.05 * confidence_score  # Small weight for confidence
        )
        
        # Normalize weights to sum to 1.0
        total_weight = (
            self.config.wer_weight + (1.0 - self.config.wer_weight) +
            self.config.semantic_weight + self.config.timing_weight + 
            self.config.fluency_weight + 0.05
        )
        
        return overall / total_weight


class DetailedMetricsCalculator:
    """Calculates more detailed metrics for in-depth analysis."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize the detailed metrics calculator with configuration."""
        self.config = config
        self.base_calculator = MetricsCalculator(config)
    
    def calculate_detailed_metrics(self, alignment_result: AlignmentResult) -> Dict[str, float]:
        """
        Calculate detailed metrics for in-depth analysis.
        
        Args:
            alignment_result: Result of transcript alignment
            
        Returns:
            Dictionary of detailed metric names and values
        """
        # Get base metrics
        quality_metrics = self.base_calculator.calculate_metrics(alignment_result)
        
        # Calculate additional detailed metrics
        detailed_metrics = {
            # Error counts
            "substitution_count": self._count_substitutions(alignment_result),
            "insertion_count": self._count_insertions(alignment_result),
            "deletion_count": self._count_deletions(alignment_result),
            "exact_match_count": self._count_exact_matches(alignment_result),
            "semantic_match_count": self._count_semantic_matches(alignment_result),
            
            # Error rates by type
            "substitution_rate": self._calculate_substitution_rate(alignment_result),
            "insertion_rate": self._calculate_insertion_rate(alignment_result),
            "deletion_rate": self._calculate_deletion_rate(alignment_result),
            
            # Alignment statistics
            "alignment_coverage": self._calculate_alignment_coverage(alignment_result),
            "average_similarity": self._calculate_average_similarity(alignment_result),
            "similarity_stddev": self._calculate_similarity_stddev(alignment_result),
            
            # Timing statistics
            "average_timing_diff": self._calculate_average_timing_diff(alignment_result),
            "max_timing_diff": self._calculate_max_timing_diff(alignment_result),
            
            # Segment statistics
            "average_segment_length": self._calculate_average_segment_length(alignment_result),
            "segment_length_variance": self._calculate_segment_length_variance(alignment_result)
        }
        
        # Add base metrics
        detailed_metrics.update({
            "word_error_rate": quality_metrics.word_error_rate,
            "match_error_rate": quality_metrics.match_error_rate,
            "semantic_similarity": quality_metrics.semantic_similarity,
            "timing_accuracy": quality_metrics.timing_accuracy,
            "fluency_score": quality_metrics.fluency_score,
            "confidence_score": quality_metrics.confidence_score,
            "overall_quality": quality_metrics.overall_quality
        })
        
        return detailed_metrics
    
    def _count_substitutions(self, alignment_result: AlignmentResult) -> int:
        """Count number of substitutions."""
        return sum(1 for pair in alignment_result.aligned_pairs 
                  if pair.alignment_type == "substitution")
    
    def _count_insertions(self, alignment_result: AlignmentResult) -> int:
        """Count number of insertions."""
        return sum(1 for pair in alignment_result.aligned_pairs 
                  if pair.alignment_type == "insertion")
    
    def _count_deletions(self, alignment_result: AlignmentResult) -> int:
        """Count number of deletions."""
        return sum(1 for pair in alignment_result.aligned_pairs 
                  if pair.alignment_type == "deletion")
    
    def _count_exact_matches(self, alignment_result: AlignmentResult) -> int:
        """Count number of exact matches."""
        return sum(1 for pair in alignment_result.aligned_pairs 
                  if pair.alignment_type == "exact")
    
    def _count_semantic_matches(self, alignment_result: AlignmentResult) -> int:
        """Count number of semantic matches."""
        return sum(1 for pair in alignment_result.aligned_pairs 
                  if pair.alignment_type == "semantic")
    
    def _calculate_substitution_rate(self, alignment_result: AlignmentResult) -> float:
        """Calculate substitution rate."""
        total_reference = sum(1 for pair in alignment_result.aligned_pairs 
                            if pair.reference_segment)
        total_reference += len(alignment_result.unaligned_reference)
        
        if total_reference == 0:
            return 0.0
        
        substitutions = self._count_substitutions(alignment_result)
        return substitutions / total_reference
    
    def _calculate_insertion_rate(self, alignment_result: AlignmentResult) -> float:
        """Calculate insertion rate."""
        total_reference = sum(1 for pair in alignment_result.aligned_pairs 
                            if pair.reference_segment)
        total_reference += len(alignment_result.unaligned_reference)
        
        if total_reference == 0:
            return 0.0
        
        insertions = self._count_insertions(alignment_result)
        return insertions / total_reference
    
    def _calculate_deletion_rate(self, alignment_result: AlignmentResult) -> float:
        """Calculate deletion rate."""
        total_reference = sum(1 for pair in alignment_result.aligned_pairs 
                            if pair.reference_segment)
        total_reference += len(alignment_result.unaligned_reference)
        
        if total_reference == 0:
            return 0.0
        
        deletions = self._count_deletions(alignment_result)
        return deletions / total_reference
    
    def _calculate_alignment_coverage(self, alignment_result: AlignmentResult) -> float:
        """Calculate proportion of segments that were aligned."""
        total_segments = (len(alignment_result.aligned_pairs) + 
                         len(alignment_result.unaligned_reference) + 
                         len(alignment_result.unaligned_hypothesis))
        
        if total_segments == 0:
            return 0.0
        
        aligned_segments = len(alignment_result.aligned_pairs)
        return aligned_segments / total_segments
    
    def _calculate_average_similarity(self, alignment_result: AlignmentResult) -> float:
        """Calculate average similarity score."""
        similarities = [pair.similarity_score for pair in alignment_result.aligned_pairs 
                       if pair.alignment_type in ["exact", "semantic", "substitution"]]
        
        if not similarities:
            return 0.0
        
        return sum(similarities) / len(similarities)
    
    def _calculate_similarity_stddev(self, alignment_result: AlignmentResult) -> float:
        """Calculate standard deviation of similarity scores."""
        similarities = [pair.similarity_score for pair in alignment_result.aligned_pairs 
                       if pair.alignment_type in ["exact", "semantic", "substitution"]]
        
        if len(similarities) < 2:
            return 0.0
        
        avg = sum(similarities) / len(similarities)
        variance = sum((s - avg) ** 2 for s in similarities) / len(similarities)
        
        return variance ** 0.5
    
    def _calculate_average_timing_diff(self, alignment_result: AlignmentResult) -> float:
        """Calculate average timing difference."""
        timing_diffs = [pair.timing_difference for pair in alignment_result.aligned_pairs 
                       if pair.alignment_type in ["exact", "semantic"]]
        
        if not timing_diffs:
            return 0.0
        
        return sum(timing_diffs) / len(timing_diffs)
    
    def _calculate_max_timing_diff(self, alignment_result: AlignmentResult) -> float:
        """Calculate maximum timing difference."""
        timing_diffs = [pair.timing_difference for pair in alignment_result.aligned_pairs 
                       if pair.alignment_type in ["exact", "semantic"]]
        
        if not timing_diffs:
            return 0.0
        
        return max(timing_diffs)
    
    def _calculate_average_segment_length(self, alignment_result: AlignmentResult) -> float:
        """Calculate average segment length in characters."""
        all_segments = []
        
        for pair in alignment_result.aligned_pairs:
            if pair.reference_segment:
                all_segments.append(pair.reference_segment)
            if pair.hypothesis_segment:
                all_segments.append(pair.hypothesis_segment)
        
        all_segments.extend(alignment_result.unaligned_reference)
        all_segments.extend(alignment_result.unaligned_hypothesis)
        
        if not all_segments:
            return 0.0
        
        total_length = sum(len(seg.text) for seg in all_segments)
        return total_length / len(all_segments)
    
    def _calculate_segment_length_variance(self, alignment_result: AlignmentResult) -> float:
        """Calculate variance in segment lengths."""
        all_segments = []
        
        for pair in alignment_result.aligned_pairs:
            if pair.reference_segment:
                all_segments.append(pair.reference_segment)
            if pair.hypothesis_segment:
                all_segments.append(pair.hypothesis_segment)
        
        all_segments.extend(alignment_result.unaligned_reference)
        all_segments.extend(alignment_result.unaligned_hypothesis)
        
        if len(all_segments) < 2:
            return 0.0
        
        lengths = [len(seg.text) for seg in all_segments]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        return variance