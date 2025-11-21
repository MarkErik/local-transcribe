"""
Transcript alignment module.

This module handles alignment of transcripts with different segmentation structures,
using dynamic programming and semantic similarity to find optimal alignments.
"""

from typing import Dict, List, Optional, Tuple

from .data_structures import (
    EnhancedTranscript, TextSegment, AlignmentResult, AlignedPair, ComparisonConfig
)
from .similarity import SimilarityCalculator, BatchSimilarityCalculator, SimilarityScore


class TranscriptAlignmentEngine:
    """Handles alignment of transcripts with different segmentation structures."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize the alignment engine with configuration."""
        self.config = config
        self.similarity_calculator = SimilarityCalculator(config)
        self.batch_calculator = BatchSimilarityCalculator(config)
    
    def align_transcripts(self, reference: EnhancedTranscript, 
                         hypothesis: EnhancedTranscript) -> AlignmentResult:
        """
        Align two transcripts using optimal alignment algorithm.
        
        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript to align with reference
            
        Returns:
            AlignmentResult containing aligned pairs and unaligned segments
        """
        # Step 1: Preprocess transcripts if needed
        ref_segments = reference.segments
        hyp_segments = hypothesis.segments
        
        # Step 2: Perform sequence alignment using modified Needleman-Wunsch
        aligned_pairs = self._sequence_align(ref_segments, hyp_segments)
        
        # Step 3: Identify unaligned segments
        aligned_ref_indices = {pair.reference_segment.features.get("original_index", i) 
                              for i, pair in enumerate(aligned_pairs) if pair.reference_segment}
        aligned_hyp_indices = {pair.hypothesis_segment.features.get("original_index", j) 
                              for j, pair in enumerate(aligned_pairs) if pair.hypothesis_segment}
        
        unaligned_reference = [seg for i, seg in enumerate(ref_segments) 
                              if i not in aligned_ref_indices]
        unaligned_hypothesis = [seg for j, seg in enumerate(hyp_segments) 
                              if j not in aligned_hyp_indices]
        
        # Step 4: Calculate overall alignment score
        alignment_score = self._calculate_alignment_score(aligned_pairs)
        
        return AlignmentResult(
            aligned_pairs=aligned_pairs,
            unaligned_reference=unaligned_reference,
            unaligned_hypothesis=unaligned_hypothesis,
            alignment_score=alignment_score
        )
    
    def _sequence_align(self, ref_segments: List[TextSegment], 
                       hyp_segments: List[TextSegment]) -> List[AlignedPair]:
        """
        Perform sequence alignment using modified Needleman-Wunsch algorithm.
        
        Args:
            ref_segments: Reference transcript segments
            hyp_segments: Hypothesis transcript segments
            
        Returns:
            List of aligned segment pairs
        """
        if not ref_segments or not hyp_segments:
            return self._handle_empty_transcripts(ref_segments, hyp_segments)
        
        # Calculate similarity matrix
        sim_matrix = self.batch_calculator.calculate_batch_similarity(ref_segments, hyp_segments)
        
        # Initialize DP matrix
        n = len(ref_segments)
        m = len(hyp_segments)
        
        # Create DP matrix with scores
        dp = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]
        
        # Create traceback matrix
        traceback = [[None for _ in range(m + 1)] for _ in range(n + 1)]
        
        # Initialize first row and column
        for i in range(1, n + 1):
            dp[i][0] = dp[i-1][0] - self.config.gap_penalty
            traceback[i][0] = 'up'
        
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j-1] - self.config.gap_penalty
            traceback[0][j] = 'left'
        
        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Get similarity score
                sim_score = sim_matrix[i-1][j-1].combined
                
                # Calculate scores for three possible moves
                match_score = dp[i-1][j-1] + sim_score
                delete_score = dp[i-1][j] - self.config.gap_penalty
                insert_score = dp[i][j-1] - self.config.gap_penalty
                
                # Choose the best score
                best_score = max(match_score, delete_score, insert_score)
                dp[i][j] = best_score
                
                # Record traceback
                if best_score == match_score:
                    traceback[i][j] = 'diagonal'
                elif best_score == delete_score:
                    traceback[i][j] = 'up'
                else:
                    traceback[i][j] = 'left'
        
        # Traceback to find optimal alignment
        aligned_pairs = self._traceback_alignment(
            ref_segments, hyp_segments, dp, traceback, sim_matrix
        )
        
        return aligned_pairs
    
    def _handle_empty_transcripts(self, ref_segments: List[TextSegment], 
                                 hyp_segments: List[TextSegment]) -> List[AlignedPair]:
        """Handle case where one or both transcripts are empty."""
        aligned_pairs = []
        
        # If reference is empty, all hypothesis segments are insertions
        if not ref_segments and hyp_segments:
            for j, seg in enumerate(hyp_segments):
                pair = AlignedPair(
                    reference_segment=None,
                    hypothesis_segment=seg,
                    similarity_score=0.0,
                    alignment_type="insertion",
                    timing_difference=0.0,
                    confidence=1.0
                )
                aligned_pairs.append(pair)
        
        # If hypothesis is empty, all reference segments are deletions
        elif ref_segments and not hyp_segments:
            for i, seg in enumerate(ref_segments):
                pair = AlignedPair(
                    reference_segment=seg,
                    hypothesis_segment=None,
                    similarity_score=0.0,
                    alignment_type="deletion",
                    timing_difference=0.0,
                    confidence=1.0
                )
                aligned_pairs.append(pair)
        
        return aligned_pairs
    
    def _traceback_alignment(self, ref_segments: List[TextSegment], 
                            hyp_segments: List[TextSegment],
                            dp: List[List[float]], 
                            traceback: List[List[str]],
                            sim_matrix: List[List[SimilarityScore]]) -> List[AlignedPair]:
        """
        Traceback through DP matrix to find optimal alignment.
        
        Args:
            ref_segments: Reference transcript segments
            hyp_segments: Hypothesis transcript segments
            dp: DP matrix with scores
            traceback: Traceback matrix
            sim_matrix: Similarity matrix
            
        Returns:
            List of aligned segment pairs
        """
        i = len(ref_segments)
        j = len(hyp_segments)
        aligned_pairs = []
        
        while i > 0 or j > 0:
            if i == 0:
                # All remaining are insertions
                for k in range(j-1, -1, -1):
                    pair = AlignedPair(
                        reference_segment=None,
                        hypothesis_segment=hyp_segments[k],
                        similarity_score=0.0,
                        alignment_type="insertion",
                        timing_difference=0.0,
                        confidence=1.0
                    )
                    aligned_pairs.append(pair)
                break
            
            elif j == 0:
                # All remaining are deletions
                for k in range(i-1, -1, -1):
                    pair = AlignedPair(
                        reference_segment=ref_segments[k],
                        hypothesis_segment=None,
                        similarity_score=0.0,
                        alignment_type="deletion",
                        timing_difference=0.0,
                        confidence=1.0
                    )
                    aligned_pairs.append(pair)
                break
            
            else:
                direction = traceback[i][j]
                
                if direction == 'diagonal':
                    # Match or substitution
                    ref_seg = ref_segments[i-1]
                    hyp_seg = hyp_segments[j-1]
                    sim_score = sim_matrix[i-1][j-1]
                    
                    # Determine alignment type
                    if sim_score.combined >= self.config.semantic_threshold:
                        alignment_type = "semantic"
                    elif ref_seg.text.lower() == hyp_seg.text.lower():
                        alignment_type = "exact"
                    else:
                        alignment_type = "substitution"
                    
                    pair = AlignedPair(
                        reference_segment=ref_seg,
                        hypothesis_segment=hyp_seg,
                        similarity_score=sim_score.combined,
                        alignment_type=alignment_type,
                        timing_difference=abs(ref_seg.start - hyp_seg.start),
                        confidence=sim_score.combined
                    )
                    aligned_pairs.append(pair)
                    
                    i -= 1
                    j -= 1
                
                elif direction == 'up':
                    # Deletion
                    pair = AlignedPair(
                        reference_segment=ref_segments[i-1],
                        hypothesis_segment=None,
                        similarity_score=0.0,
                        alignment_type="deletion",
                        timing_difference=0.0,
                        confidence=1.0
                    )
                    aligned_pairs.append(pair)
                    
                    i -= 1
                
                elif direction == 'left':
                    # Insertion
                    pair = AlignedPair(
                        reference_segment=None,
                        hypothesis_segment=hyp_segments[j-1],
                        similarity_score=0.0,
                        alignment_type="insertion",
                        timing_difference=0.0,
                        confidence=1.0
                    )
                    aligned_pairs.append(pair)
                    
                    j -= 1
        
        # Reverse to get correct order
        aligned_pairs.reverse()
        
        return aligned_pairs
    
    def _calculate_alignment_score(self, aligned_pairs: List[AlignedPair]) -> float:
        """Calculate overall alignment score."""
        if not aligned_pairs:
            return 0.0
        
        total_score = sum(pair.similarity_score for pair in aligned_pairs)
        return total_score / len(aligned_pairs)


class HierarchicalAlignmentEngine:
    """Handles hierarchical alignment at multiple granularity levels."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize the hierarchical alignment engine with configuration."""
        self.config = config
        self.base_engine = TranscriptAlignmentEngine(config)
    
    def align_hierarchically(self, reference: EnhancedTranscript, 
                            hypothesis: EnhancedTranscript) -> AlignmentResult:
        """
        Align transcripts at multiple granularity levels.
        
        Args:
            reference: Reference transcript
            hypothesis: Hypothesis transcript to align with reference
            
        Returns:
            AlignmentResult containing aligned pairs and unaligned segments
        """
        # Step 1: Group segments into phrases for coarse alignment
        ref_phrases = self._group_into_phrases(reference.segments)
        hyp_phrases = self._group_into_phrases(hypothesis.segments)
        
        # Step 2: Create phrase-level transcripts
        ref_phrase_transcript = EnhancedTranscript(
            segments=ref_phrases,
            metadata=reference.metadata
        )
        
        hyp_phrase_transcript = EnhancedTranscript(
            segments=hyp_phrases,
            metadata=hypothesis.metadata
        )
        
        # Step 3: Perform coarse alignment at phrase level
        phrase_alignment = self.base_engine.align_transcripts(
            ref_phrase_transcript, hyp_phrase_transcript
        )
        
        # Step 4: Perform fine alignment within aligned phrases
        word_alignment = self._fine_align_within_phrases(
            reference.segments, hypothesis.segments, phrase_alignment
        )
        
        return word_alignment
    
    def _group_into_phrases(self, segments: List[TextSegment]) -> List[TextSegment]:
        """Group word segments into phrases."""
        if not segments:
            return []
        
        phrases = []
        phrase_length = self.config.phrase_length
        
        for i in range(0, len(segments), phrase_length):
            phrase_segments = segments[i:i+phrase_length]
            
            if phrase_segments:
                # Create a phrase segment
                phrase_text = " ".join(seg.text for seg in phrase_segments)
                phrase_start = phrase_segments[0].start
                phrase_end = phrase_segments[-1].end
                
                # Store original indices for reference
                original_indices = [i + j for j in range(len(phrase_segments))]
                
                phrase = TextSegment(
                    text=phrase_text,
                    start=phrase_start,
                    end=phrase_end,
                    segment_id=f"phrase_{i//phrase_length}",
                    features={
                        "constituent_words": [seg.text for seg in phrase_segments],
                        "original_indices": original_indices
                    }
                )
                phrases.append(phrase)
        
        return phrases
    
    def _fine_align_within_phrases(self, ref_words: List[TextSegment], 
                                  hyp_words: List[TextSegment],
                                  phrase_alignment: AlignmentResult) -> AlignmentResult:
        """Perform fine alignment within aligned phrases."""
        all_aligned_pairs = []
        
        for pair in phrase_alignment.aligned_pairs:
            if pair.alignment_type in ["exact", "semantic"] and pair.reference_segment and pair.hypothesis_segment:
                # Get original word indices for this phrase
                ref_indices = pair.reference_segment.features.get("original_indices", [])
                hyp_indices = pair.hypothesis_segment.features.get("original_indices", [])
                
                # Extract word segments for this phrase
                ref_phrase_words = [ref_words[i] for i in ref_indices if i < len(ref_words)]
                hyp_phrase_words = [hyp_words[j] for j in hyp_indices if j < len(hyp_words)]
                
                # Align words within this phrase
                if ref_phrase_words and hyp_phrase_words:
                    phrase_ref_transcript = EnhancedTranscript(
                        segments=ref_phrase_words,
                        metadata=None  # Not needed for this sub-alignment
                    )
                    
                    phrase_hyp_transcript = EnhancedTranscript(
                        segments=hyp_phrase_words,
                        metadata=None  # Not needed for this sub-alignment
                    )
                    
                    phrase_word_alignment = self.base_engine.align_transcripts(
                        phrase_ref_transcript, phrase_hyp_transcript
                    )
                    
                    all_aligned_pairs.extend(phrase_word_alignment.aligned_pairs)
        
        # Identify unaligned segments
        aligned_ref_indices = set()
        aligned_hyp_indices = set()
        
        for pair in all_aligned_pairs:
            if pair.reference_segment:
                idx = pair.reference_segment.features.get("original_index")
                if idx is not None:
                    aligned_ref_indices.add(idx)
            
            if pair.hypothesis_segment:
                idx = pair.hypothesis_segment.features.get("original_index")
                if idx is not None:
                    aligned_hyp_indices.add(idx)
        
        unaligned_reference = [seg for i, seg in enumerate(ref_words) 
                              if i not in aligned_ref_indices]
        unaligned_hypothesis = [seg for j, seg in enumerate(hyp_words) 
                              if j not in aligned_hyp_indices]
        
        # Calculate overall alignment score
        alignment_score = self.base_engine._calculate_alignment_score(all_aligned_pairs)
        
        return AlignmentResult(
            aligned_pairs=all_aligned_pairs,
            unaligned_reference=unaligned_reference,
            unaligned_hypothesis=unaligned_hypothesis,
            alignment_score=alignment_score
        )