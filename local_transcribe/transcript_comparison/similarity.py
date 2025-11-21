"""
Similarity calculation module.

This module provides multiple similarity measures for comparing text segments,
including semantic similarity, fuzzy string matching, phonetic similarity,
and contextual similarity.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import metaphone
    METAPHONE_AVAILABLE = True
except ImportError:
    METAPHONE_AVAILABLE = False

from .data_structures import TextSegment, ComparisonConfig


@dataclass
class SimilarityScore:
    """Container for different similarity measures."""
    semantic: float = 0.0
    fuzzy: float = 0.0
    phonetic: float = 0.0
    contextual: float = 0.0
    combined: float = 0.0


class SimilarityCalculator:
    """Computes semantic similarity between transcript segments."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize the similarity calculator with configuration."""
        self.config = config
        
        # Initialize embedding model if available
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(config.embedding_model)
            except Exception as e:
                print(f"Warning: Could not load embedding model {config.embedding_model}: {e}")
        
        # Initialize fuzzy matcher if available
        self.fuzzy_matcher = None
        if FUZZYWUZZY_AVAILABLE:
            self.fuzzy_matcher = FuzzyMatcher(config.fuzzy_threshold)
        
        # Initialize phonetic matcher if available
        self.phonetic_matcher = None
        if METAPHONE_AVAILABLE:
            self.phonetic_matcher = PhoneticMatcher(config.phonetic_threshold)
        
        # Initialize contextual matcher
        self.contextual_matcher = ContextualMatcher(config)
    
    def calculate_similarity(self, segment1: TextSegment, segment2: TextSegment) -> SimilarityScore:
        """
        Calculate similarity between two text segments using multiple measures.
        
        Args:
            segment1: First text segment
            segment2: Second text segment
            
        Returns:
            SimilarityScore object with individual and combined scores
        """
        # Calculate individual similarity scores
        semantic_sim = self._calculate_semantic_similarity(segment1.text, segment2.text)
        fuzzy_sim = self._calculate_fuzzy_similarity(segment1.text, segment2.text)
        phonetic_sim = self._calculate_phonetic_similarity(segment1.text, segment2.text)
        contextual_sim = self._calculate_contextual_similarity(segment1, segment2)
        
        # Combine scores with weights
        combined_score = (
            self.config.semantic_weight * semantic_sim +
            self.config.fuzzy_weight * fuzzy_sim +
            self.config.phonetic_weight * phonetic_sim +
            self.config.contextual_weight * contextual_sim
        )
        
        return SimilarityScore(
            semantic=semantic_sim,
            fuzzy=fuzzy_sim,
            phonetic=phonetic_sim,
            contextual=contextual_sim,
            combined=combined_score
        )
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings."""
        if not self.embedding_model or not text1 or not text2:
            return 0.0
        
        try:
            # Generate embeddings
            emb1 = self.embedding_model.encode([text1])
            emb2 = self.embedding_model.encode([text2])
            
            # Calculate cosine similarity
            sim = cosine_similarity(emb1, emb2)[0][0]
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, sim))
        except Exception as e:
            print(f"Warning: Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy string similarity."""
        if not self.fuzzy_matcher or not text1 or not text2:
            return 0.0
        
        try:
            return self.fuzzy_matcher.calculate_similarity(text1, text2)
        except Exception as e:
            print(f"Warning: Error calculating fuzzy similarity: {e}")
            return 0.0
    
    def _calculate_phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity."""
        if not self.phonetic_matcher or not text1 or not text2:
            return 0.0
        
        try:
            return self.phonetic_matcher.calculate_similarity(text1, text2)
        except Exception as e:
            print(f"Warning: Error calculating phonetic similarity: {e}")
            return 0.0
    
    def _calculate_contextual_similarity(self, segment1: TextSegment, segment2: TextSegment) -> float:
        """Calculate contextual similarity based on surrounding words."""
        try:
            return self.contextual_matcher.calculate_similarity(segment1, segment2)
        except Exception as e:
            print(f"Warning: Error calculating contextual similarity: {e}")
            return 0.0


class FuzzyMatcher:
    """Handles fuzzy string matching."""
    
    def __init__(self, threshold: float = 0.7):
        """Initialize with similarity threshold."""
        self.threshold = threshold
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy similarity using Levenshtein distance."""
        if not text1 or not text2:
            return 0.0
        
        # Use different fuzzy matching methods and take the maximum
        ratios = [
            fuzz.ratio(text1, text2) / 100.0,
            fuzz.partial_ratio(text1, text2) / 100.0,
            fuzz.token_sort_ratio(text1, text2) / 100.0,
            fuzz.token_set_ratio(text1, text2) / 100.0
        ]
        
        # Take the maximum ratio
        max_ratio = max(ratios)
        
        # Apply threshold
        return max_ratio if max_ratio >= self.threshold else 0.0


class PhoneticMatcher:
    """Handles phonetic similarity matching."""
    
    def __init__(self, threshold: float = 0.6):
        """Initialize with similarity threshold."""
        self.threshold = threshold
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity using Metaphone."""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Get phonetic codes
            code1 = metaphone.doublemetaphone(text1.lower())
            code2 = metaphone.doublemetaphone(text2.lower())
            
            # Check primary and secondary codes
            if code1[0] == code2[0]:
                return 1.0  # Exact phonetic match
            elif code1[0] == code2[1] or code1[1] == code2[0]:
                return 0.8  # Close phonetic match
            else:
                # Calculate similarity based on code overlap
                sim = self._calculate_code_similarity(code1, code2)
                return sim if sim >= self.threshold else 0.0
        except Exception:
            return 0.0
    
    def _calculate_code_similarity(self, code1: Tuple[str, str], code2: Tuple[str, str]) -> float:
        """Calculate similarity between phonetic codes."""
        # Simple character-based similarity for codes
        all_codes = [code1[0], code1[1], code2[0], code2[1]]
        valid_codes = [c for c in all_codes if c]
        
        if not valid_codes:
            return 0.0
        
        # Count matching characters at each position
        max_len = max(len(c) for c in valid_codes)
        total_sim = 0.0
        count = 0
        
        for i in range(max_len):
            chars_at_pos = []
            for c in valid_codes:
                if i < len(c):
                    chars_at_pos.append(c[i])
            
            if chars_at_pos:
                # Calculate similarity for characters at this position
                unique_chars = set(chars_at_pos)
                pos_sim = 1.0 / len(unique_chars) if unique_chars else 0.0
                total_sim += pos_sim
                count += 1
        
        return total_sim / count if count > 0 else 0.0


class ContextualMatcher:
    """Handles contextual similarity based on surrounding words."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize with configuration."""
        self.config = config
        self.window_size = max(1, config.context_window_size)
    
    def calculate_similarity(self, segment1: TextSegment, segment2: TextSegment) -> float:
        """Calculate contextual similarity based on surrounding words."""
        # This is a simplified implementation
        # In a more sophisticated version, we would consider the actual context
        # from the transcript
        
        # For now, just check if the words are similar in length
        len1 = len(segment1.text.split())
        len2 = len(segment2.text.split())
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Calculate length similarity
        len_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
        
        return len_sim


class BatchSimilarityCalculator:
    """Calculates similarity for multiple pairs efficiently."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize with configuration."""
        self.config = config
        self.calculator = SimilarityCalculator(config)
    
    def calculate_batch_similarity(self, segments1: List[TextSegment], 
                                  segments2: List[TextSegment]) -> List[List[SimilarityScore]]:
        """
        Calculate similarity matrix between two lists of segments.
        
        Args:
            segments1: First list of text segments
            segments2: Second list of text segments
            
        Returns:
            2D matrix of similarity scores
        """
        # Initialize similarity matrix
        similarity_matrix = [[None for _ in segments2] for _ in segments1]
        
        # Calculate similarities
        for i, seg1 in enumerate(segments1):
            for j, seg2 in enumerate(segments2):
                similarity_matrix[i][j] = self.calculator.calculate_similarity(seg1, seg2)
        
        return similarity_matrix
    
    def find_best_matches(self, segments1: List[TextSegment], 
                         segments2: List[TextSegment]) -> List[Tuple[int, int, float]]:
        """
        Find best matching pairs between two lists of segments.
        
        Args:
            segments1: First list of text segments
            segments2: Second list of text segments
            
        Returns:
            List of tuples (index1, index2, similarity_score) for best matches
        """
        # Calculate similarity matrix
        sim_matrix = self.calculate_batch_similarity(segments1, segments2)
        
        # Find best matches using greedy approach
        matches = []
        used_indices1 = set()
        used_indices2 = set()
        
        # Continue until we can't find more matches above threshold
        while True:
            best_score = 0.0
            best_i, best_j = -1, -1
            
            # Find the best remaining match
            for i in range(len(segments1)):
                if i in used_indices1:
                    continue
                
                for j in range(len(segments2)):
                    if j in used_indices2:
                        continue
                    
                    score = sim_matrix[i][j].combined
                    if score > best_score:
                        best_score = score
                        best_i, best_j = i, j
            
            # Check if we found a match above threshold
            if best_score >= self.config.semantic_threshold:
                matches.append((best_i, best_j, best_score))
                used_indices1.add(best_i)
                used_indices2.add(best_j)
            else:
                break
        
        return matches