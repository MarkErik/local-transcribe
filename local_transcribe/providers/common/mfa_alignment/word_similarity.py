"""
Word similarity calculations for alignment purposes.

Provides functions for normalizing words and calculating similarity scores
using Levenshtein distance and other metrics.
"""

from typing import Optional


class WordSimilarity:
    """
    Word similarity calculator for alignment operations.
    
    Provides methods for normalizing words and calculating similarity scores
    between word pairs using various matching strategies.
    """
    
    DEFAULT_MIN_SIMILARITY_THRESHOLD = 0.6
    
    def __init__(self, min_similarity_threshold: float = DEFAULT_MIN_SIMILARITY_THRESHOLD):
        """
        Initialize WordSimilarity.
        
        Args:
            min_similarity_threshold: Minimum threshold for meaningful similarity scores
        """
        self.min_similarity_threshold = min_similarity_threshold
    
    def normalize_word_for_matching(self, word: str) -> str:
        """
        Normalize word for comparison during alignment.
        
        Converts to lowercase and removes non-alphanumeric characters.
        
        Args:
            word: The word to normalize
            
        Returns:
            Normalized word string
        """
        if not word:
            return ""
        return ''.join(c.lower() for c in word if c.isalnum())
    
    def calculate_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words for alignment purposes.
        
        Uses a multi-strategy approach:
        1. Exact match (after normalization) -> 1.0
        2. Prefix match -> 0.7-0.9 based on length ratio
        3. Levenshtein distance ratio for partial matches
        
        Args:
            word1: First word to compare
            word2: Second word to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Handle empty strings
        if not word1 or not word2:
            return 0.0
        
        # Normalize both words
        norm1 = self.normalize_word_for_matching(word1)
        norm2 = self.normalize_word_for_matching(word2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Prefix match (handles "we" matching "we'll" or "well")
        if norm1.startswith(norm2) or norm2.startswith(norm1):
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return 0.7 + (0.2 * shorter / longer)
        
        # Character-level Levenshtein ratio
        return self._calculate_levenshtein_ratio(norm1, norm2)
    
    def _calculate_levenshtein_ratio(self, s1: str, s2: str) -> float:
        """
        Calculate Levenshtein distance ratio between two strings.
        
        Uses dynamic programming to compute edit distance and converts
        to a similarity ratio.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        len1, len2 = len(s1), len(s2)
        
        # Early exit for very different lengths
        if abs(len1 - len2) > max(len1, len2) // 2:
            return 0.0
        
        # Ensure s1 is the longer string for consistency
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        # Initialize DP table
        prev_row = list(range(len2 + 1))
        
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            
            prev_row = curr_row
        
        distance = prev_row[-1]
        max_len = max(len1, len2)
        ratio = 1.0 - (distance / max_len)
        
        # Only return meaningful similarity for close matches
        return ratio if ratio >= self.min_similarity_threshold else 0.0
