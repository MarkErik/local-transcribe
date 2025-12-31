"""
Sequence alignment using dynamic programming.

Implements Needleman-Wunsch-style alignment for matching source words
to target word dictionaries with timestamps.
"""

from typing import List, Dict, Any, Tuple, Optional
from local_transcribe.providers.common.mfa_alignment.word_similarity import WordSimilarity


class SequenceAligner:
    """
    Dynamic programming sequence aligner for word sequences.
    
    Aligns source words (from transcript) to target words (from MFA)
    using a gap-penalty scoring system.
    """
    
    DEFAULT_GAP_PENALTY = -0.5
    
    def __init__(
        self, 
        gap_penalty: float = DEFAULT_GAP_PENALTY,
        word_similarity: Optional[WordSimilarity] = None
    ):
        """
        Initialize SequenceAligner.
        
        Args:
            gap_penalty: Penalty for gaps in alignment (default: -0.5)
            word_similarity: WordSimilarity instance for scoring (optional)
        """
        self.gap_penalty = gap_penalty
        self.word_similarity = word_similarity or WordSimilarity()
    
    def align_word_sequences(
        self, 
        source_words: List[str], 
        target_words: List[Dict[str, Any]]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Align source and target word sequences using dynamic programming.
        
        Uses Needleman-Wunsch-style alignment with:
        - Match/mismatch scoring based on word similarity
        - Gap penalties for insertions/deletions
        
        Args:
            source_words: List of words from original transcript
            target_words: List of word dictionaries from MFA alignment
            
        Returns:
            List of alignment pairs (source_idx, target_idx) where None
            indicates a gap in that sequence
        """
        if not source_words and not target_words:
            return []
        if not source_words:
            return [(None, i) for i in range(len(target_words))]
        if not target_words:
            return [(i, None) for i in range(len(source_words))]
        
        n = len(source_words)
        m = len(target_words)
        
        # Extract target texts for comparison
        target_texts = [wd["text"] for wd in target_words]
        
        # Initialize DP table and backpointer
        dp = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]
        backptr = [[(0, 0, '') for _ in range(m + 1)] for _ in range(n + 1)]
        
        # Initialize first row and column with gap penalties
        for i in range(1, n + 1):
            dp[i][0] = dp[i-1][0] + self.gap_penalty
            backptr[i][0] = (i-1, 0, 'source_gap')
        
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j-1] + self.gap_penalty
            backptr[0][j] = (0, j-1, 'target_gap')
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                source_word = source_words[i-1]
                target_word = target_texts[j-1]
                
                # Calculate similarity score
                similarity = self.word_similarity.calculate_similarity(source_word, target_word)
                
                # Consider three operations:
                # 1. Match/substitute
                match_score = dp[i-1][j-1] + similarity
                
                # 2. Gap in target (source has extra word)
                source_gap_score = dp[i-1][j] + self.gap_penalty
                
                # 3. Gap in source (target has extra word - likely a split)
                target_gap_score = dp[i][j-1] + self.gap_penalty
                
                # Choose best operation
                if match_score >= source_gap_score and match_score >= target_gap_score:
                    dp[i][j] = match_score
                    backptr[i][j] = (i-1, j-1, 'match')
                elif source_gap_score >= target_gap_score:
                    dp[i][j] = source_gap_score
                    backptr[i][j] = (i-1, j, 'source_gap')
                else:
                    dp[i][j] = target_gap_score
                    backptr[i][j] = (i, j-1, 'target_gap')
        
        # Traceback to find optimal alignment
        return self._traceback_alignment(backptr, n, m)
    
    def _traceback_alignment(
        self, 
        backptr: List[List[Tuple[int, int, str]]], 
        n: int, 
        m: int
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        """
        Perform traceback to construct alignment from DP table.
        
        Args:
            backptr: Backpointer table from DP computation
            n: Length of source sequence
            m: Length of target sequence
            
        Returns:
            List of alignment pairs in forward order
        """
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i == 0:
                # Remaining target words have no source match
                alignment.append((None, j-1))
                j -= 1
            elif j == 0:
                # Remaining source words have no target match
                alignment.append((i-1, None))
                i -= 1
            else:
                # Get the move that led to this cell
                prev_i, prev_j, move = backptr[i][j]
                
                if move == 'match':
                    alignment.append((i-1, j-1))
                    i -= 1
                    j -= 1
                elif move == 'source_gap':
                    alignment.append((i-1, None))
                    i -= 1
                else:  # target_gap
                    alignment.append((None, j-1))
                    j -= 1
        
        alignment.reverse()
        return alignment
