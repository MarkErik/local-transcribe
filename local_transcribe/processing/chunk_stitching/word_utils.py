#!/usr/bin/env python3
"""
Word utility functions for chunk stitching.

This module provides utilities for working with words in different formats
(string or dict with timestamps) and word comparison functions.
"""

from typing import List, Any, Dict, Union
from difflib import SequenceMatcher


# =============================================================================
# Word format utilities
# =============================================================================

def get_word_text(word: Union[str, Dict[str, Any]]) -> str:
    """
    Extract text from a word (handles both string and dict formats).
    
    Args:
        word: Either a string or a dict with 'text' key
        
    Returns:
        The word text
    """
    if isinstance(word, str):
        return word
    return word.get("text", "")


def get_word_texts(words: List[Union[str, Dict[str, Any]]]) -> List[str]:
    """
    Extract text from a list of words.
    
    Args:
        words: List of words (strings or dicts)
        
    Returns:
        List of word texts
    """
    return [get_word_text(w) for w in words]


def has_timestamps(words: List[Any]) -> bool:
    """
    Check if words have timestamp information.
    
    Args:
        words: List of words to check
        
    Returns:
        True if words are dicts with 'text' and 'start' keys
    """
    if not words:
        return False
    first_word = words[0]
    return isinstance(first_word, dict) and "text" in first_word and "start" in first_word


# =============================================================================
# Word comparison utilities
# =============================================================================

def words_similar(word1: str, word2: str, similarity_threshold: float = 0.7) -> bool:
    """
    Check if two words are similar using exact match or fuzzy matching.
    
    Args:
        word1: First word
        word2: Second word
        similarity_threshold: Threshold for SequenceMatcher ratio (default 0.7)
        
    Returns:
        True if words are considered similar
    """
    if word1 == word2:
        return True
    if word1.lower() == word2.lower():
        return True
    
    ratio = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
    return ratio >= similarity_threshold


def is_fuzzy_match(
    words1: List[str], 
    words2: List[str], 
    min_overlap_ratio: float = 0.6,
    similarity_threshold: float = 0.7
) -> bool:
    """
    Check if two word lists are fuzzy matches.
    
    Args:
        words1: First word list
        words2: Second word list
        min_overlap_ratio: Minimum ratio of matching words (default 0.6)
        similarity_threshold: Threshold for individual word similarity (default 0.7)
        
    Returns:
        True if the word lists are considered fuzzy matches
    """
    if len(words1) != len(words2):
        return False
    
    matches = 0
    for w1, w2 in zip(words1, words2):
        if words_similar(w1, w2, similarity_threshold):
            matches += 1
    
    return matches / len(words1) >= min_overlap_ratio


def is_partial_word_match(word1: str, word2: str) -> bool:
    """
    Check if one word could be a partial match for another.
    
    This handles cases like "generational" -> "rational" where a word
    gets cut off at chunk boundaries.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        True if one word is a partial match of the other
    """
    w1, w2 = word1.lower(), word2.lower()
    
    # One word contains the other
    if len(w1) > len(w2) * 1.5 and w2 in w1:
        return True
    if len(w2) > len(w1) * 1.5 and w1 in w2:
        return True
    
    # Check for significant suffix/prefix overlap
    min_len = min(len(w1), len(w2))
    for i in range(min_len // 2, min_len):
        if w1[-i:] == w2[-i:] or w1[:i] == w2[:i]:
            return True
    
    return False
