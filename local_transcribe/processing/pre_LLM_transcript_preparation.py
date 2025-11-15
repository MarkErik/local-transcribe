#!/usr/bin/env python3
"""
Pre-LLM transcript preparation module.

This module provides functions to prepare raw transcripts for LLM processing in the 
transcript cleanup step. It handles standardizing speaker labels, managing segment lengths,
preserving sentence boundaries, normalizing whitespace, handling special characters,
and maintaining context.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from local_transcribe.framework.plugin_interfaces import Turn

# Configure logger
logger = logging.getLogger(__name__)


def prepare_transcript_for_llm(
    turns: List[Turn],
    max_words_per_segment: int = 500,
    preparation_mode: str = "basic",
    standardize_speakers: bool = True,
    normalize_whitespace: bool = True,
    handle_special_chars: bool = True
) -> Dict[str, Any]:
    """
    Prepare transcript for LLM processing by applying various text transformations.
    
    Args:
        turns: List of Turn objects representing the conversation
        max_words_per_segment: Maximum words allowed in a segment before splitting
        standardize_speakers: Whether to standardize speaker labels
        normalize_whitespace: Whether to normalize whitespace in the transcript
        handle_special_chars: Whether to handle special characters
        
    Returns:
        Dictionary containing prepared transcript data with keys:
        - 'turns': List of processed Turn objects
        - 'segments': List of text segments ready for LLM processing
        - 'stats': Statistics about the preparation process
    """
    logger.info(f"Preparing {len(turns)} turns for LLM processing in {preparation_mode} mode")
    
    # Validate preparation mode
    if preparation_mode not in ["basic", "advanced"]:
        raise ValueError(f"Invalid preparation mode: {preparation_mode}. Must be 'basic' or 'advanced'")
    
    # Initialize statistics
    stats = {
        'original_turns': len(turns),
        'processed_turns': 0,
        'segments_created': 0,
        'words_processed': 0,
        'turns_split': 0,
        'preparation_mode': preparation_mode
    }
    
    try:
        # Create a copy of turns to avoid modifying the original
        processed_turns = []
        
        for turn in turns:
            processed_text = turn.text
            
            # Apply advanced preprocessing only in advanced mode
            if preparation_mode == "advanced":
                # Normalize whitespace if requested
                if normalize_whitespace:
                    processed_text = _normalize_whitespace(processed_text)
                
                # Handle special characters if requested
                if handle_special_chars:
                    processed_text = _handle_special_characters(processed_text)
            
            # Standardize speaker label if requested (applied in both modes)
            processed_speaker = turn.speaker
            if standardize_speakers:
                processed_speaker = _standardize_speaker_label(turn.speaker)
            
            # Create processed turn
            processed_turn = Turn(
                speaker=processed_speaker,
                start=turn.start,
                end=turn.end,
                text=processed_text
            )
            processed_turns.append(processed_turn)
            
            stats['processed_turns'] += 1
            stats['words_processed'] += len(processed_text.split())
        
        # Convert turns to segments for LLM processing
        segments = _convert_turns_to_segments(processed_turns, max_words_per_segment)
        stats['segments_created'] = len(segments)
        
        # Count how many turns were split into multiple segments
        if len(turns) > 0:
            stats['turns_split'] = len(segments) - len(turns)
        
        logger.info(f"Transcript preparation complete ({preparation_mode} mode): {stats['segments_created']} segments created from {stats['processed_turns']} turns")
        
        return {
            'turns': processed_turns,
            'segments': segments,
            'stats': stats
        }
        
    except Exception as e:
        logger.error(f"Error during transcript preparation: {str(e)}")
        raise


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text by replacing multiple spaces with single space
    and trimming leading/trailing whitespace.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    return text


def _handle_special_characters(text: str) -> str:
    """
    Handle special characters in the text to ensure compatibility with LLM processing.
    
    Args:
        text: Input text to process
        
    Returns:
        Text with special characters handled appropriately
    """
    # Replace common problematic Unicode characters with ASCII equivalents
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...', # Ellipsis
    }
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    return text


def _standardize_speaker_label(speaker: str) -> str:
    """
    Standardize speaker label format.
    
    Args:
        speaker: Original speaker label
        
    Returns:
        Standardized speaker label
    """
    if not speaker:
        return "Unknown"
    
    # Remove any special characters and normalize
    standardized = re.sub(r'[^\w\s-]', '', speaker)
    standardized = _normalize_whitespace(standardized)
    
    # Capitalize first letter of each word
    standardized = ' '.join(word.capitalize() for word in standardized.split())
    
    return standardized


def _convert_turns_to_segments(turns: List[Turn], max_words_per_segment: int) -> List[str]:
    """
    Convert turns to text segments suitable for LLM processing.
    
    Args:
        turns: List of Turn objects
        max_words_per_segment: Maximum words allowed in a segment
        
    Returns:
        List of text segments
    """
    # First convert turns to text segments with speaker labels
    text_segments = []
    for turn in turns:
        segment = f"{turn.speaker}: {turn.text}"
        text_segments.append(segment)
    
    # Split long segments while preserving sentence boundaries
    split_segments = split_long_lines(text_segments, max_words_per_segment)
    
    return split_segments


def split_long_lines(segments: List[str], max_words: int = 500) -> List[str]:
    """
    Split lines that exceed the max word limit while preserving speaker labels and sentence boundaries.
    
    Args:
        segments: List of text segments with speaker labels
        max_words: Maximum words allowed in a segment
        
    Returns:
        List of split segments
    """
    split_segments = []
    sentence_endings = [". ", "! ", "? ", ".", "!", "?"]

    for segment in segments:
        # Find where the speaker label ends (look for ": " after "SPEAKER_" or any speaker name)
        speaker_end = segment.find(": ")
        if speaker_end == -1:
            # No speaker label found, treat as plain text
            split_segments.extend(_split_text_by_sentences(segment, max_words))
            continue
            
        speaker_label = segment[:speaker_end]
        text = segment[speaker_end + 2:]
        
        sentences = _split_into_sentences(text, sentence_endings)
        grouped_sentences = _group_sentences(sentences, max_words, speaker_label)
        split_segments.extend(grouped_sentences)

    return split_segments


def _split_text_by_sentences(text: str, max_words: int) -> List[str]:
    """
    Split plain text (without speaker labels) by sentences while respecting max word limit.
    
    Args:
        text: Text to split
        max_words: Maximum words per segment
        
    Returns:
        List of text segments
    """
    sentence_endings = [". ", "! ", "? ", ".", "!", "?"]
    sentences = _split_into_sentences(text, sentence_endings)
    return _group_sentences_without_speaker(sentences, max_words)


def _split_into_sentences(text: str, sentence_endings: List[str]) -> List[str]:
    """
    Split text into sentences based on common sentence endings.
    
    Args:
        text: Text to split
        sentence_endings: List of sentence ending patterns
        
    Returns:
        List of sentences
    """
    sentences = []
    current_pos = 0
    
    while current_pos < len(text):
        next_end = float('inf')
        for ending in sentence_endings:
            pos = text.find(ending, current_pos)
            if pos != -1 and pos < next_end:
                next_end = pos + len(ending)
        
        if next_end == float('inf'):
            sentences.append(text[current_pos:])
            break
        else:
            sentences.append(text[current_pos:next_end])
            current_pos = next_end
    
    return sentences


def _group_sentences(
    sentences: List[str],
    max_words: int,
    speaker_label: str
) -> List[str]:
    """
    Group sentences into chunks that don't exceed max_words, preserving speaker labels.
    
    Args:
        sentences: List of sentences to group
        max_words: Maximum words per chunk
        speaker_label: Speaker label to prepend to each chunk
        
    Returns:
        List of grouped text segments with speaker labels
    """
    grouped_segments = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                grouped_segments.append(f"{speaker_label}: {chunk_text}")
            current_chunk = []
            current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        if chunk_text:
            grouped_segments.append(f"{speaker_label}: {chunk_text}")
    
    return grouped_segments


def _group_sentences_without_speaker(
    sentences: List[str],
    max_words: int
) -> List[str]:
    """
    Group sentences into chunks that don't exceed max_words, without speaker labels.
    
    Args:
        sentences: List of sentences to group
        max_words: Maximum words per chunk
        
    Returns:
        List of grouped text segments
    """
    grouped_segments = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                grouped_segments.append(chunk_text)
            current_chunk = []
            current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        if chunk_text:
            grouped_segments.append(chunk_text)
    
    return grouped_segments