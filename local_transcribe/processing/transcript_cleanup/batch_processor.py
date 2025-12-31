#!/usr/bin/env python3
"""
Batch processor for LLM transcript cleanup.

This module handles:
- Batching turns into groups for efficient LLM processing
- Formatting batches with speaker labels for the LLM
- Parsing LLM responses back to individual turns
- Applying cleaned text to TranscriptFlow while preserving structure
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from copy import deepcopy

from local_transcribe.processing.turn_building.turn_building_data_structures import (
    TranscriptFlow, HierarchicalTurn, InterjectionSegment
)
from local_transcribe.framework.plugin_interfaces import WordSegment

logger = logging.getLogger(__name__)


@dataclass
class TurnBatch:
    """A batch of turns to be processed together."""
    turn_indices: List[int]  # Indices into the original turns list
    original_texts: List[str]  # Original text for each turn
    speakers: List[str]  # Speaker for each turn
    formatted_text: str = ""  # Combined text sent to LLM
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = sum(len(text.split()) for text in self.original_texts)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_words_per_batch: int = 500  # Target max words per batch
    max_turns_per_batch: int = 20  # Hard limit on turns per batch
    speaker_label_format: str = "{speaker}:"  # Format for speaker labels
    turn_separator: str = "\n"  # Separator between turns


class BatchProcessor:
    """
    Processes transcript turns in batches through an LLM cleanup provider.
    
    The processor:
    1. Groups turns into batches based on word count
    2. Formats each batch with speaker labels
    3. Sends batches to the LLM for cleanup
    4. Parses responses back to individual turns
    5. Creates a new TranscriptFlow with cleaned text
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
    
    def process_transcript(
        self,
        transcript: TranscriptFlow,
        cleanup_provider: Any,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> TranscriptFlow:
        """
        Process entire transcript through LLM cleanup.
        
        Args:
            transcript: Original TranscriptFlow to clean
            cleanup_provider: LLM cleanup provider with transcript_cleanup_segment method
            progress_callback: Optional callback(batch_num, total_batches, status_msg)
            
        Returns:
            New TranscriptFlow with cleaned turn text
        """
        # Create batches
        batches = create_batches_from_transcript(transcript, self.config)
        total_batches = len(batches)
        
        logger.info(f"Created {total_batches} batches from {len(transcript.turns)} turns")
        
        # Process each batch
        cleaned_texts_by_index: Dict[int, str] = {}
        
        for batch_num, batch in enumerate(batches, 1):
            if progress_callback:
                progress_callback(batch_num, total_batches, f"Processing batch {batch_num}/{total_batches}")
            
            # Send to LLM
            cleaned_batch_text = cleanup_provider.transcript_cleanup_segment(
                batch.formatted_text,
                timeout=120  # Allow 2 minutes per batch
            )
            
            # Parse response back to individual turns
            parsed_texts = self._parse_batch_response(
                cleaned_batch_text,
                batch.speakers,
                batch.original_texts
            )
            
            # Map back to turn indices
            for idx, cleaned_text in zip(batch.turn_indices, parsed_texts):
                cleaned_texts_by_index[idx] = cleaned_text
            
            logger.debug(f"Batch {batch_num}/{total_batches} processed: {len(parsed_texts)} turns cleaned")
        
        # Apply cleaned text to create new TranscriptFlow
        return apply_cleaned_text_to_transcript(transcript, cleaned_texts_by_index)
    
    def _parse_batch_response(
        self,
        response_text: str,
        expected_speakers: List[str],
        original_texts: List[str]
    ) -> List[str]:
        """
        Parse LLM response back into individual turn texts.
        
        Args:
            response_text: The LLM's cleaned response
            expected_speakers: List of speakers in order
            original_texts: Original texts (fallback if parsing fails)
            
        Returns:
            List of cleaned texts, one per turn
        """
        # Build regex pattern to split by speaker labels
        # Handle variations in speaker label formatting
        speaker_pattern = self._build_speaker_pattern(expected_speakers)
        
        # Try to split by speaker labels
        parts = re.split(speaker_pattern, response_text, flags=re.IGNORECASE)
        
        # Filter out empty parts and clean up
        parts = [p.strip() for p in parts if p and p.strip()]
        
        # If we got the right number of parts, use them
        if len(parts) == len(expected_speakers):
            return parts
        
        # Try alternative parsing: look for each speaker label in order
        cleaned_texts = []
        remaining_text = response_text
        
        for i, speaker in enumerate(expected_speakers):
            # Find this speaker's label
            speaker_label = f"{speaker}:"
            label_match = re.search(
                rf'\b{re.escape(speaker)}\s*:',
                remaining_text,
                re.IGNORECASE
            )
            
            if label_match:
                # Find the next speaker label (if any)
                next_speaker_idx = None
                for j in range(i + 1, len(expected_speakers)):
                    next_match = re.search(
                        rf'\b{re.escape(expected_speakers[j])}\s*:',
                        remaining_text[label_match.end():],
                        re.IGNORECASE
                    )
                    if next_match:
                        next_speaker_idx = label_match.end() + next_match.start()
                        break
                
                if next_speaker_idx:
                    text = remaining_text[label_match.end():next_speaker_idx].strip()
                    remaining_text = remaining_text[next_speaker_idx:]
                else:
                    text = remaining_text[label_match.end():].strip()
                    remaining_text = ""
                
                cleaned_texts.append(text)
            else:
                # Fallback to original if we can't find the speaker
                logger.warning(f"Could not find speaker '{speaker}' in LLM response, using original text")
                cleaned_texts.append(original_texts[i] if i < len(original_texts) else "")
        
        # Validate we got enough texts
        while len(cleaned_texts) < len(expected_speakers):
            idx = len(cleaned_texts)
            logger.warning(f"Missing cleaned text for turn {idx}, using original")
            cleaned_texts.append(original_texts[idx] if idx < len(original_texts) else "")
        
        return cleaned_texts[:len(expected_speakers)]
    
    def _build_speaker_pattern(self, speakers: List[str]) -> str:
        """Build regex pattern to split text by speaker labels."""
        # Escape speaker names and create alternation
        escaped_speakers = [re.escape(s) for s in speakers]
        # Match speaker name followed by colon (with optional whitespace)
        return r'(?:' + '|'.join(escaped_speakers) + r')\s*:\s*'


def create_batches_from_transcript(
    transcript: TranscriptFlow,
    config: Optional[BatchConfig] = None
) -> List[TurnBatch]:
    """
    Create batches of turns for LLM processing.
    
    Groups consecutive turns until word limit is reached,
    respecting the max_turns_per_batch limit.
    
    Args:
        transcript: TranscriptFlow to batch
        config: Batch configuration
        
    Returns:
        List of TurnBatch objects
    """
    config = config or BatchConfig()
    batches: List[TurnBatch] = []
    
    current_indices: List[int] = []
    current_texts: List[str] = []
    current_speakers: List[str] = []
    current_word_count = 0
    
    for idx, turn in enumerate(transcript.turns):
        turn_word_count = len(turn.text.split())
        
        # Check if adding this turn would exceed limits
        would_exceed_words = current_word_count + turn_word_count > config.max_words_per_batch
        would_exceed_turns = len(current_indices) >= config.max_turns_per_batch
        
        # Start new batch if needed
        if current_indices and (would_exceed_words or would_exceed_turns):
            batch = _create_batch(current_indices, current_texts, current_speakers, config)
            batches.append(batch)
            current_indices = []
            current_texts = []
            current_speakers = []
            current_word_count = 0
        
        # Add turn to current batch
        current_indices.append(idx)
        current_texts.append(turn.text)
        current_speakers.append(turn.primary_speaker)
        current_word_count += turn_word_count
    
    # Don't forget the last batch
    if current_indices:
        batch = _create_batch(current_indices, current_texts, current_speakers, config)
        batches.append(batch)
    
    return batches


def _create_batch(
    indices: List[int],
    texts: List[str],
    speakers: List[str],
    config: BatchConfig
) -> TurnBatch:
    """Create a TurnBatch with formatted text."""
    # Format the batch text with speaker labels
    formatted_lines = []
    for speaker, text in zip(speakers, texts):
        label = config.speaker_label_format.format(speaker=speaker)
        formatted_lines.append(f"{label} {text}")
    
    formatted_text = config.turn_separator.join(formatted_lines)
    
    return TurnBatch(
        turn_indices=indices.copy(),
        original_texts=texts.copy(),
        speakers=speakers.copy(),
        formatted_text=formatted_text
    )


def apply_cleaned_text_to_transcript(
    original: TranscriptFlow,
    cleaned_texts: Dict[int, str]
) -> TranscriptFlow:
    """
    Create a new TranscriptFlow with cleaned turn text.
    
    Preserves:
    - Turn structure (id, speaker, start, end)
    - Interjections (unchanged - typically single words)
    - Metadata (with cleanup flag added)
    
    Clears/Updates:
    - Turn text (replaced with cleaned version)
    - Word list (cleared - timing now invalid)
    - Word-based metrics (marked as stale)
    
    Args:
        original: Original TranscriptFlow
        cleaned_texts: Mapping of turn index to cleaned text
        
    Returns:
        New TranscriptFlow with cleaned text
    """
    # Deep copy to avoid modifying original
    new_turns: List[HierarchicalTurn] = []
    
    for idx, turn in enumerate(original.turns):
        # Get cleaned text or keep original
        cleaned_text = cleaned_texts.get(idx, turn.text)
        
        # Create new turn with cleaned text but preserved structure
        new_turn = HierarchicalTurn(
            turn_id=turn.turn_id,
            primary_speaker=turn.primary_speaker,
            start=turn.start,
            end=turn.end,
            text=cleaned_text,
            words=[],  # Clear words - timing no longer valid
            interjections=deepcopy(turn.interjections),  # Preserve interjections
            flow_continuity=turn.flow_continuity,
            turn_type=turn.turn_type,
        )
        
        # Manually set word count and duration since we cleared words
        new_turn.word_count = len(cleaned_text.split())
        new_turn.duration = turn.duration
        
        # Recalculate speaking rate with new word count
        if new_turn.duration > 0:
            new_turn.speaking_rate = round((new_turn.word_count / new_turn.duration) * 60, 1)
        
        new_turns.append(new_turn)
    
    # Create new metadata with cleanup flag
    new_metadata = deepcopy(original.metadata)
    new_metadata['llm_cleaned'] = True
    new_metadata['word_timing_available'] = False
    new_metadata['cleanup_stats'] = {
        'original_word_count': sum(len(t.text.split()) for t in original.turns),
        'cleaned_word_count': sum(len(t.text.split()) for t in new_turns),
    }
    
    # Recalculate conversation metrics
    total_words = sum(t.word_count for t in new_turns)
    new_conversation_metrics = deepcopy(original.conversation_metrics)
    new_conversation_metrics['total_words_cleaned'] = total_words
    
    # Recalculate speaker statistics
    new_speaker_stats = {}
    for speaker in original.speakers:
        speaker_turns = [t for t in new_turns if t.primary_speaker == speaker]
        if speaker_turns:
            total_speaker_words = sum(t.word_count for t in speaker_turns)
            total_duration = sum(t.duration for t in speaker_turns)
            
            new_speaker_stats[speaker] = {
                **deepcopy(original.speaker_statistics.get(speaker, {})),
                'total_words_cleaned': total_speaker_words,
                'avg_speaking_rate_cleaned': round((total_speaker_words / total_duration) * 60, 1) if total_duration > 0 else 0,
            }
    
    return TranscriptFlow(
        turns=new_turns,
        metadata=new_metadata,
        conversation_metrics=new_conversation_metrics,
        speaker_statistics=new_speaker_stats
    )


def format_turn_for_llm(turn: HierarchicalTurn, include_interjections: bool = False) -> str:
    """
    Format a single turn for LLM processing.
    
    Args:
        turn: The turn to format
        include_interjections: Whether to include interjection text inline
        
    Returns:
        Formatted string with speaker label and text
    """
    text = turn.text
    
    if include_interjections and turn.interjections:
        # Could optionally weave interjections into the text
        # For now, we keep the main text as-is
        pass
    
    return f"{turn.primary_speaker}: {text}"
