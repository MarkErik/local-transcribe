#!/usr/bin/env python3
"""
Combined audio turn building processor.

This processor is used for single audio files with diarization,
where all speakers are in one audio stream. It groups consecutive
words by speaker into hierarchical turns.
"""

from typing import List, Dict, Any
from datetime import datetime

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_progress, log_debug

from local_transcribe.processing.turn_building.data_structures import (
    TurnBuilderConfig,
    TranscriptFlow,
    HierarchicalTurn,
    RawSegment
)
from local_transcribe.processing.turn_building.base import (
    group_by_speaker,
    assemble_hierarchical_turns,
    build_transcript_flow
)


def build_turns_combined_audio(
    words: List[WordSegment],
    **kwargs
) -> TranscriptFlow:
    """
    Build turns from words with speaker assignments for combined audio mode.

    This is the main entry point for combined audio turn building.
    It groups words into turns based on speaker and timing, producing 
    a TranscriptFlow with hierarchical structure. In combined audio mode, 
    interjection detection is simpler since all audio comes from a single 
    source with accurate relative timing.

    Args:
        words: Word segments with speaker assignments from diarization
        **kwargs: Configuration options including:
            - max_gap_to_merge_turns: Max gap (seconds) to merge same-speaker turns
            - max_interjection_duration: Max duration for interjections
            - max_interjection_words: Max word count for interjections

    Returns:
        TranscriptFlow with hierarchical turn structure
    """
    if not words:
        log_progress("No words provided to turn builder")
        return TranscriptFlow(turns=[], metadata={"builder": "combined_audio", "error": "no_words"})
    
    # Create and update config from kwargs
    config = TurnBuilderConfig()
    _update_config_from_kwargs(config, kwargs)
    
    log_progress(f"Building turns from {len(words)} word segments (combined audio)")
    
    # Step 1: Group by speaker into raw segments
    log_debug("Step 1: Grouping by speaker")
    segments = group_by_speaker(words)
    log_progress(f"Grouped into {len(segments)} raw segments")
    
    # Step 2: For combined audio, we do simpler classification
    # Short segments between same-speaker segments are interjections
    log_debug("Step 2: Classifying segments")
    _classify_segments_simple(segments, config)
    
    primary_count = sum(1 for s in segments if not s.is_interjection)
    interjection_count = sum(1 for s in segments if s.is_interjection)
    log_progress(f"Classification: {primary_count} primary, {interjection_count} interjections")
    
    # Step 3: Assemble hierarchical turns
    log_debug("Step 3: Assembling hierarchical turns")
    hierarchical_turns = assemble_hierarchical_turns(segments, config)
    log_progress(f"Assembled {len(hierarchical_turns)} hierarchical turns")
    
    # Step 4: Build TranscriptFlow with metrics
    log_debug("Step 4: Building TranscriptFlow")
    transcript_flow = build_transcript_flow(
        hierarchical_turns,
        config,
        metadata={
            "builder": "combined_audio",
            "mode": "combined_audio",
            "timestamp": datetime.now().isoformat(),
            "total_words": len(words),
            "total_segments": len(segments)
        }
    )
    
    log_progress(f"Turn building complete: {transcript_flow.total_turns} turns, {transcript_flow.total_interjections} interjections")
    
    return transcript_flow


def _update_config_from_kwargs(config: TurnBuilderConfig, kwargs: Dict[str, Any]) -> None:
    """Update configuration from kwargs."""
    if 'max_interjection_duration' in kwargs:
        config.max_interjection_duration = kwargs['max_interjection_duration']
    if 'max_interjection_words' in kwargs:
        config.max_interjection_words = kwargs['max_interjection_words']
    if 'max_gap_to_merge_turns' in kwargs:
        config.max_gap_to_merge_turns = kwargs['max_gap_to_merge_turns']


def _classify_segments_simple(segments: List[RawSegment], config: TurnBuilderConfig) -> None:
    """
    Simple classification for combined audio mode.
    
    In combined audio, timing is more reliable so we use simpler rules:
    - Very short segments (≤2 words, <1s) sandwiched between same speaker = interjection
    - Everything else = primary turn
    """
    for i, segment in enumerate(segments):
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i < len(segments) - 1 else None
        
        # Default: not an interjection
        segment.is_interjection = False
        segment.classification_method = "rule_default"
        
        # Hard limits: can't be an interjection if too long
        if segment.duration > config.max_interjection_duration:
            continue
        if segment.word_count > config.max_interjection_words:
            continue
        
        # Check if sandwiched between same speaker
        is_sandwiched = (
            prev_seg and next_seg and
            prev_seg.speaker == next_seg.speaker and
            prev_seg.speaker != segment.speaker
        )
        
        # Very short segments that are sandwiched = interjection
        if is_sandwiched and segment.word_count <= 2 and segment.duration < 1.0:
            segment.is_interjection = True
            segment.classification_method = "rule_sandwiched_short"
            segment.interjection_confidence = 0.9
        # Short segments with acknowledgment patterns
        elif is_sandwiched and segment.word_count <= 3:
            text_lower = segment.text.lower().strip()
            if any(p in text_lower for p in config.acknowledgment_patterns):
                segment.is_interjection = True
                segment.classification_method = "rule_sandwiched_pattern"
                segment.interjection_confidence = 0.85
