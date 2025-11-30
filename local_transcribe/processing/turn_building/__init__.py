#!/usr/bin/env python3
"""
Turn building processors.

This module provides turn building functionality for grouping word segments
into conversational turns with interjection detection.

Available processors:
- build_turns_combined_audio: For single audio files with diarization
- build_turns_split_audio: turn builder for split audio mode (not yet implemented)

The main entry point is the `build_turns` function which automatically
selects the appropriate processor based on the mode.
"""

from local_transcribe.processing.turn_building.turn_building_data_structures import (
    RawSegment,
    InterjectionSegment,
    HierarchicalTurn,
    TranscriptFlow,
    TurnBuilderConfig
)

from local_transcribe.processing.turn_building.combined_audio_turn_builder import (
    build_turns_combined_audio
)

from local_transcribe.processing.turn_building.turn_building_base import (
    TurnBuildingAuditLog,
    normalize_word_timestamps,
    group_words_by_speaker,
    classify_segments,
    classify_interjection_type,
    is_potential_interjection,
    calculate_interrupt_level,
    raw_segment_to_interjection
)


def build_turns_split_audio(words, **kwargs) -> TranscriptFlow:
    """
    Build conversation turns from split audio word segments.
    
    NOTE: This is a placeholder. Split audio mode turn building
    is not yet implemented. For now, it delegates to combined audio mode.
    
    Args:
        words: List of WordSegment with speaker assignments
        **kwargs: Configuration options
    
    Returns:
        TranscriptFlow with hierarchical turn structure
    """
    from local_transcribe.lib.program_logger import get_logger
    get_logger().warning("Split audio turn building not yet implemented, using combined audio mode")
    return build_turns_combined_audio(words, **kwargs)


__all__ = [
    # Main entry point
    'build_turns',
    # Mode-specific functions
    'build_turns_combined_audio',
    'build_turns_split_audio',
    # Data structures
    'RawSegment',
    'InterjectionSegment',
    'HierarchicalTurn',
    'TranscriptFlow',
    'TurnBuilderConfig',
    # Base utilities
    'TurnBuildingAuditLog',
    'normalize_word_timestamps',
    'group_words_by_speaker',
    'classify_segments',
    'classify_interjection_type',
    'is_potential_interjection',
    'calculate_interrupt_level',
    'raw_segment_to_interjection',
]


def build_turns(words, mode: str, **kwargs) -> TranscriptFlow:
    """
    Build conversation turns from word segments.
    
    This is the main entry point for turn building. It automatically
    selects the appropriate processor based on the mode:
    - "combined_audio": Uses rule-based turn building with interjection detection
    - "split_audio": Turn building for split audio mode (delegates to combined for now)
    
    Args:
        words: List of WordSegment with speaker assignments
        mode: Processing mode - "combined_audio" or "split_audio"
        **kwargs: Additional configuration options including:
            - intermediate_dir: Path for intermediate files and audit logs
            - max_interjection_duration: Max duration for interjections (default: 2.0s)
            - max_interjection_words: Max word count for interjections (default: 5)
            - max_gap_to_merge_turns: Max gap to merge same-speaker turns (default: 3.0s)
    
    Returns:
        TranscriptFlow with hierarchical turn structure
    """
    if mode == "combined_audio":
        return build_turns_combined_audio(words, **kwargs)
    elif mode == "split_audio":
        return build_turns_split_audio(words, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'combined_audio' or 'split_audio'")
