#!/usr/bin/env python3
"""
Turn building processors.

This module provides turn building functionality for grouping word segments
into conversational turns with interjection detection.

Available processors:
- build_turns_combined_audio: For single audio files with diarization
- build_turns_split_audio: Turn builder for split audio mode (not yet implemented)
- build_turns_vad_split_audio: VAD-first approach for split audio files

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
    'build_turns_vad_split_audio',
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


def build_turns_vad_split_audio(speaker_audio_files, **kwargs) -> TranscriptFlow:
    """
    Build conversation turns using VAD-first approach for split audio files.
    
    This is the recommended approach for interview-style recordings where
    each speaker has their own audio track.
    
    Args:
        speaker_audio_files: Dict mapping speaker_id -> audio file path
        **kwargs: Configuration options including:
            - transcriber_provider: ASR provider (required)
            - config: VADBlockBuilderConfig
            - intermediate_dir: Path for debug files
            - models_dir: Path to model cache
            - vad_threshold: VAD sensitivity (0-1)
            - remote_granite_url: URL for remote transcription
    
    Returns:
        TranscriptFlow with hierarchical turn structure
    """
    from local_transcribe.processing.turn_building.vad_turn_builder import (
        build_turns_vad_split_audio as _build_turns_vad
    )
    return _build_turns_vad(speaker_audio_files, **kwargs)


def build_turns(words_or_files, mode: str, **kwargs) -> TranscriptFlow:
    """
    Build conversation turns from word segments or audio files.
    
    This is the main entry point for turn building. It automatically
    selects the appropriate processor based on the mode:
    - "combined_audio": Uses rule-based turn building with interjection detection
    - "split_audio": Turn building for split audio mode (delegates to combined for now)
    - "vad_split_audio": VAD-first approach for split audio files (recommended)
    
    Args:
        words_or_files: Either List[WordSegment] or Dict[str, str] (speaker -> audio path)
        mode: Processing mode - "combined_audio", "split_audio", or "vad_split_audio"
        **kwargs: Additional configuration options including:
            - intermediate_dir: Path for intermediate files and audit logs
            - max_interjection_duration: Max duration for interjections (default: 2.0s)
            - max_interjection_words: Max word count for interjections (default: 5)
            - max_gap_to_merge_turns: Max gap to merge same-speaker turns (default: 3.0s)
            
            For vad_split_audio mode:
            - transcriber_provider: ASR provider (required)
            - config: VADBlockBuilderConfig
            - vad_threshold: VAD sensitivity (0-1)
            - remote_granite_url: URL for remote transcription
    
    Returns:
        TranscriptFlow with hierarchical turn structure
    """
    if mode == "combined_audio":
        return build_turns_combined_audio(words_or_files, **kwargs)
    elif mode == "split_audio":
        return build_turns_split_audio(words_or_files, **kwargs)
    elif mode == "vad_split_audio":
        return build_turns_vad_split_audio(words_or_files, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'combined_audio', 'split_audio', or 'vad_split_audio'")
