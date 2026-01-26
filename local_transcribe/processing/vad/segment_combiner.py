#!/usr/bin/env python3
"""
VAD segment combination for ASR.

This module provides stateless functions for combining raw VAD segments into
natural conversation chunks based on gap analysis and pause patterns.
"""

from typing import List
import logging

from local_transcribe.processing.vad.types import (
    VADSegment,
    CombinedSegment,
    SegmentCombinationConfig,
)

logger = logging.getLogger(__name__)


def should_combine_segments(
    prev_segment: VADSegment,
    current_segment: VADSegment,
    gap: float,
    config: SegmentCombinationConfig
) -> bool:
    """Determine if two consecutive segments should be combined based on gap and context.
    
    Args:
        prev_segment: The previous segment
        current_segment: The current segment
        gap: Time gap between segments (current_segment.start_s - prev_segment.end_s)
        config: Configuration parameters
        
    Returns:
        True if segments should be combined, False otherwise
    """
    # Rule 1: Micro-pauses (< threshold) - always combine
    if gap < config.micro_pause_threshold:
        return True
    
    # Rule 2: Topic change boundaries (>= natural threshold) - never combine
    if gap >= config.natural_boundary_threshold:
        return False
    
    # Rule 3: Short disfluency handling
    # If segment < disfluency_threshold and adjacent to other segments, combine
    if (prev_segment.duration_s < config.disfluency_threshold or
        current_segment.duration_s < config.disfluency_threshold):
        # Check if there's enough context to preserve disfluencies
        context_duration = prev_segment.duration_s + current_segment.duration_s
        if context_duration >= config.min_disfluency_context:
            return True
    
    # Rule 4: Thinking pauses (< thinking threshold) - combine with context evaluation
    if gap < config.thinking_pause_threshold:
        # Additional heuristic: if both segments are short, combine them
        if prev_segment.duration_s < 3.0 and current_segment.duration_s < 3.0:
            return True
        
        # If the gap is small relative to segment lengths, combine
        avg_segment_duration = (prev_segment.duration_s + current_segment.duration_s) / 2
        if gap < avg_segment_duration * 0.5:  # Gap is less than half the average segment duration
            return True
    
    # Default: don't combine
    return False


def combine_segments(
    segments: List[VADSegment],
    config: SegmentCombinationConfig
) -> List[CombinedSegment]:
    """Combine segments based on gap analysis.
    
    This is the main entry point for combining VAD segments into larger chunks.
    
    Args:
        segments: List of VAD segments to combine
        config: Configuration parameters
        
    Returns:
        List of combined segments
    """
    if not segments:
        return []
    
    # Sort segments chronologically
    sorted_segments = sorted(segments, key=lambda x: x.start_s)
    
    combined_segments = []
    current_group = [sorted_segments[0]]
    
    logger.info(f"Starting combination of {len(sorted_segments)} segments")
    
    for i in range(1, len(sorted_segments)):
        previous_segment = sorted_segments[i-1]
        current_segment = sorted_segments[i]
        
        # Calculate gap between segments
        gap = current_segment.start_s - previous_segment.end_s
        
        # Determine if segments should be combined
        should_combine = should_combine_segments(
            current_group[-1], current_segment, gap, config
        )
        
        if should_combine:
            current_group.append(current_segment)
        else:
            # Finalize current group and start new one
            combined_segments.append(CombinedSegment(current_group))
            current_group = [current_segment]
    
    # Add the last group
    if current_group:
        combined_segments.append(CombinedSegment(current_group))
    
    logger.info(f"Combined into {len(combined_segments)} segments")
    return combined_segments
