#!/usr/bin/env python3
"""
VAD segment combination and splitting for ASR.

This module provides stateless functions for combining raw VAD segments into
natural conversation chunks and splitting long segments at natural pause
boundaries, respecting transcriber limits.
"""

from typing import List, Optional
import logging

from local_transcribe.processing.vad.types import (
    VADSegment,
    CombinedSegment,
    SegmentCombinationConfig,
)
from local_transcribe.lib.program_logger import get_logger, log_debug

logger = logging.getLogger(__name__)


def segment_for_asr(
    raw_segments: List[VADSegment],
    max_segment_duration: float = 30.0,
    config: Optional[SegmentCombinationConfig] = None
) -> List[CombinedSegment]:
    """Combine and split raw VAD segments into ASR-ready chunks.
    
    Uses pause analysis to split at natural boundaries.
    This is the main entry point for the segmentation logic.
    
    Args:
        raw_segments: Raw VAD segments from provider
        max_segment_duration: Maximum chunk duration (from transcriber capability)
        config: Optional custom configuration (uses interview defaults if None)
        
    Returns:
        List of CombinedSegment ready for ASR processing
    """
    if not raw_segments:
        return []
    
    # Create config if not provided
    if config is None:
        config = SegmentCombinationConfig(max_segment_duration=max_segment_duration)
    else:
        config.max_segment_duration = max_segment_duration
    
    # Step 1: Initial combination based on gap analysis
    combined = _initial_combination(raw_segments, config)
    
    # Step 2: Split long segments at natural boundaries
    final = _split_long_segments_enhanced(combined, config)
    
    # Step 3: Second pass for remaining long segments
    final = _split_long_segments_second_pass(final, config)
    
    logger.info(f"Final result: {len(final)} segments after combination and splitting")
    
    return final


def _should_combine_segments(
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


def _initial_combination(
    segments: List[VADSegment],
    config: SegmentCombinationConfig
) -> List[CombinedSegment]:
    """Initial combination of segments based on gap analysis.
    
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
        should_combine = _should_combine_segments(
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


def _calculate_split_score(
    segment: CombinedSegment,
    split_index: int,
    gap: float,
    first_seg_duration: float,
    second_seg_duration: float,
    config: SegmentCombinationConfig
) -> float:
    """Calculate score for a potential split point.
    
    Higher scores indicate better split points.
    
    Args:
        segment: The segment being split
        split_index: Index of the split point
        gap: Gap duration at the split point
        first_seg_duration: Duration of first resulting segment
        second_seg_duration: Duration of second resulting segment
        config: Configuration parameters
        
    Returns:
        Score for this split point (higher is better)
    """
    # Balance Score (40% weight)
    duration_ratio = min(first_seg_duration, second_seg_duration) / max(first_seg_duration, second_seg_duration)
    balance_score = duration_ratio * 0.4
    
    # Gap Score (30% weight)
    gap_score = min(gap / config.min_gap_for_primary_split, 1.0) * 0.3
    
    # Context Score (30% weight)
    context_score = 0.0
    if split_index > 1:
        # Add context from previous gap
        prev_gap = segment.segments[split_index - 1].start_s - segment.segments[split_index - 2].end_s
        context_score += (gap - prev_gap) * 0.1  # Reward if this gap is larger than previous
    
    if split_index < len(segment.segments) - 1:
        # Add context from next gap
        next_gap = segment.segments[split_index + 1].start_s - segment.segments[split_index].end_s
        context_score += (gap - next_gap) * 0.1  # Reward if this gap is larger than next
    
    context_score = max(0, context_score) * 0.3  # 30% weight for context, ensure non-negative
    
    # Total score
    total_score = balance_score + gap_score + context_score
    
    return total_score


def _find_best_split_points(
    segment: CombinedSegment,
    config: SegmentCombinationConfig
) -> List[int]:
    """Find the best split points for a segment.
    
    Args:
        segment: The segment to find split points for
        config: Configuration parameters
        
    Returns:
        List of split indices (positions in segment.segments)
    """
    potential_splits = []
    
    for i in range(1, len(segment.segments)):
        gap = segment.segments[i].start_s - segment.segments[i-1].end_s
        
        # Check gap against tiered threshold system
        if gap >= config.min_gap_for_tertiary_split:
            # Verify split would create meaningful segments
            first_seg_duration = segment.segments[i].start_s - segment.segments[0].start_s
            second_seg_duration = segment.segments[-1].end_s - segment.segments[i].start_s
            
            if first_seg_duration >= config.min_split_segment_duration and second_seg_duration >= config.min_split_segment_duration:
                score = _calculate_split_score(segment, i, gap, first_seg_duration, second_seg_duration, config)
                potential_splits.append((i, gap, score, first_seg_duration, second_seg_duration))
    
    # Sort by score and select best splits
    potential_splits.sort(key=lambda x: x[2], reverse=True)
    
    # Select best split points, ensuring they're not too close to each other
    selected_splits = []
    last_split_idx = 0
    
    for split_idx, gap, score, first_duration, second_duration in potential_splits:
        if len(selected_splits) >= config.max_splits_per_segment:
            break
            
        # Ensure this split is not too close to the previous one
        if split_idx - last_split_idx >= 2:  # At least 2 segments between splits
            # Check if this split creates segments of reasonable length
            if first_duration >= config.min_split_segment_duration and second_duration >= config.min_split_segment_duration:
                selected_splits.append(split_idx)
                last_split_idx = split_idx
    
    return selected_splits

def _find_force_split_points(
    segment: CombinedSegment,
    config: SegmentCombinationConfig
) -> List[int]:
    """Find split points for a segment that must be split, using relaxed criteria.
    
    This is a fallback when _find_best_split_points fails to find suitable splits
    but the segment still exceeds max_segment_duration. It uses more relaxed
    criteria to ensure segments can be split.
    
    Args:
        segment: The segment to find split points for
        config: Configuration parameters
        
    Returns:
        List of split indices (positions in segment.segments)
    """
    if len(segment.segments) < 2:
        return []
    
    # Calculate target segment duration
    target_duration = config.max_segment_duration * 0.8  # Aim for 80% of max
    
    # Find all gaps with their indices and calculate cumulative durations
    gaps_with_info = []
    for i in range(1, len(segment.segments)):
        gap = segment.segments[i].start_s - segment.segments[i-1].end_s
        cumulative_duration = segment.segments[i].start_s - segment.segments[0].start_s
        remaining_duration = segment.segments[-1].end_s - segment.segments[i].start_s
        gaps_with_info.append((i, gap, cumulative_duration, remaining_duration))
    
    # Sort by gap size (prefer larger gaps)
    gaps_with_info.sort(key=lambda x: x[1], reverse=True)
    
    selected_splits = []
    current_start = segment.segments[0].start_s
    segments_covered = set()
    
    # Greedy approach: select split points that create segments close to target duration
    for split_idx, gap, cumulative_duration, remaining_duration in gaps_with_info:
        if split_idx in segments_covered:
            continue
            
        # Calculate the duration of segment if we split here
        segment_before_split = cumulative_duration - (current_start - segment.segments[0].start_s)
        
        # Accept this split if:
        # 1. It creates a reasonable segment before the split (>= 3s or whatever we can get)
        # 2. The segment after wouldn't be too tiny (>= 3s)
        min_acceptable = 3.0  # Minimum acceptable segment duration for force split
        
        if segment_before_split >= min_acceptable and remaining_duration >= min_acceptable:
            selected_splits.append(split_idx)
            segments_covered.add(split_idx)
            
            # Check if we've split enough
            # Recalculate from the last split point to see if remaining segment is OK
            last_split_start = segment.segments[split_idx].start_s
            remaining_from_split = segment.segments[-1].end_s - last_split_start
            
            if remaining_from_split <= config.max_segment_duration:
                break
                
            # Update current_start for next iteration
            current_start = last_split_start
    
    # If we still haven't found enough splits, try time-based approach
    if not selected_splits:
        # Find split point closest to target duration
        best_split = None
        best_distance = float('inf')
        
        for i in range(1, len(segment.segments)):
            duration_to_here = segment.segments[i].start_s - segment.segments[0].start_s
            distance = abs(duration_to_here - target_duration)
            
            if distance < best_distance:
                best_distance = distance
                best_split = i
        
        if best_split is not None:
            selected_splits = [best_split]
    
    return sorted(selected_splits)

def _split_long_segments_enhanced(
    segments: List[CombinedSegment],
    config: SegmentCombinationConfig
) -> List[CombinedSegment]:
    """Enhanced splitting logic for long segments.
    
    Args:
        segments: List of combined segments
        config: Configuration parameters
        
    Returns:
        List of combined segments with long segments split
    """
    final_segments = []
    
    for segment in segments:
        if segment.duration_s <= config.max_segment_duration:
            final_segments.append(segment)
            continue
        
        # Find best split points
        split_points = _find_best_split_points(segment, config)
        
        if not split_points:
            # No good split points found, keep as is
            final_segments.append(segment)
            continue
        
        # Split the segment at the identified points
        original_segments = segment.segments
        split_groups = []
        current_group = [original_segments[0]]
        
        for i in range(1, len(original_segments) + 1):
            if i in split_points:
                # Finalize current group and start new one
                split_groups.append(current_group)
                current_group = [original_segments[i]]
            elif i < len(original_segments):
                current_group.append(original_segments[i])
        
        # Add the last group
        if current_group:
            split_groups.append(current_group)
        
        # Convert groups to CombinedSegments
        for group in split_groups:
            if group:
                final_segments.append(CombinedSegment(group))
    
    return final_segments


def _split_long_segments_second_pass(
    segments: List[CombinedSegment],
    config: SegmentCombinationConfig
) -> List[CombinedSegment]:
    """Second-pass splitting for very long segments using recursive queue.
    
    Args:
        segments: List of combined segments
        config: Configuration parameters
        
    Returns:
        List of combined segments with remaining long segments split
    """
    final_segments = []
    
    for segment in segments:
        if segment.duration_s <= config.max_segment_duration:
            final_segments.append(segment)
            continue
        
        # Apply enhanced splitting
        enhanced_splits = _split_long_segments_enhanced([segment], config)
        
        # Handle segments that are still too long with recursive splitting
        for enhanced_segment in enhanced_splits:
            if enhanced_segment.duration_s <= config.max_segment_duration:
                final_segments.append(enhanced_segment)
            else:
                # Use a queue for recursive splitting of very long segments
                recursively_split_segments = []
                split_queue = [enhanced_segment]
                
                while split_queue:
                    current_segment = split_queue.pop(0)
                    
                    if current_segment.duration_s <= config.max_segment_duration:
                        recursively_split_segments.append(current_segment)
                        continue
                    
                    # Find best split points for this very long segment
                    split_points = _find_best_split_points(current_segment, config)
                    
                    if not split_points:
                        # No good split points found - try force split
                        split_points = _find_force_split_points(current_segment, config)
                        
                        if not split_points:
                            # Even force split didn't work, keep as is (very rare)
                            logger.warning(
                                f"Unable to split segment {current_segment.start_s:.2f}s - "
                                f"{current_segment.end_s:.2f}s ({current_segment.duration_s:.2f}s)"
                            )
                            recursively_split_segments.append(current_segment)
                            continue
                    
                    # Split the segment
                    original_segments = current_segment.segments
                    split_groups = []
                    current_group = [original_segments[0]]
                    
                    for i in range(1, len(original_segments) + 1):
                        if i in split_points:
                            # Finalize current group and add to results
                            if current_group:
                                split_groups.append(current_group)
                            current_group = [original_segments[i]] if i < len(original_segments) else []
                        elif i < len(original_segments):
                            current_group.append(original_segments[i])
                    
                    # Add the last group
                    if current_group:
                        split_groups.append(current_group)
                    
                    # Add split groups back to queue for further processing if needed
                    for group in split_groups:
                        if group:
                            new_segment = CombinedSegment(group)
                            if new_segment.duration_s > config.max_segment_duration:
                                split_queue.append(new_segment)
                            else:
                                recursively_split_segments.append(new_segment)
                
                # Add all recursively split segments
                final_segments.extend(recursively_split_segments)
    
    return final_segments


# Convenience class for stateful usage (optional)
class VADSegmenter:
    """Convenience class for VAD segmentation with stored configuration.
    
    This class wraps the stateless functions for easier use when the
    same configuration is used multiple times.
    """
    
    def __init__(
        self,
        max_segment_duration: float = 30.0,
        config: Optional[SegmentCombinationConfig] = None
    ):
        """Initialize the segmenter.
        
        Args:
            max_segment_duration: Maximum chunk duration (from transcriber capability)
            config: Optional custom configuration (uses interview defaults if None)
        """
        self.logger = get_logger()
        
        if config is None:
            self.config = SegmentCombinationConfig(
                max_segment_duration=max_segment_duration
            )
        else:
            self.config = config
            self.config.max_segment_duration = max_segment_duration
    
    def segment_for_asr(
        self,
        raw_segments: List[VADSegment]
    ) -> List[CombinedSegment]:
        """Combine and split raw VAD segments into ASR-ready chunks.
        
        Args:
            raw_segments: Raw VAD segments from provider
            
        Returns:
            List of CombinedSegment ready for ASR processing
        """
        return segment_for_asr(
            raw_segments,
            max_segment_duration=self.config.max_segment_duration,
            config=self.config
        )
