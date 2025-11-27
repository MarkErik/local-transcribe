#!/usr/bin/env python3
"""
Shared base logic for turn builders.

This module provides common functionality used by both the local (rule-based)
and LLM-enhanced turn builders, including:
- Word stream merging from multiple speakers
- Speaker segment grouping
- Interjection pattern detection
- Metrics calculation
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_progress, log_debug
from local_transcribe.providers.turn_builders.split_audio_data_structures import (
    RawSegment,
    InterjectionSegment,
    HierarchicalTurn,
    TranscriptFlow,
    TurnBuilderConfig
)


def merge_word_streams(words: List[WordSegment]) -> List[WordSegment]:
    """
    Merge word segments from all speakers into a unified timeline.
    
    Args:
        words: List of WordSegment objects from all speakers (may be unsorted)
        
    Returns:
        Sorted list of WordSegment objects by start time
    """
    if not words:
        return []
    
    # Sort by start time, then by end time for ties
    sorted_words = sorted(words, key=lambda w: (w.start, w.end))
    
    log_debug(f"Merged {len(sorted_words)} words from all speakers")
    
    return sorted_words


def group_by_speaker(words: List[WordSegment]) -> List[RawSegment]:
    """
    Group consecutive words by speaker into raw segments.
    
    Args:
        words: Sorted list of WordSegment objects
        
    Returns:
        List of RawSegment objects, each containing consecutive words from one speaker
    """
    if not words:
        return []
    
    segments: List[RawSegment] = []
    current_words: List[WordSegment] = []
    current_speaker: Optional[str] = None
    
    for word in words:
        speaker = word.speaker or "Unknown"
        
        if speaker != current_speaker:
            # Speaker change - save current segment and start new one
            if current_words:
                segment = _create_raw_segment(current_words, current_speaker)
                segments.append(segment)
            
            current_words = [word]
            current_speaker = speaker
        else:
            # Same speaker - add to current segment
            current_words.append(word)
    
    # Don't forget the last segment
    if current_words:
        segment = _create_raw_segment(current_words, current_speaker)
        segments.append(segment)
    
    # Calculate gaps between segments
    _calculate_segment_gaps(segments)
    
    log_debug(f"Grouped into {len(segments)} raw segments")
    
    return segments


def _create_raw_segment(words: List[WordSegment], speaker: str) -> RawSegment:
    """Create a RawSegment from a list of words."""
    text = " ".join(w.text for w in words)
    return RawSegment(
        speaker=speaker,
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=words
    )


def _calculate_segment_gaps(segments: List[RawSegment]) -> None:
    """Calculate gap_before and gap_after for each segment."""
    for i, segment in enumerate(segments):
        # Gap before
        if i > 0:
            segment.gap_before = segment.start - segments[i - 1].end
        else:
            segment.gap_before = None
        
        # Gap after
        if i < len(segments) - 1:
            segment.gap_after = segments[i + 1].start - segment.end
        else:
            segment.gap_after = None


def detect_interjection_type(text: str, config: TurnBuilderConfig) -> Optional[str]:
    """
    Detect the type of interjection based on text patterns.
    
    Args:
        text: The text to analyze (will be lowercased)
        config: Configuration with pattern lists
        
    Returns:
        "acknowledgment", "question", "reaction", "fragment", or None if no pattern matches
    """
    text_lower = text.lower().strip()
    
    # Remove punctuation for matching
    text_clean = re.sub(r'[^\w\s]', '', text_lower)
    
    # Check acknowledgment patterns
    for pattern in config.acknowledgment_patterns:
        if pattern in text_clean or text_clean == pattern:
            return "acknowledgment"
    
    # Check question patterns (also look for question marks)
    if "?" in text:
        return "question"
    for pattern in config.question_patterns:
        if pattern in text_clean or text_clean == pattern:
            return "question"
    
    # Check reaction patterns
    for pattern in config.reaction_patterns:
        if pattern in text_clean or text_clean == pattern:
            return "reaction"
    
    # Check fragment patterns (incomplete phrases)
    if hasattr(config, 'fragment_patterns'):
        for pattern in config.fragment_patterns:
            if pattern in text_clean or text_clean == pattern:
                return "fragment"
    
    return None


def calculate_interjection_confidence(
    segment: RawSegment,
    prev_segment: Optional[RawSegment],
    next_segment: Optional[RawSegment],
    config: TurnBuilderConfig
) -> Tuple[float, str]:
    """
    Calculate confidence that a segment is an interjection.
    
    Uses multiple signals:
    - Duration (short = more likely interjection)
    - Word count (few words = more likely interjection)
    - Pattern matching (known patterns = more likely)
    - Context (sandwiched between same speaker = more likely)
    - Overlap (temporally overlapping with other speaker = more likely)
    
    Args:
        segment: The segment to analyze
        prev_segment: Previous segment (may be None)
        next_segment: Next segment (may be None)
        config: Configuration with thresholds
        
    Returns:
        Tuple of (confidence score 0-1, detected interjection type or "unclear")
    """
    score = 0.0
    weights_used = 0.0
    
    # Get configurable thresholds with defaults
    very_short_word_count = getattr(config, 'very_short_word_count', 2)
    very_short_duration = getattr(config, 'very_short_duration', 1.0)
    
    # 1. Duration score (weight: 0.2)
    duration_weight = 0.2
    if segment.duration < 0.5:
        score += duration_weight * 1.0
    elif segment.duration < 1.0:
        score += duration_weight * 0.8
    elif segment.duration < 1.5:
        score += duration_weight * 0.5
    elif segment.duration < config.max_interjection_duration:
        score += duration_weight * 0.3
    else:
        score += duration_weight * 0.0
    weights_used += duration_weight
    
    # 2. Word count score (weight: 0.25)
    word_weight = 0.25
    if segment.word_count == 1:
        score += word_weight * 1.0
    elif segment.word_count <= 2:
        score += word_weight * 0.9
    elif segment.word_count <= 3:
        score += word_weight * 0.7
    elif segment.word_count <= 4:
        score += word_weight * 0.5
    elif segment.word_count <= config.max_interjection_words:
        score += word_weight * 0.3
    else:
        score += word_weight * 0.0
    weights_used += word_weight
    
    # 3. Pattern match score (weight: 0.2)
    pattern_weight = 0.2
    interjection_type = detect_interjection_type(segment.text, config)
    if interjection_type:
        score += pattern_weight * 1.0
    else:
        # Even without a pattern match, very short segments get partial credit
        if segment.word_count <= very_short_word_count:
            score += pattern_weight * 0.4
        else:
            score += pattern_weight * 0.0
    weights_used += pattern_weight
    
    # 4. Context score (weight: 0.2)
    context_weight = 0.2
    context_score = 0.0
    
    if prev_segment and next_segment:
        # Check if sandwiched between same speaker
        if prev_segment.speaker == next_segment.speaker and prev_segment.speaker != segment.speaker:
            context_score = 1.0
        # Check if sandwiched between different speakers (still could be interjection)
        elif prev_segment.speaker != segment.speaker and next_segment.speaker != segment.speaker:
            context_score = 0.6
    elif prev_segment:
        if prev_segment.speaker != segment.speaker:
            context_score = 0.5
    elif next_segment:
        if next_segment.speaker != segment.speaker:
            context_score = 0.5
    
    # Boost for small/negative gaps (indicates overlapping or rapid speech)
    if segment.gap_before is not None:
        if segment.gap_before < 0:  # Overlapping!
            context_score = min(1.0, context_score + 0.4)
        elif segment.gap_before < 0.3:
            context_score = min(1.0, context_score + 0.2)
    if segment.gap_after is not None:
        if segment.gap_after < 0:  # Overlapping!
            context_score = min(1.0, context_score + 0.4)
        elif segment.gap_after < 0.3:
            context_score = min(1.0, context_score + 0.2)
    
    score += context_weight * context_score
    weights_used += context_weight
    
    # 5. Overlap detection score (weight: 0.15) - NEW
    overlap_weight = 0.15
    overlap_score = 0.0
    
    # Check for temporal overlap with adjacent segments from different speakers
    if prev_segment and prev_segment.speaker != segment.speaker:
        if segment.start < prev_segment.end:  # Overlaps with previous
            overlap_score = max(overlap_score, 0.8)
    if next_segment and next_segment.speaker != segment.speaker:
        if segment.end > next_segment.start:  # Overlaps with next
            overlap_score = max(overlap_score, 0.8)
    
    score += overlap_weight * overlap_score
    weights_used += overlap_weight
    
    # Normalize score
    final_score = score / weights_used if weights_used > 0 else 0.0
    
    # Boost for very short segments (high confidence these are interjections)
    if segment.word_count <= very_short_word_count and segment.duration < very_short_duration:
        final_score = min(1.0, final_score + 0.15)
    
    return (final_score, interjection_type or "unclear")


def classify_segments_rule_based(
    segments: List[RawSegment],
    config: TurnBuilderConfig
) -> Tuple[List[RawSegment], List[RawSegment]]:
    """
    Classify segments as primary turns or interjections using rules only.
    
    Args:
        segments: List of raw segments to classify
        config: Configuration with thresholds
        
    Returns:
        Tuple of (primary_segments, interjection_segments)
    """
    primary_segments: List[RawSegment] = []
    interjection_segments: List[RawSegment] = []
    
    for i, segment in enumerate(segments):
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i < len(segments) - 1 else None
        
        # Calculate confidence
        confidence, interjection_type = calculate_interjection_confidence(
            segment, prev_seg, next_seg, config
        )
        
        # Store results in segment
        segment.interjection_confidence = confidence
        
        # Hard rules: definitely not an interjection
        if segment.duration > config.max_interjection_duration:
            segment.is_interjection = False
            segment.classification_method = "rule_duration"
            primary_segments.append(segment)
            continue
        
        if segment.word_count > config.max_interjection_words:
            segment.is_interjection = False
            segment.classification_method = "rule_word_count"
            primary_segments.append(segment)
            continue
        
        # Special case: very short segments (1-2 words) that overlap or are sandwiched
        # are almost certainly interjections, even without pattern match
        very_short_word_count = getattr(config, 'very_short_word_count', 2)
        is_sandwiched = (
            prev_seg and next_seg and 
            prev_seg.speaker == next_seg.speaker and 
            prev_seg.speaker != segment.speaker
        )
        has_overlap = (
            (prev_seg and segment.start < prev_seg.end) or
            (next_seg and segment.end > next_seg.start)
        )
        
        if segment.word_count <= very_short_word_count and (is_sandwiched or has_overlap):
            segment.is_interjection = True
            segment.classification_method = "rule_very_short_contextual"
            interjection_segments.append(segment)
            continue
        
        # Classification based on confidence
        if confidence >= config.high_confidence_threshold:
            segment.is_interjection = True
            segment.classification_method = "rule_high_confidence"
            interjection_segments.append(segment)
        elif confidence <= config.low_confidence_threshold:
            segment.is_interjection = False
            segment.classification_method = "rule_low_confidence"
            primary_segments.append(segment)
        else:
            # Ambiguous case - use additional heuristics
            # If it's short (<=3 words) and has any contextual signal, lean toward interjection
            if segment.word_count <= 3 and (is_sandwiched or has_overlap or confidence >= 0.4):
                segment.is_interjection = True
                segment.classification_method = "rule_ambiguous_short"
                interjection_segments.append(segment)
            else:
                segment.is_interjection = False
                segment.classification_method = "rule_ambiguous_default"
                primary_segments.append(segment)
    
    log_debug(f"Classified {len(primary_segments)} primary, {len(interjection_segments)} interjections")
    
    return primary_segments, interjection_segments


def determine_interrupt_level(
    interjection: RawSegment,
    primary_turn_start: float,
    primary_turn_end: float
) -> str:
    """
    Determine the interrupt level of an interjection.
    
    Args:
        interjection: The interjection segment
        primary_turn_start: Start time of the primary turn
        primary_turn_end: End time of the primary turn
        
    Returns:
        "none", "low", "medium", or "high"
    """
    # Check if interjection overlaps with primary turn timing
    overlap_start = max(interjection.start, primary_turn_start)
    overlap_end = min(interjection.end, primary_turn_end)
    
    if overlap_start >= overlap_end:
        # No temporal overlap
        return "none"
    
    overlap_duration = overlap_end - overlap_start
    interjection_ratio = overlap_duration / interjection.duration if interjection.duration > 0 else 0
    
    if interjection_ratio < 0.3:
        return "low"
    elif interjection_ratio < 0.7:
        return "medium"
    else:
        return "high"


def create_interjection_from_segment(
    segment: RawSegment,
    primary_turn_start: float,
    primary_turn_end: float,
    config: TurnBuilderConfig
) -> InterjectionSegment:
    """
    Create an InterjectionSegment from a classified RawSegment.
    """
    interjection_type = detect_interjection_type(segment.text, config) or "unclear"
    interrupt_level = determine_interrupt_level(segment, primary_turn_start, primary_turn_end)
    
    return InterjectionSegment(
        speaker=segment.speaker,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        words=segment.words,
        confidence=segment.interjection_confidence,
        interjection_type=interjection_type,
        interrupt_level=interrupt_level,
        classification_method=segment.classification_method
    )


def assemble_hierarchical_turns(
    segments: List[RawSegment],
    config: TurnBuilderConfig
) -> List[HierarchicalTurn]:
    """
    Assemble classified segments into hierarchical turns.
    
    This function takes segments that have been classified as either
    primary turns or interjections and assembles them into HierarchicalTurn
    objects, attaching interjections to the appropriate primary turns.
    
    Args:
        segments: All segments with is_interjection set
        config: Configuration with thresholds
        
    Returns:
        List of HierarchicalTurn objects
    """
    if not segments:
        return []
    
    # Separate into primary and interjection segments
    primary_segments = [s for s in segments if not s.is_interjection]
    interjection_segments = [s for s in segments if s.is_interjection]
    
    # If no primary segments, convert interjections to primary turns
    if not primary_segments:
        log_debug("No primary segments found, converting interjections to primary turns")
        primary_segments = interjection_segments
        interjection_segments = []
        for seg in primary_segments:
            seg.is_interjection = False
    
    # Build primary turns, merging consecutive same-speaker segments
    hierarchical_turns: List[HierarchicalTurn] = []
    current_turn_segments: List[RawSegment] = []
    current_speaker: Optional[str] = None
    turn_id = 1
    
    for segment in primary_segments:
        should_merge = (
            current_speaker == segment.speaker and
            current_turn_segments and
            (segment.start - current_turn_segments[-1].end) <= config.max_gap_to_merge_turns
        )
        
        if should_merge:
            # Same speaker within gap threshold - merge
            current_turn_segments.append(segment)
        else:
            # Different speaker or gap too large - finalize current turn and start new
            if current_turn_segments:
                hturn = _create_hierarchical_turn(current_turn_segments, turn_id)
                hierarchical_turns.append(hturn)
                turn_id += 1
            
            current_turn_segments = [segment]
            current_speaker = segment.speaker
    
    # Finalize last turn
    if current_turn_segments:
        hturn = _create_hierarchical_turn(current_turn_segments, turn_id)
        hierarchical_turns.append(hturn)
    
    # Attach interjections to appropriate turns
    _attach_interjections_to_turns(hierarchical_turns, interjection_segments, config)
    
    log_debug(f"Assembled {len(hierarchical_turns)} hierarchical turns")
    
    return hierarchical_turns


def _create_hierarchical_turn(segments: List[RawSegment], turn_id: int) -> HierarchicalTurn:
    """Create a HierarchicalTurn from one or more RawSegments."""
    # Combine all words and text
    all_words = []
    all_text_parts = []
    
    for segment in segments:
        all_words.extend(segment.words)
        all_text_parts.append(segment.text)
    
    return HierarchicalTurn(
        turn_id=turn_id,
        primary_speaker=segments[0].speaker,
        start=segments[0].start,
        end=segments[-1].end,
        text=" ".join(all_text_parts),
        words=all_words,
        interjections=[]
    )


def _attach_interjections_to_turns(
    turns: List[HierarchicalTurn],
    interjections: List[RawSegment],
    config: TurnBuilderConfig
) -> None:
    """Attach interjection segments to the most appropriate primary turns."""
    for ij_segment in interjections:
        best_turn = None
        best_score = -1
        
        for turn in turns:
            # Calculate how well this interjection fits with this turn
            score = _score_interjection_fit(ij_segment, turn)
            if score > best_score:
                best_score = score
                best_turn = turn
        
        if best_turn is not None:
            # Create InterjectionSegment and attach
            interjection = create_interjection_from_segment(
                ij_segment,
                best_turn.start,
                best_turn.end,
                config
            )
            best_turn.interjections.append(interjection)
            
            # Recalculate turn metrics
            best_turn._calculate_flow_continuity()
            best_turn._determine_turn_type()


def _score_interjection_fit(interjection: RawSegment, turn: HierarchicalTurn) -> float:
    """
    Score how well an interjection fits with a turn.
    
    Higher score = better fit.
    """
    # Different speaker required
    if interjection.speaker == turn.primary_speaker:
        return -1
    
    score = 0.0
    
    # Temporal overlap or adjacency
    if interjection.start >= turn.start and interjection.end <= turn.end:
        # Interjection is fully within turn bounds
        score += 1.0
    elif interjection.start >= turn.start - 0.5 and interjection.end <= turn.end + 0.5:
        # Interjection is near turn bounds (within 0.5s)
        score += 0.7
    elif interjection.start >= turn.start - 1.0 and interjection.end <= turn.end + 1.0:
        # Interjection is close to turn bounds (within 1s)
        score += 0.4
    else:
        # Interjection is far from turn
        return 0.0
    
    return score


def calculate_conversation_metrics(turns: List[HierarchicalTurn]) -> Dict[str, Any]:
    """
    Calculate conversation-level metrics.
    
    Args:
        turns: List of hierarchical turns
        
    Returns:
        Dictionary of metrics
    """
    if not turns:
        return {}
    
    # Basic counts
    total_turns = len(turns)
    total_interjections = sum(len(t.interjections) for t in turns)
    
    # Duration metrics
    total_duration = turns[-1].end - turns[0].start if turns else 0
    
    # Speaker statistics
    speakers = set(t.primary_speaker for t in turns)
    speaker_stats = {}
    
    for speaker in speakers:
        speaker_turns = [t for t in turns if t.primary_speaker == speaker]
        speaker_interjections = []
        for t in turns:
            speaker_interjections.extend([ij for ij in t.interjections if ij.speaker == speaker])
        
        speaker_stats[speaker] = {
            "turn_count": len(speaker_turns),
            "interjection_count": len(speaker_interjections),
            "total_words": sum(t.word_count for t in speaker_turns),
            "total_speaking_time": sum(t.duration for t in speaker_turns),
            "average_turn_duration": (
                sum(t.duration for t in speaker_turns) / len(speaker_turns)
                if speaker_turns else 0
            ),
            "average_speaking_rate": (
                sum(t.speaking_rate for t in speaker_turns) / len(speaker_turns)
                if speaker_turns else 0
            )
        }
    
    # Flow metrics
    avg_flow_continuity = (
        sum(t.flow_continuity for t in turns) / len(turns)
        if turns else 1.0
    )
    
    # Turn type distribution
    turn_types = {}
    for t in turns:
        turn_types[t.turn_type] = turn_types.get(t.turn_type, 0) + 1
    
    # Interjection type distribution
    interjection_types = {}
    for t in turns:
        for ij in t.interjections:
            interjection_types[ij.interjection_type] = interjection_types.get(ij.interjection_type, 0) + 1
    
    return {
        "total_turns": total_turns,
        "total_interjections": total_interjections,
        "total_duration": total_duration,
        "average_flow_continuity": avg_flow_continuity,
        "turn_type_distribution": turn_types,
        "interjection_type_distribution": interjection_types,
        "speaker_statistics": speaker_stats
    }


def build_transcript_flow(
    turns: List[HierarchicalTurn],
    config: TurnBuilderConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> TranscriptFlow:
    """
    Build a complete TranscriptFlow from hierarchical turns.
    
    Args:
        turns: List of hierarchical turns
        config: Configuration used for building
        metadata: Optional additional metadata
        
    Returns:
        Complete TranscriptFlow object with metrics
    """
    conversation_metrics = calculate_conversation_metrics(turns)
    
    return TranscriptFlow(
        turns=turns,
        metadata=metadata or {},
        speaker_statistics=conversation_metrics.get("speaker_statistics", {}),
        conversation_metrics=conversation_metrics
    )
