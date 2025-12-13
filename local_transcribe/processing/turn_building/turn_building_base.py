#!/usr/bin/env python3
"""
Shared base logic for turn builders.

This module provides common functionality used by turn builders, including:
- Timestamp normalization and sorting
- Word stream grouping by speaker
- Interjection pattern detection and classification
- Gap and interrupt level calculations
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_progress, log_debug, get_logger
from local_transcribe.processing.turn_building.turn_building_data_structures import (
    RawSegment,
    InterjectionSegment,
    HierarchicalTurn,
    TranscriptFlow,
    TurnBuilderConfig
)


# =============================================================================
# Audit Logging
# =============================================================================

class TurnBuildingAuditLog:
    """
    Audit logger for turn building operations.
    
    Captures detailed information about each step of the turn building
    process for debugging and analysis.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.output_dir = output_dir
        self.entries: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, category: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log an audit entry."""
        if not self.enabled:
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_ms": (datetime.now() - self.start_time).total_seconds() * 1000,
            "category": category,
            "message": message,
            "data": data or {}
        }
        self.entries.append(entry)
        
        # Also log to program logger for real-time visibility
        log_debug(f"[TurnBuilding:{category}] {message}")
        if data:
            log_debug(f"  Data: {json.dumps(data, default=str)[:500]}")
    
    def log_segment_classification(
        self, 
        segment: RawSegment, 
        is_interjection: bool,
        confidence: float,
        interjection_type: str,
        reason: str
    ):
        """Log a segment classification decision."""
        self.log(
            "classification",
            f"Segment '{segment.text[:50]}...' -> {'INTERJECTION' if is_interjection else 'PRIMARY'}",
            {
                "speaker": segment.speaker,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "word_count": segment.word_count,
                "text": segment.text,
                "is_interjection": is_interjection,
                "confidence": confidence,
                "interjection_type": interjection_type,
                "reason": reason
            }
        )
    
    def log_turn_created(self, turn: HierarchicalTurn, merged_from: int = 1):
        """Log turn creation."""
        self.log(
            "turn_created",
            f"Turn {turn.turn_id}: [{turn.primary_speaker}] {turn.word_count} words, "
            f"{len(turn.interjections)} interjections",
            {
                "turn_id": turn.turn_id,
                "speaker": turn.primary_speaker,
                "start": turn.start,
                "end": turn.end,
                "word_count": turn.word_count,
                "duration": turn.duration,
                "interjection_count": len(turn.interjections),
                "flow_continuity": turn.flow_continuity,
                "turn_type": turn.turn_type,
                "merged_from_segments": merged_from,
                "text_preview": turn.text[:100] + "..." if len(turn.text) > 100 else turn.text
            }
        )
    
    def log_interjection_attached(
        self, 
        interjection: InterjectionSegment, 
        turn_id: int,
        reason: str
    ):
        """Log interjection attachment to a turn."""
        self.log(
            "interjection_attached",
            f"Interjection '{interjection.text}' -> Turn {turn_id}",
            {
                "interjection_speaker": interjection.speaker,
                "interjection_text": interjection.text,
                "interjection_start": interjection.start,
                "interjection_type": interjection.interjection_type,
                "target_turn_id": turn_id,
                "attachment_reason": reason
            }
        )
    
    def save(self, filename: str = "turn_building_audit.json"):
        """Save audit log to file."""
        if not self.output_dir:
            get_logger().warning("Cannot save audit log: no output directory specified")
            return
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_entries": len(self.entries),
                "entries": self.entries
            }, f, indent=2, default=str)
        
        log_progress(f"Audit log saved to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the audit log."""
        categories: Dict[str, int] = {}
        for entry in self.entries:
            cat = entry["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "categories": categories,
            "duration_ms": (datetime.now() - self.start_time).total_seconds() * 1000
        }


# =============================================================================
# Timestamp Normalization
# =============================================================================

def normalize_word_timestamps(
    words: List[WordSegment],
    audit_log: Optional[TurnBuildingAuditLog] = None
) -> List[WordSegment]:
    """
    Sort words by start time and handle timestamp anomalies.
    
    Args:
        words: List of WordSegment objects (may be unsorted or have anomalies)
        audit_log: Optional audit logger for tracking changes
        
    Returns:
        Sorted list of WordSegment objects by start time
    """
    if not words:
        return []
    
    if audit_log:
        audit_log.log("normalize", f"Starting normalization of {len(words)} words")
    
    # Check for anomalies before sorting
    anomalies = []
    for i in range(len(words) - 1):
        curr = words[i]
        next_w = words[i + 1]
        
        # Check for backwards time jump
        if next_w.start < curr.start:
            anomaly = {
                "index": i,
                "word1": {"text": curr.text, "start": curr.start, "end": curr.end, "speaker": curr.speaker},
                "word2": {"text": next_w.text, "start": next_w.start, "end": next_w.end, "speaker": next_w.speaker},
                "time_jump": curr.start - next_w.start
            }
            anomalies.append(anomaly)
            
            if float(anomaly["time_jump"]) if isinstance(anomaly["time_jump"], (int, float)) else 0.0 > 0.5:
                get_logger().warning(
                    f"Significant backwards time jump at index {i}: "
                    f"'{curr.text}' ({curr.start:.2f}s) -> '{next_w.text}' ({next_w.start:.2f}s), "
                    f"jump: {anomaly['time_jump']:.2f}s"
                )
    
    if audit_log and anomalies:
        audit_log.log(
            "normalize_anomalies",
            f"Found {len(anomalies)} timestamp anomalies",
            {"anomalies": anomalies}
        )
    
    # Sort by (start, end) to establish consistent ordering
    sorted_words = sorted(words, key=lambda w: (w.start, w.end))
    
    if audit_log:
        audit_log.log(
            "normalize_complete",
            f"Normalization complete: {len(sorted_words)} words sorted",
            {"anomaly_count": len(anomalies)}
        )
    
    log_debug(f"Normalized {len(sorted_words)} words, found {len(anomalies)} timestamp anomalies")
    
    return sorted_words


# =============================================================================
# Speaker Grouping
# =============================================================================

def group_words_by_speaker(
    words: List[WordSegment],
    audit_log: Optional[TurnBuildingAuditLog] = None
) -> List[RawSegment]:
    """
    Group consecutive words by speaker into RawSegments.
    
    Args:
        words: Sorted list of WordSegment objects
        audit_log: Optional audit logger
        
    Returns:
        List of RawSegment objects, each containing consecutive words from one speaker
    """
    if not words:
        return []
    
    if audit_log:
        audit_log.log("grouping", f"Starting speaker grouping for {len(words)} words")
    
    segments: List[RawSegment] = []
    current_words: List[WordSegment] = []
    current_speaker: Optional[str] = None
    
    for word in words:
        speaker = word.speaker or "Unknown"
        
        if speaker != current_speaker:
            # Speaker change - save current segment and start new one
            if current_words:
                segment = _create_raw_segment(current_words, current_speaker or "Unknown")
                segments.append(segment)
            
            current_words = [word]
            current_speaker = speaker
        else:
            # Same speaker - add to current segment
            current_words.append(word)
    
    # Don't forget the last segment
    if current_words:
        segment = _create_raw_segment(current_words, current_speaker or "Unknown")
        segments.append(segment)
    
    # Calculate gaps between segments
    _calculate_segment_gaps(segments)
    
    if audit_log:
        audit_log.log(
            "grouping_complete",
            f"Created {len(segments)} raw segments from {len(words)} words",
            {
                "segment_count": len(segments),
                "speakers": list(set(s.speaker for s in segments)),
                "avg_segment_words": sum(s.word_count for s in segments) / len(segments) if segments else 0
            }
        )
    
    log_debug(f"Grouped {len(words)} words into {len(segments)} raw segments")
    
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


# =============================================================================
# Interjection Pattern Matching
# =============================================================================

def classify_interjection_type(
    text: str,
    config: TurnBuilderConfig
) -> Tuple[str, float]:
    """
    Classify interjection type based on text patterns.
    
    Args:
        text: The text to classify
        config: Configuration with interjection patterns
        
    Returns:
        Tuple of (interjection_type, confidence_boost)
    """
    text_lower = text.lower().strip()
    
    # Check each category
    for category, patterns in config.interjection_patterns.items():
        for pattern in patterns:
            # Check for exact match or pattern at start/end
            if text_lower == pattern:
                return (category, 0.3)  # High confidence for exact match
            elif text_lower.startswith(pattern + " ") or text_lower.endswith(" " + pattern):
                return (category, 0.2)  # Medium confidence for partial match
            elif pattern in text_lower:
                return (category, 0.1)  # Lower confidence for contains
    
    return ("unclear", 0.0)


def is_potential_interjection(
    segment: RawSegment,
    config: TurnBuilderConfig,
    audit_log: Optional[TurnBuildingAuditLog] = None
) -> Tuple[bool, float, str, str]:
    """
    Determine if a segment is likely an interjection.
    
    Args:
        segment: The RawSegment to classify
        config: Configuration with thresholds
        audit_log: Optional audit logger
        
    Returns:
        Tuple of (is_interjection, confidence, interjection_type, reason)
    """
    reasons = []
    
    # Check duration threshold
    duration_ok = segment.duration <= config.max_interjection_duration
    if duration_ok:
        reasons.append(f"duration {segment.duration:.2f}s <= {config.max_interjection_duration}s")
    else:
        reasons.append(f"duration {segment.duration:.2f}s > {config.max_interjection_duration}s threshold")
    
    # Check word count threshold
    word_count_ok = segment.word_count <= config.max_interjection_words
    if word_count_ok:
        reasons.append(f"word_count {segment.word_count} <= {config.max_interjection_words}")
    else:
        reasons.append(f"word_count {segment.word_count} > {config.max_interjection_words} threshold")
    
    # Must pass BOTH thresholds to be considered an interjection
    if not (duration_ok and word_count_ok):
        reason = f"Failed thresholds: {'; '.join(reasons)}"
        return (False, 0.0, "none", reason)
    
    # Calculate base confidence from how far under thresholds
    duration_confidence = 1.0 - (segment.duration / config.max_interjection_duration)
    word_count_confidence = 1.0 - (segment.word_count / config.max_interjection_words)
    base_confidence = (duration_confidence + word_count_confidence) / 2
    
    # Check for pattern match
    interjection_type, pattern_boost = classify_interjection_type(segment.text, config)
    
    if pattern_boost > 0:
        reasons.append(f"pattern_match: {interjection_type} (+{pattern_boost:.1f})")
    
    # Final confidence
    confidence = min(1.0, base_confidence + pattern_boost)
    
    # Determine if it's an interjection
    # High confidence with pattern match -> definitely interjection
    # Medium confidence without pattern -> likely interjection
    # Low confidence without pattern -> uncertain, but still classify as interjection if meets thresholds
    is_interjection = True
    
    reason = f"Classified as interjection: {'; '.join(reasons)}"
    
    return (is_interjection, confidence, interjection_type, reason)


# =============================================================================
# Interrupt Level Calculation
# =============================================================================

def calculate_interrupt_level(
    interjection_segment: RawSegment,
    prev_segment: Optional[RawSegment],
    next_segment: Optional[RawSegment]
) -> str:
    """
    Calculate how disruptive an interjection is based on timing.
    
    Args:
        interjection_segment: The interjection segment
        prev_segment: Previous segment (may be None)
        next_segment: Next segment (may be None)
        
    Returns:
        Interrupt level: "none", "low", "medium", "high"
    """
    gap_before = interjection_segment.gap_before
    gap_after = interjection_segment.gap_after
    
    # Check for overlaps (negative gaps indicate overlap)
    has_overlap_before = gap_before is not None and gap_before < 0
    has_overlap_after = gap_after is not None and gap_after < 0
    
    if has_overlap_before and has_overlap_after:
        # Overlaps both sides - high interruption
        return "high"
    elif has_overlap_before or has_overlap_after:
        # Overlaps one side
        overlap_amount: float = 0.0
        if has_overlap_before:
            overlap_amount = abs(float(gap_before)) if gap_before is not None else 0.0
        if has_overlap_after:
            overlap_amount = float(max(overlap_amount, abs(float(gap_after)))) if gap_after is not None else overlap_amount
        
        if overlap_amount > 0.5:
            return "high"
        elif overlap_amount > 0.2:
            return "medium"
        else:
            return "low"
    else:
        # No overlaps - check gap sizes
        # Small gaps suggest interjection during brief pause
        min_gap = float('inf')
        if gap_before is not None:
            min_gap = min(min_gap, gap_before)
        if gap_after is not None:
            min_gap = min(min_gap, gap_after)
        
        if min_gap < 0.3:
            return "low"
        else:
            return "none"


# =============================================================================
# Segment Classification
# =============================================================================

def classify_segments(
    segments: List[RawSegment],
    config: TurnBuilderConfig,
    audit_log: Optional[TurnBuildingAuditLog] = None
) -> Tuple[List[RawSegment], List[RawSegment]]:
    """
    Classify segments as primary turns or interjections.
    
    Args:
        segments: List of RawSegment objects
        config: Configuration with thresholds
        audit_log: Optional audit logger
        
    Returns:
        Tuple of (primary_segments, interjection_segments)
    """
    if audit_log:
        audit_log.log("classification_start", f"Classifying {len(segments)} segments")
    
    for i, segment in enumerate(segments):
        # Get surrounding segments for context
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i < len(segments) - 1 else None
        
        # Classify
        is_interjection, confidence, interjection_type, reason = is_potential_interjection(
            segment, config, audit_log
        )
        
        # Update segment with classification
        segment.is_interjection = is_interjection
        segment.interjection_confidence = confidence
        segment.interjection_type = interjection_type
        segment.classification_method = "rule"
        
        # Calculate interrupt level if it's an interjection
        if is_interjection:
            segment.interrupt_level = calculate_interrupt_level(segment, prev_seg, next_seg)
        
        # Check for potential diarization errors
        # Single word that doesn't match any pattern and appears between same-speaker segments
        if (segment.word_count == 1 and 
            interjection_type == "unclear" and
            prev_seg and next_seg and
            prev_seg.speaker == next_seg.speaker and
            prev_seg.speaker != segment.speaker):
            segment.likely_diarization_error = True
            reason += " [LIKELY DIARIZATION ERROR]"
        
        if audit_log:
            audit_log.log_segment_classification(
                segment, is_interjection, confidence, interjection_type, reason
            )
    
    # Split into primary and interjection lists
    primary_segments = [s for s in segments if not s.is_interjection]
    interjection_segments = [s for s in segments if s.is_interjection]
    
    if audit_log:
        audit_log.log(
            "classification_complete",
            f"Classification complete: {len(primary_segments)} primary, {len(interjection_segments)} interjections",
            {
                "primary_count": len(primary_segments),
                "interjection_count": len(interjection_segments),
                "diarization_error_count": sum(1 for s in segments if s.likely_diarization_error)
            }
        )
    
    log_progress(
        f"Classified {len(segments)} segments: "
        f"{len(primary_segments)} primary, {len(interjection_segments)} interjections"
    )
    
    return primary_segments, interjection_segments


# =============================================================================
# Helper for Creating InterjectionSegment from RawSegment
# =============================================================================

def raw_segment_to_interjection(segment: RawSegment) -> InterjectionSegment:
    """Convert a classified RawSegment to an InterjectionSegment."""
    return InterjectionSegment(
        speaker=segment.speaker,
        start=segment.start,
        end=segment.end,
        text=segment.text,
        words=segment.words,
        confidence=segment.interjection_confidence,
        interjection_type=segment.interjection_type,
        interrupt_level=segment.interrupt_level,
        classification_method=segment.classification_method,
        likely_diarization_error=segment.likely_diarization_error
    )
