#!/usr/bin/env python3
"""
Combined Audio Turn Builder.

This module implements turn building for combined audio mode, where all speakers'
words and timestamps exist in a single diarized file.

The algorithm:
1. Normalizes timestamps (handles out-of-order words)
2. Groups consecutive words by speaker into raw segments
3. Classifies segments as primary turns or interjections
4. Builds hierarchical turns by merging same-speaker segments and attaching interjections
5. Calculates conversation metrics and speaker statistics
"""

from typing import List, Dict, Any, Optional, Union
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
from local_transcribe.processing.turn_building.turn_building_base import (
    TurnBuildingAuditLog,
    normalize_word_timestamps,
    group_words_by_speaker,
    classify_segments,
    raw_segment_to_interjection
)


def build_turns_combined_audio(
    words: Union[List[WordSegment], List[Dict[str, Any]]],
    intermediate_dir: Optional[Path] = None,
    config: Optional[TurnBuilderConfig] = None,
    **kwargs
) -> TranscriptFlow:
    """
    Build hierarchical conversation turns from combined audio word segments.
    
    This is the main entry point for combined audio mode turn building.
    
    Args:
        words: List of WordSegment objects or dicts with speaker assignments
        intermediate_dir: Optional path for saving audit logs
        config: Optional TurnBuilderConfig (uses defaults if not provided)
        **kwargs: Additional configuration options that override config:
            - max_interjection_duration: Max duration for interjections (default: 2.0s)
            - max_interjection_words: Max word count for interjections (default: 5)
            - max_gap_to_merge_turns: Max gap to merge same-speaker turns (default: 3.0s)
    
    Returns:
        TranscriptFlow with hierarchical turn structure
    """
    log_progress(f"Starting combined audio turn building with {len(words)} words")
    
    # Setup configuration
    if config is None:
        config = TurnBuilderConfig()
    
    # Override config with kwargs if provided
    if 'max_interjection_duration' in kwargs:
        config.max_interjection_duration = kwargs['max_interjection_duration']
    if 'max_interjection_words' in kwargs:
        config.max_interjection_words = kwargs['max_interjection_words']
    if 'max_gap_to_merge_turns' in kwargs:
        config.max_gap_to_merge_turns = kwargs['max_gap_to_merge_turns']
    
    # Setup audit logging
    audit_dir = None
    if intermediate_dir:
        audit_dir = Path(intermediate_dir) / "turn_building" / datetime.now().strftime("%Y%m%d_%H%M%S")
    
    audit_log = TurnBuildingAuditLog(output_dir=audit_dir, enabled=True)
    audit_log.log("init", "Turn building started", {
        "word_count": len(words),
        "config": config.to_dict()
    })
    
    # Convert dict inputs to WordSegment if needed
    word_segments = _ensure_word_segments(words)
    
    # Step 1: Normalize timestamps
    log_progress("Step 1: Normalizing timestamps")
    normalized_words = normalize_word_timestamps(word_segments, audit_log)
    
    # Step 2: Group by speaker
    log_progress("Step 2: Grouping words by speaker")
    raw_segments = group_words_by_speaker(normalized_words, audit_log)
    
    # Step 3: Classify segments
    log_progress("Step 3: Classifying segments as primary or interjection")
    primary_segments, interjection_segments = classify_segments(raw_segments, config, audit_log)
    
    # Step 4: Build hierarchical turns
    log_progress("Step 4: Building hierarchical turns")
    turns = _build_hierarchical_turns(
        raw_segments,  # Pass all segments in order
        config,
        audit_log
    )
    
    # Step 5: Calculate metrics
    log_progress("Step 5: Calculating conversation metrics")
    metadata = _build_metadata(normalized_words, config)
    conversation_metrics = _calculate_conversation_metrics(turns)
    speaker_statistics = _calculate_speaker_statistics(turns)
    
    # Create TranscriptFlow (raw, with flagged diarization errors)
    raw_transcript_flow = TranscriptFlow(
        turns=turns,
        metadata=metadata,
        conversation_metrics=conversation_metrics,
        speaker_statistics=speaker_statistics
    )
    
    audit_log.log("raw_complete", "Raw turn building complete (before diarization error merging)", {
        "total_turns": len(turns),
        "total_interjections": raw_transcript_flow.total_interjections,
        "duration": raw_transcript_flow.duration
    })
    
    # Save raw transcript flow to audit directory (includes flagged diarization errors)
    if audit_dir:
        raw_output_path = audit_dir / "transcript_flow_raw.json"
        with open(raw_output_path, 'w') as f:
            json.dump(raw_transcript_flow.to_dict(), f, indent=2)
        log_progress(f"Saved raw transcript flow to {raw_output_path}")
        audit_log.log("raw_saved", f"Raw transcript flow saved to {raw_output_path}")
    
    # Step 6: Refine by merging diarization errors with adjacent segments
    log_progress("Step 6: Merging likely diarization errors with adjacent segments")
    refined_turns = _merge_diarization_errors(turns, audit_log)
    
    # Recalculate metrics for refined transcript
    refined_metadata = metadata.copy()
    refined_metadata["refinement_applied"] = True
    refined_metadata["diarization_errors_merged"] = len(turns) - len(refined_turns) + _count_merged_interjections(turns, refined_turns)
    
    refined_conversation_metrics = _calculate_conversation_metrics(refined_turns)
    refined_speaker_statistics = _calculate_speaker_statistics(refined_turns)
    
    # Create refined TranscriptFlow
    refined_transcript_flow = TranscriptFlow(
        turns=refined_turns,
        metadata=refined_metadata,
        conversation_metrics=refined_conversation_metrics,
        speaker_statistics=refined_speaker_statistics
    )
    
    audit_log.log("refined_complete", "Refined turn building complete (after diarization error merging)", {
        "total_turns": len(refined_turns),
        "total_interjections": refined_transcript_flow.total_interjections,
        "duration": refined_transcript_flow.duration,
        "turns_before": len(turns),
        "turns_after": len(refined_turns)
    })
    
    # Save audit log
    if audit_dir:
        audit_log.save()
    
    log_progress(
        f"Turn building complete: {len(refined_turns)} turns, "
        f"{refined_transcript_flow.total_interjections} interjections, "
        f"{refined_transcript_flow.duration:.1f}s duration"
    )
    
    return refined_transcript_flow


def _ensure_word_segments(words: Union[List[WordSegment], List[Dict[str, Any]]]) -> List[WordSegment]:
    """Convert dict inputs to WordSegment objects if needed."""
    if not words:
        return []
    
    # Check if already WordSegment objects
    if hasattr(words[0], 'text') and hasattr(words[0], 'start'):
        return words
    
    # Convert dicts to WordSegment
    segments = []
    for w in words:
        segment = WordSegment(
            text=w.get('text', ''),
            start=w.get('start', 0.0),
            end=w.get('end', 0.0),
            speaker=w.get('speaker')
        )
        segments.append(segment)
    
    log_debug(f"Converted {len(segments)} dict inputs to WordSegment objects")
    return segments


def _build_hierarchical_turns(
    all_segments: List[RawSegment],
    config: TurnBuilderConfig,
    audit_log: TurnBuildingAuditLog
) -> List[HierarchicalTurn]:
    """
    Build hierarchical turns from classified segments.
    
    This algorithm:
    1. Iterates through segments in time order
    2. Accumulates primary segments, merging same-speaker segments if gap is small
    3. Collects interjections and attaches them to appropriate turns
    4. Creates HierarchicalTurn objects with embedded interjections
    """
    audit_log.log("build_turns_start", f"Building turns from {len(all_segments)} segments")
    
    if not all_segments:
        return []
    
    turns: List[HierarchicalTurn] = []
    turn_id_counter = 1
    
    # Current state
    current_primary_words: List[WordSegment] = []
    current_primary_speaker: Optional[str] = None
    current_primary_start: Optional[float] = None
    current_primary_end: Optional[float] = None
    pending_interjections: List[InterjectionSegment] = []
    merged_segment_count = 0
    
    def finalize_current_turn():
        """Create a turn from accumulated state."""
        nonlocal turn_id_counter, current_primary_words, current_primary_speaker
        nonlocal current_primary_start, current_primary_end, pending_interjections
        nonlocal merged_segment_count
        
        if not current_primary_words:
            return
        
        # Create the turn
        text = " ".join(w.text for w in current_primary_words)
        turn = HierarchicalTurn(
            turn_id=turn_id_counter,
            primary_speaker=current_primary_speaker,
            start=current_primary_start,
            end=current_primary_end,
            text=text,
            words=current_primary_words.copy(),
            interjections=[]
        )
        
        # Attach pending interjections that belong to this turn
        for ij in pending_interjections:
            # Interjection belongs to this turn if:
            # 1. It's from a different speaker, AND
            # 2. It falls within or near this turn's time range
            if ij.speaker != turn.primary_speaker:
                # Check if interjection is temporally associated with this turn
                # (during the turn or in a small gap after)
                if ij.start >= turn.start - 0.5 and ij.end <= turn.end + 1.0:
                    turn.interjections.append(ij)
                    audit_log.log_interjection_attached(
                        ij, turn.turn_id,
                        f"During turn ({turn.start:.2f}-{turn.end:.2f})"
                    )
        
        # Sort interjections by start time
        turn.interjections.sort(key=lambda x: x.start)
        
        # Recalculate metrics with interjections
        turn.recalculate_metrics()
        
        turns.append(turn)
        audit_log.log_turn_created(turn, merged_segment_count)
        
        turn_id_counter += 1
        
        # Reset state
        current_primary_words = []
        current_primary_speaker = None
        current_primary_start = None
        current_primary_end = None
        pending_interjections = []
        merged_segment_count = 0
    
    # Process segments in order
    for i, segment in enumerate(all_segments):
        if segment.is_interjection:
            # Store interjection for later attachment
            interjection = raw_segment_to_interjection(segment)
            pending_interjections.append(interjection)
            
            audit_log.log(
                "interjection_queued",
                f"Queued interjection: '{segment.text}'",
                {
                    "speaker": segment.speaker,
                    "start": segment.start,
                    "type": segment.interjection_type
                }
            )
        else:
            # Primary segment
            if current_primary_speaker is None:
                # First primary segment - start accumulating
                current_primary_words = segment.words.copy()
                current_primary_speaker = segment.speaker
                current_primary_start = segment.start
                current_primary_end = segment.end
                merged_segment_count = 1
                
                audit_log.log(
                    "primary_started",
                    f"Started accumulating primary: [{segment.speaker}] '{segment.text[:50]}...'",
                    {"speaker": segment.speaker, "start": segment.start}
                )
            
            elif segment.speaker == current_primary_speaker:
                # Same speaker - check if we should merge
                gap = segment.start - current_primary_end
                
                if gap <= config.max_gap_to_merge_turns:
                    # Merge with current turn
                    current_primary_words.extend(segment.words)
                    current_primary_end = segment.end
                    merged_segment_count += 1
                    
                    audit_log.log(
                        "primary_merged",
                        f"Merged segment (gap={gap:.2f}s): '{segment.text[:30]}...'",
                        {"gap": gap, "total_words": len(current_primary_words)}
                    )
                else:
                    # Gap too large - finalize current turn and start new one
                    audit_log.log(
                        "primary_gap_too_large",
                        f"Gap too large ({gap:.2f}s > {config.max_gap_to_merge_turns}s), finalizing turn",
                        {"gap": gap}
                    )
                    finalize_current_turn()
                    
                    # Start new turn
                    current_primary_words = segment.words.copy()
                    current_primary_speaker = segment.speaker
                    current_primary_start = segment.start
                    current_primary_end = segment.end
                    merged_segment_count = 1
            
            else:
                # Different speaker - finalize current turn
                audit_log.log(
                    "speaker_change",
                    f"Speaker change: {current_primary_speaker} -> {segment.speaker}",
                    {"from": current_primary_speaker, "to": segment.speaker}
                )
                finalize_current_turn()
                
                # Start new turn with new speaker
                current_primary_words = segment.words.copy()
                current_primary_speaker = segment.speaker
                current_primary_start = segment.start
                current_primary_end = segment.end
                merged_segment_count = 1
    
    # Don't forget to finalize the last turn
    finalize_current_turn()
    
    # Handle any remaining interjections that weren't attached
    # (these would be at the very end of the transcript)
    if pending_interjections and turns:
        last_turn = turns[-1]
        for ij in pending_interjections:
            if ij.speaker != last_turn.primary_speaker:
                last_turn.interjections.append(ij)
                audit_log.log_interjection_attached(
                    ij, last_turn.turn_id,
                    "Attached to final turn (orphaned interjection)"
                )
        last_turn.interjections.sort(key=lambda x: x.start)
        last_turn.recalculate_metrics()
    
    audit_log.log(
        "build_turns_complete",
        f"Created {len(turns)} hierarchical turns",
        {"turn_count": len(turns)}
    )
    
    return turns


def _build_metadata(
    words: List[WordSegment],
    config: TurnBuilderConfig
) -> Dict[str, Any]:
    """Build metadata dictionary for the TranscriptFlow."""
    speakers = list(set(w.speaker for w in words if w.speaker))
    
    return {
        "total_words": len(words),
        "speakers": sorted(speakers),
        "speaker_count": len(speakers),
        "start_time": words[0].start if words else 0.0,
        "end_time": words[-1].end if words else 0.0,
        "duration": (words[-1].end - words[0].start) if words else 0.0,
        "format_version": "2.0",
        "processing_config": config.to_dict(),
        "processed_at": datetime.now().isoformat()
    }


def _calculate_conversation_metrics(turns: List[HierarchicalTurn]) -> Dict[str, Any]:
    """Calculate conversation-level metrics."""
    if not turns:
        return {
            "total_turns": 0,
            "total_interjections": 0,
            "total_duration": 0.0,
            "avg_turn_duration": 0.0,
            "avg_turn_words": 0.0,
            "avg_speaking_rate": 0.0,
            "interjection_rate": 0.0,
            "avg_flow_continuity": 0.0
        }
    
    total_interjections = sum(len(t.interjections) for t in turns)
    total_duration = turns[-1].end - turns[0].start
    total_words = sum(t.word_count for t in turns)
    
    return {
        "total_turns": len(turns),
        "total_interjections": total_interjections,
        "total_duration": round(total_duration, 2),
        "total_words": total_words,
        "avg_turn_duration": round(sum(t.duration for t in turns) / len(turns), 2),
        "avg_turn_words": round(total_words / len(turns), 1),
        "avg_speaking_rate": round(sum(t.speaking_rate for t in turns) / len(turns), 1),
        "interjection_rate": round((total_interjections / total_duration) * 60, 2) if total_duration > 0 else 0,
        "avg_flow_continuity": round(sum(t.flow_continuity for t in turns) / len(turns), 3),
        "turn_type_distribution": _count_turn_types(turns)
    }


def _count_turn_types(turns: List[HierarchicalTurn]) -> Dict[str, int]:
    """Count turns by type."""
    distribution = {"monologue": 0, "acknowledged": 0, "interrupted": 0}
    for turn in turns:
        if turn.turn_type in distribution:
            distribution[turn.turn_type] += 1
    return distribution


def _calculate_speaker_statistics(turns: List[HierarchicalTurn]) -> Dict[str, Dict[str, Any]]:
    """Calculate per-speaker statistics."""
    stats: Dict[str, Dict[str, Any]] = {}
    
    # Initialize stats for all speakers
    all_speakers = set()
    for turn in turns:
        all_speakers.add(turn.primary_speaker)
        for ij in turn.interjections:
            all_speakers.add(ij.speaker)
    
    for speaker in all_speakers:
        stats[speaker] = {
            "total_turns": 0,
            "total_words": 0,
            "total_duration": 0.0,
            "total_interjections": 0,  # Times this speaker interjected
            "avg_turn_duration": 0.0,
            "avg_speaking_rate": 0.0,
            "avg_flow_continuity": 0.0
        }
    
    # Calculate primary turn stats
    for turn in turns:
        speaker = turn.primary_speaker
        stats[speaker]["total_turns"] += 1
        stats[speaker]["total_words"] += turn.word_count
        stats[speaker]["total_duration"] += turn.duration
        
        # Count interjections BY this speaker (in other people's turns)
        for ij in turn.interjections:
            stats[ij.speaker]["total_interjections"] += 1
    
    # Calculate averages
    for speaker, s in stats.items():
        if s["total_turns"] > 0:
            s["avg_turn_duration"] = round(s["total_duration"] / s["total_turns"], 2)
            s["total_duration"] = round(s["total_duration"], 2)
        
        # Calculate avg speaking rate from turns
        speaker_turns = [t for t in turns if t.primary_speaker == speaker]
        if speaker_turns:
            s["avg_speaking_rate"] = round(
                sum(t.speaking_rate for t in speaker_turns) / len(speaker_turns), 1
            )
            s["avg_flow_continuity"] = round(
                sum(t.flow_continuity for t in speaker_turns) / len(speaker_turns), 3
            )
    
    return stats


def _merge_diarization_errors(
    turns: List[HierarchicalTurn],
    audit_log: TurnBuildingAuditLog
) -> List[HierarchicalTurn]:
    """
    Merge likely diarization errors with adjacent segments.
    
    For interjections flagged as likely_diarization_error, we merge their
    words into the surrounding turn (either the current turn or an adjacent one).
    
    Strategy:
    - If the diarization error appears within a turn's time range, merge into that turn
    - The error's words are inserted at the appropriate position based on timing
    """
    audit_log.log("merge_start", f"Starting diarization error merging for {len(turns)} turns")
    
    # Count total diarization errors before merging
    total_errors = sum(
        1 for turn in turns 
        for ij in turn.interjections 
        if ij.likely_diarization_error
    )
    audit_log.log("merge_error_count", f"Found {total_errors} likely diarization errors to merge")
    
    if total_errors == 0:
        audit_log.log("merge_skip", "No diarization errors to merge")
        return turns
    
    refined_turns: List[HierarchicalTurn] = []
    
    for turn in turns:
        # Separate diarization errors from real interjections
        real_interjections = []
        error_interjections = []
        
        for ij in turn.interjections:
            if ij.likely_diarization_error:
                error_interjections.append(ij)
            else:
                real_interjections.append(ij)
        
        if not error_interjections:
            # No errors in this turn - keep as is but create new object
            new_turn = HierarchicalTurn(
                turn_id=turn.turn_id,
                primary_speaker=turn.primary_speaker,
                start=turn.start,
                end=turn.end,
                text=turn.text,
                words=turn.words.copy(),
                interjections=real_interjections
            )
            refined_turns.append(new_turn)
            continue
        
        # Merge error words into the turn
        merged_words = turn.words.copy()
        
        for error_ij in error_interjections:
            audit_log.log(
                "merge_error",
                f"Merging diarization error '{error_ij.text}' into turn {turn.turn_id}",
                {
                    "turn_id": turn.turn_id,
                    "turn_speaker": turn.primary_speaker,
                    "error_speaker": error_ij.speaker,
                    "error_text": error_ij.text,
                    "error_start": error_ij.start
                }
            )
            
            # Find insertion position based on timing
            insert_pos = 0
            for i, word in enumerate(merged_words):
                if word.start > error_ij.start:
                    insert_pos = i
                    break
                insert_pos = i + 1
            
            # Insert error words at the correct position
            # Update speaker to match the turn's speaker
            for j, error_word in enumerate(error_ij.words):
                new_word = WordSegment(
                    text=error_word.text,
                    start=error_word.start,
                    end=error_word.end,
                    speaker=turn.primary_speaker  # Reassign to turn's speaker
                )
                merged_words.insert(insert_pos + j, new_word)
        
        # Sort words by start time to ensure correct order
        merged_words.sort(key=lambda w: w.start)
        
        # Rebuild text from merged words
        merged_text = " ".join(w.text for w in merged_words)
        
        # Update start/end times
        new_start = min(turn.start, min(w.start for w in merged_words))
        new_end = max(turn.end, max(w.end for w in merged_words))
        
        # Create refined turn
        new_turn = HierarchicalTurn(
            turn_id=turn.turn_id,
            primary_speaker=turn.primary_speaker,
            start=new_start,
            end=new_end,
            text=merged_text,
            words=merged_words,
            interjections=real_interjections
        )
        
        audit_log.log(
            "merge_complete_turn",
            f"Turn {turn.turn_id} after merging: {len(error_interjections)} errors merged, "
            f"{turn.word_count} -> {new_turn.word_count} words",
            {
                "turn_id": turn.turn_id,
                "errors_merged": len(error_interjections),
                "words_before": turn.word_count,
                "words_after": new_turn.word_count,
                "interjections_remaining": len(real_interjections)
            }
        )
        
        refined_turns.append(new_turn)
    
    # Renumber turn IDs to be sequential
    for i, turn in enumerate(refined_turns):
        turn.turn_id = i + 1
    
    audit_log.log(
        "merge_complete",
        f"Diarization error merging complete: {len(turns)} -> {len(refined_turns)} turns, "
        f"{total_errors} errors merged",
        {
            "turns_before": len(turns),
            "turns_after": len(refined_turns),
            "errors_merged": total_errors
        }
    )
    
    log_progress(f"Merged {total_errors} diarization errors into adjacent turns")
    
    return refined_turns


def _count_merged_interjections(
    original_turns: List[HierarchicalTurn],
    refined_turns: List[HierarchicalTurn]
) -> int:
    """Count how many interjections were merged (removed as diarization errors)."""
    original_interjections = sum(len(t.interjections) for t in original_turns)
    refined_interjections = sum(len(t.interjections) for t in refined_turns)
    return original_interjections - refined_interjections
