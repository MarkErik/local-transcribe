# src/cross_talk.py
from __future__ import annotations
from typing import List, Dict
import logging

from config import is_debug_enabled, is_info_enabled

logger = logging.getLogger(__name__)

BASIC_CROSS_TALK_CONFIG = {
    "overlap_threshold": 0.1,        # Minimum overlap duration (seconds)
    "confidence_threshold": 0.6,     # Minimum confidence for assignment
    "min_word_duration": 0.05,       # Minimum word duration to process
    "mark_cross_talk": True,         # Whether to mark cross-talk in outputs
    "basic_confidence": True,        # Whether to calculate basic confidence
    "boundary_buffer_zone": 0.3,     # Buffer zone around speaker boundaries (seconds)
    "boundary_smoothing": True,      # Whether to apply boundary smoothing
    "min_boundary_overlap": 0.02     # Minimum overlap for boundary corrections (seconds)
}


def detect_basic_cross_talk(diar_segments: List[Dict], overlap_threshold: float = 0.1) -> List[Dict]:
    """
    Identify temporal overlaps between different speaker segments.
    
    Enhanced to detect boundary transitions and short overlaps that might indicate
    speaker changes even if they don't meet the standard overlap threshold.
    
    Parameters
    ----------
    diar_segments : List[Dict]
        List of diarization segments with 'start', 'end', and 'label' keys
    overlap_threshold : float, optional
        Minimum overlap duration in seconds to consider as cross-talk, by default 0.1
        
    Returns
    -------
    List[Dict]
        List of cross-talk segments with 'start', 'end', 'speakers', and 'duration' keys
    """
    if is_debug_enabled():
        logger.debug(f"Detecting cross-talk with {len(diar_segments)} segments, threshold: {overlap_threshold}")
    cross_talk_segments = []
    
    # Sort segments by start time for easier overlap detection
    sorted_segments = sorted(diar_segments, key=lambda x: x["start"])
    
    # Check each pair of segments for overlaps
    overlap_count = 0
    boundary_transition_count = 0
    
    # First, detect standard cross-talk overlaps
    for i in range(len(sorted_segments)):
        for j in range(i + 1, len(sorted_segments)):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[j]
            
            # Skip if same speaker
            if seg1["label"] == seg2["label"]:
                continue
                
            # Calculate overlap
            overlap_start = max(seg1["start"], seg2["start"])
            overlap_end = min(seg1["end"], seg2["end"])
            overlap_duration = overlap_end - overlap_start
            
            # Only consider if there's actual overlap and it meets threshold
            if overlap_duration > 0 and overlap_duration >= overlap_threshold:
                overlap_count += 1
                if is_debug_enabled():
                    logger.debug(f"Found overlap {overlap_count}: {seg1['label']} ({seg1['start']:.2f}-{seg1['end']:.2f}) "
                               f"with {seg2['label']} ({seg2['start']:.2f}-{seg2['end']:.2f}) "
                               f"duration: {overlap_duration:.3f}")
                cross_talk_segments.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "speakers": [seg1["label"], seg2["label"]],
                    "duration": overlap_duration,
                    "type": "standard"
                })
    
    # Second, detect boundary transitions (segments that are close but don't overlap)
    boundary_buffer = BASIC_CROSS_TALK_CONFIG["boundary_buffer_zone"]
    
    for i in range(len(sorted_segments) - 1):
        seg1 = sorted_segments[i]
        seg2 = sorted_segments[i + 1]
        
        # Skip if same speaker
        if seg1["label"] == seg2["label"]:
            continue
            
        # Check if segments are close to each other (boundary transition)
        gap = seg2["start"] - seg1["end"]
        
        if 0 < gap <= boundary_buffer:  # Segments are close but don't overlap
            boundary_transition_count += 1
            
            # Create a boundary transition zone
            transition_start = seg1["end"] - (boundary_buffer / 2)
            transition_end = seg2["start"] + (boundary_buffer / 2)
            
            if is_debug_enabled():
                logger.debug(f"Found boundary transition {boundary_transition_count}: {seg1['label']} -> {seg2['label']} "
                           f"gap: {gap:.3f}, buffer_zone: {transition_start:.2f}-{transition_end:.2f}")
            
            cross_talk_segments.append({
                "start": transition_start,
                "end": transition_end,
                "speakers": [seg1["label"], seg2["label"]],
                "duration": transition_end - transition_start,
                "type": "boundary_transition",
                "gap": gap
            })
    
    # Sort cross-talk segments by start time
    cross_talk_segments.sort(key=lambda x: x["start"])
    
    if is_info_enabled():
        logger.info(f"Detected {len(cross_talk_segments)} cross-talk segments: "
                   f"{overlap_count} standard overlaps, {boundary_transition_count} boundary transitions")
    return cross_talk_segments


def assign_words_with_basic_cross_talk(
    words: List[Dict],
    diar_segments: List[Dict],
    cross_talk_segments: List[Dict]
) -> List[Dict]:
    """
    Enhance word assignment by correcting speaker assignments during cross-talk and adding confidence scores.
    
    Enhanced with boundary-aware smoothing algorithms to reduce word leakage at speaker transitions.
    
    Parameters
    ----------
    words : List[Dict]
        List of word dictionaries with 'text', 'start', 'end', and 'speaker' keys
    diar_segments : List[Dict]
        List of diarization segments with 'start', 'end', and 'label' keys
    cross_talk_segments : List[Dict]
        List of cross-talk segments with 'start', 'end', and 'speakers' keys
        
    Returns
    -------
    List[Dict]
        Enhanced list of word dictionaries with corrected speakers, 'cross_talk' and 'confidence' keys
    """
    if is_debug_enabled():
        logger.debug(f"Processing {len(words)} words with {len(cross_talk_segments)} cross-talk segments")
    enhanced_words = []
    cross_talk_word_count = 0
    corrected_speaker_count = 0
    boundary_smoothed_count = 0
    
    # Helper function to calculate overlap
    def _calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
        return max(0.0, min(end1, end2) - max(start1, start2))
    
    # Helper function to apply boundary smoothing
    def _apply_boundary_smoothing(word: Dict, context_words: List[Dict]) -> str:
        """
        Apply smoothing algorithm for boundary words based on context.
        
        Parameters
        ----------
        word : Dict
            The current word to smooth
        context_words : List[Dict]
            Previous and next words for context
            
        Returns
        -------
        str
            The smoothed speaker assignment
        """
        if not context_words:
            return word.get("speaker", "UNKNOWN")
        
        # Count speaker assignments in context
        speaker_counts = {}
        total_weight = 0.0
        
        for ctx_word in context_words:
            ctx_speaker = ctx_word.get("speaker", "UNKNOWN")
            ctx_confidence = ctx_word.get("confidence", 1.0)
            
            # Weight by confidence and recency (closer words have higher weight)
            weight = ctx_confidence
            speaker_counts[ctx_speaker] = speaker_counts.get(ctx_speaker, 0.0) + weight
            total_weight += weight
        
        if not speaker_counts:
            return word.get("speaker", "UNKNOWN")
        
        # Find the most common speaker in context
        best_speaker = max(speaker_counts, key=speaker_counts.get)
        confidence_score = speaker_counts[best_speaker] / total_weight if total_weight > 0 else 0.0
        
        # Only apply smoothing if confidence is high enough
        if confidence_score > 0.6:  # 60% confidence threshold for smoothing
            return best_speaker
        
        return word.get("speaker", "UNKNOWN")
    
    # Process words with boundary awareness
    for i, word in enumerate(words):
        # Create a copy of the word to avoid modifying the original
        enhanced_word = dict(word)
        
        # Initialize cross-talk flag and confidence
        enhanced_word["cross_talk"] = False
        enhanced_word["confidence"] = 1.0
        
        # Check if word overlaps with any cross-talk segment
        word_start = word["start"]
        word_end = word["end"]
        word_duration = word_end - word_start
        word_text = word.get("text", "")
        current_speaker = word.get("speaker", "UNKNOWN")
        
        # Skip words that are too short
        if word_duration < BASIC_CROSS_TALK_CONFIG["min_word_duration"]:
            enhanced_words.append(enhanced_word)
            continue
            
        # Check each cross-talk segment for overlap
        for ct_segment in cross_talk_segments:
            # Calculate overlap between word and cross-talk segment
            overlap_start = max(word_start, ct_segment["start"])
            overlap_end = min(word_end, ct_segment["end"])
            overlap_duration = overlap_end - overlap_start
            
            # If there's overlap, this is cross-talk
            if overlap_duration > 0:
                enhanced_word["cross_talk"] = True
                cross_talk_word_count += 1
                
                # During cross-talk, re-evaluate speaker assignment
                ct_speakers = ct_segment["speakers"]
                if len(ct_speakers) >= 2:
                    # Calculate overlap with each speaker involved in cross-talk
                    speaker_overlaps = {}
                    
                    for speaker in ct_speakers:
                        # Find all segments for this speaker
                        speaker_segments = [seg for seg in diar_segments if seg["label"] == speaker]
                        
                        # Calculate total overlap with this speaker's segments
                        total_overlap = 0.0
                        segment_overlaps = []
                        for seg in speaker_segments:
                            ov = _calculate_overlap(word_start, word_end, seg["start"], seg["end"])
                            if ov > 0:
                                segment_overlaps.append((seg, ov))
                                total_overlap += ov
                        
                        speaker_overlaps[speaker] = total_overlap
                        
                        # Log detailed overlap analysis for low confidence words
                        if total_overlap < BASIC_CROSS_TALK_CONFIG["min_boundary_overlap"]:
                            seg_info = [(s['label'], f"{s['start']:.1f}-{s['end']:.1f}", f"{ov:.3f}") for s, ov in segment_overlaps]
                            if is_debug_enabled():
                                logger.debug(f"LOW CONFIDENCE CROSS-TALK: '{word_text}' from {current_speaker} to {speaker} - Speaker {speaker}: total_overlap={total_overlap:.3f}, segments: {seg_info}")
                    
                    # Find the speaker with maximum overlap
                    if speaker_overlaps:
                        best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                        best_overlap = speaker_overlaps[best_speaker]
                        
                        # Apply boundary smoothing for low confidence assignments
                        if (BASIC_CROSS_TALK_CONFIG["boundary_smoothing"] and
                            best_overlap < BASIC_CROSS_TALK_CONFIG["min_boundary_overlap"] * 2):
                            
                            # Get context words for smoothing
                            context_words = []
                            if i > 0:
                                context_words.append(words[i - 1])  # Previous word
                            if i < len(words) - 1:
                                context_words.append(words[i + 1])  # Next word
                            
                            # Apply boundary smoothing
                            smoothed_speaker = _apply_boundary_smoothing(enhanced_word, context_words)
                            
                            if smoothed_speaker != current_speaker:
                                enhanced_word["speaker"] = smoothed_speaker
                                corrected_speaker_count += 1
                                boundary_smoothed_count += 1
                                
                                if is_debug_enabled():
                                   logger.debug(f"BOUNDARY SMOOTHED: '{word_text}' from {current_speaker} to {smoothed_speaker} "
                                              f"at {word_start:.3f}-{word_end:.3f} "
                                              f"overlap: {best_overlap:.3f} -> smoothed")
                                continue
                        
                        # Standard correction for higher confidence assignments
                        min_overlap_threshold = BASIC_CROSS_TALK_CONFIG["min_boundary_overlap"]
                        if (best_speaker != current_speaker and
                            best_overlap > min_overlap_threshold and
                            best_overlap > speaker_overlaps.get(current_speaker, 0)):
                            enhanced_word["speaker"] = best_speaker
                            corrected_speaker_count += 1
                            
                            # Log detailed information about corrections
                            if is_debug_enabled():
                                logger.debug(f"CORRECTED SPEAKER: '{word_text}' from {current_speaker} to {best_speaker} "
                                           f"at {word_start:.3f}-{word_end:.3f} "
                                           f"overlap: {best_overlap:.3f} "
                                           f"cross-talk speakers: {ct_speakers}")
                        elif best_overlap < BASIC_CROSS_TALK_CONFIG["min_boundary_overlap"]:
                            if is_debug_enabled():
                                logger.debug(f"LOW CONFIDENCE CROSS-TALK: '{word_text}' from {current_speaker} to {best_speaker} - NO CORRECTION: best={best_speaker} ({best_overlap:.3f}), "
                                           f"current={current_speaker} ({speaker_overlaps.get(current_speaker, 0):.3f})")
                
                # Calculate confidence based on overlap percentage
                if BASIC_CROSS_TALK_CONFIG["basic_confidence"]:
                    overlap_percentage = overlap_duration / word_duration
                    # Invert the percentage for confidence (more overlap = lower confidence)
                    enhanced_word["confidence"] = 1.0 - overlap_percentage
                    
                    # Ensure confidence is within [0, 1] range
                    enhanced_word["confidence"] = max(0.0, min(1.0, enhanced_word["confidence"]))
                
                # Log detailed information about low confidence cross-talk words
                if enhanced_word["confidence"] < 0.05:  # Very low confidence in cross-talk correction
                   if is_debug_enabled():
                       logger.debug(f"LOW CONFIDENCE CROSS-TALK: '{word_text}' from {current_speaker} to {enhanced_word['speaker']} - CROSS-TALK WORD: '{word_text}' (speaker: {enhanced_word['speaker']}) "
                                  f"at {word_start:.3f}-{word_end:.3f} "
                                  f"overlap: {overlap_duration:.3f} ({overlap_percentage:.1%}) "
                                  f"confidence: {enhanced_word['confidence']:.3f} "
                                  f"cross-talk speakers: {ct_segment['speakers']}")
                
                # Once marked as cross-talk, no need to check other segments
                break
        
        enhanced_words.append(enhanced_word)
    
    if is_info_enabled():
        logger.info(f"Processed {len(words)} words, marked {cross_talk_word_count} as cross-talk, "
                   f"corrected {corrected_speaker_count} speaker assignments "
                   f"({boundary_smoothed_count} via boundary smoothing)")
    return enhanced_words