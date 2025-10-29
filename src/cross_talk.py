# src/cross_talk.py
from __future__ import annotations
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

BASIC_CROSS_TALK_CONFIG = {
    "overlap_threshold": 0.1,        # Minimum overlap duration (seconds)
    "confidence_threshold": 0.6,     # Minimum confidence for assignment
    "min_word_duration": 0.05,       # Minimum word duration to process
    "mark_cross_talk": True,         # Whether to mark cross-talk in outputs
    "basic_confidence": True         # Whether to calculate basic confidence
}


def detect_basic_cross_talk(diar_segments: List[Dict], overlap_threshold: float = 0.1) -> List[Dict]:
    """
    Identify temporal overlaps between different speaker segments.
    
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
    logger.debug(f"Detecting cross-talk with {len(diar_segments)} segments, threshold: {overlap_threshold}")
    cross_talk_segments = []
    
    # Sort segments by start time for easier overlap detection
    sorted_segments = sorted(diar_segments, key=lambda x: x["start"])
    
    # Check each pair of segments for overlaps
    overlap_count = 0
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
                logger.debug(f"Found overlap {overlap_count}: {seg1['label']} ({seg1['start']:.2f}-{seg1['end']:.2f}) "
                           f"with {seg2['label']} ({seg2['start']:.2f}-{seg2['end']:.2f}) "
                           f"duration: {overlap_duration:.3f}")
                cross_talk_segments.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "speakers": [seg1["label"], seg2["label"]],
                    "duration": overlap_duration
                })
    
    # Sort cross-talk segments by start time
    cross_talk_segments.sort(key=lambda x: x["start"])
    
    logger.info(f"Detected {len(cross_talk_segments)} cross-talk segments from {overlap_count} overlaps")
    return cross_talk_segments


def assign_words_with_basic_cross_talk(
    words: List[Dict],
    diar_segments: List[Dict],
    cross_talk_segments: List[Dict]
) -> List[Dict]:
    """
    Enhance word assignment by correcting speaker assignments during cross-talk and adding confidence scores.
    
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
    logger.debug(f"Processing {len(words)} words with {len(cross_talk_segments)} cross-talk segments")
    enhanced_words = []
    cross_talk_word_count = 0
    corrected_speaker_count = 0
    
    # Helper function to calculate overlap
    def _calculate_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
        return max(0.0, min(end1, end2) - max(start1, start2))
    
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
                        if total_overlap < 0.05:  # Very low confidence in cross-talk correction
                            seg_info = [(s['label'], f"{s['start']:.1f}-{s['end']:.1f}", f"{ov:.3f}") for s, ov in segment_overlaps]
                            logger.debug(f"LOW CONFIDENCE CROSS-TALK: '{word_text}' from {current_speaker} to {speaker} - Speaker {speaker}: total_overlap={total_overlap:.3f}, segments: {seg_info}")
                    
                    # Find the speaker with maximum overlap
                    if speaker_overlaps:
                        best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                        best_overlap = speaker_overlaps[best_speaker]
                        
                        # Log the overlap analysis for low confidence words
                        if best_overlap < 0.05:  # Very low confidence in cross-talk correction
                            logger.debug(f"LOW CONFIDENCE CROSS-TALK: '{word_text}' from {current_speaker} to {best_speaker} - Overlap analysis: {speaker_overlaps}, "
                                       f"best: {best_speaker} ({best_overlap:.3f}), current: {current_speaker}")
                        
                        # Only correct if there's a meaningful difference and minimum overlap
                        min_overlap_threshold = 0.01  # Minimum 10ms overlap to consider
                        if (best_speaker != current_speaker and
                            best_overlap > min_overlap_threshold and
                            best_overlap > speaker_overlaps.get(current_speaker, 0)):
                            enhanced_word["speaker"] = best_speaker
                            corrected_speaker_count += 1
                            
                            # Log detailed information about corrections
                            logger.debug(f"CORRECTED SPEAKER: '{word_text}' from {current_speaker} to {best_speaker} "
                                       f"at {word_start:.3f}-{word_end:.3f} "
                                       f"overlap: {best_overlap:.3f} "
                                       f"cross-talk speakers: {ct_speakers}")
                        elif best_overlap < 0.05:  # Very low confidence in cross-talk correction
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
                    logger.debug(f"LOW CONFIDENCE CROSS-TALK: '{word_text}' from {current_speaker} to {enhanced_word['speaker']} - CROSS-TALK WORD: '{word_text}' (speaker: {enhanced_word['speaker']}) "
                               f"at {word_start:.3f}-{word_end:.3f} "
                               f"overlap: {overlap_duration:.3f} ({overlap_percentage:.1%}) "
                               f"confidence: {enhanced_word['confidence']:.3f} "
                               f"cross-talk speakers: {ct_segment['speakers']}")
                
                # Once marked as cross-talk, no need to check other segments
                break
        
        enhanced_words.append(enhanced_word)
    
    logger.info(f"Processed {len(words)} words, marked {cross_talk_word_count} as cross-talk, "
               f"corrected {corrected_speaker_count} speaker assignments")
    return enhanced_words