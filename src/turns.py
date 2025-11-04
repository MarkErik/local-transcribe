from __future__ import annotations
import logging
from typing import List, Dict

from config import is_debug_enabled, is_info_enabled

logger = logging.getLogger(__name__)

def build_turns(words: List[Dict], speaker_label: str, max_gap_s: float = 0.8, max_chars: int = 120):
    """
    Group word-level tokens into readable turns.
    Returns a list of enhanced turns:
      [{'speaker':'Interviewer','start':..,'end':..,'text':'..','cross_talk_present':..,'confidence':..,'words':..}]
    
    Parameters
    ----------
    words : List[Dict]
        List of word dictionaries with 'text', 'start', 'end' keys, and optionally
        'cross_talk' and 'confidence' keys for enhanced word objects
    speaker_label : str
        Speaker label to assign to all turns
    max_gap_s : float, optional
        Maximum gap in seconds between words to consider them part of the same turn, by default 0.8
    max_chars : int, optional
        Maximum number of characters in a turn before splitting, by default 120
        
    Returns
    -------
    List[Dict]
        List of turn dictionaries with enhanced information including cross-talk flags and confidence
    """
    turns = []
    buf = []  # Buffer for word texts
    word_buf = []  # Buffer for full word objects
    cur_start = None
    last_end = None

    for w in words:
        if w.get("text") is None:
            continue
            
        s, e, t = w["start"], w["end"], w["text"]
        
        if cur_start is None:
            cur_start = s
            buf = [t]
            word_buf = [w]
        else:
            gap = s - (last_end if last_end is not None else s)
            if gap > max_gap_s or sum(len(x)+1 for x in buf) + len(t) > max_chars:
                # Flush current turn
                turn = _create_enhanced_turn(speaker_label, cur_start, last_end or s, buf, word_buf)
                turns.append(turn)
                
                # Start new turn
                cur_start = s
                buf = [t]
                word_buf = [w]
            else:
                buf.append(t)
                word_buf.append(w)
        last_end = e

    # Flush remaining buffer
    if buf:
        turn = _create_enhanced_turn(
            speaker_label,
            cur_start if cur_start is not None else 0.0,
            last_end if last_end is not None else cur_start,
            buf,
            word_buf
        )
        turns.append(turn)
    
    if is_debug_enabled():
        logger.debug(f"Built {len(turns)} turns for speaker {speaker_label}")
    return turns


def _create_enhanced_turn(speaker_label: str, start: float, end: float, text_buf: List[str], word_buf: List[Dict]) -> Dict:
    """
    Create an enhanced turn dictionary with cross-talk information and confidence.
    
    Parameters
    ----------
    speaker_label : str
        Speaker label for the turn
    start : float
        Start time of the turn
    end : float
        End time of the turn
    text_buf : List[str]
        List of word texts in the turn
    word_buf : List[Dict]
        List of full word objects in the turn
        
    Returns
    -------
    Dict
        Enhanced turn dictionary with cross-talk and confidence information
    """
    # Check if any words in the turn have cross-talk
    cross_talk_present = False
    confidence_sum = 0.0
    confidence_count = 0
    
    for word in word_buf:
        # Check for cross-talk flag (handle backward compatibility)
        if word.get("cross_talk", False):
            cross_talk_present = True
            
        # Sum confidence scores if available (handle backward compatibility)
        if "confidence" in word:
            confidence_sum += word["confidence"]
            confidence_count += 1
    
    # Calculate average confidence or default to 1.0
    avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 1.0
    
    # Ensure confidence is within valid range
    avg_confidence = max(0.0, min(1.0, avg_confidence))
    
    turn = {
        "speaker": speaker_label,
        "start": start,
        "end": end,
        "text": " ".join(text_buf).strip(),
        "cross_talk_present": cross_talk_present,
        "confidence": avg_confidence,
        "words": word_buf  # Include full word objects for reference
    }
    
    return turn

