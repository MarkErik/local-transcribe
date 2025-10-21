from __future__ import annotations
from typing import List, Dict

def build_turns(words: List[Dict], speaker_label: str, max_gap_s: float = 0.8, max_chars: int = 120):
    """
    Group word-level tokens into readable turns.
    Returns a list of turns:
      [{'speaker':'Interviewer','start':..,'end':..,'text':'..'}]
    """
    turns = []
    buf = []
    cur_start = None
    last_end = None

    for w in words:
        if w.get("text") is None: 
            continue
        s, e, t = w["start"], w["end"], w["text"]
        if cur_start is None:
            cur_start = s
            buf = [t]
        else:
            gap = s - (last_end if last_end is not None else s)
            if gap > max_gap_s or sum(len(x)+1 for x in buf) + len(t) > max_chars:
                # flush
                turns.append({
                    "speaker": speaker_label,
                    "start": cur_start,
                    "end": last_end if last_end is not None else s,
                    "text": " ".join(buf).strip()
                })
                # new
                cur_start = s
                buf = [t]
            else:
                buf.append(t)
        last_end = e

    if buf:
        turns.append({
            "speaker": speaker_label,
            "start": cur_start if cur_start is not None else 0.0,
            "end": last_end if last_end is not None else cur_start,
            "text": " ".join(buf).strip()
        })
    return turns

