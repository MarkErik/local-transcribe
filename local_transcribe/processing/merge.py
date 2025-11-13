#!/usr/bin/env python3
from __future__ import annotations
from typing import List
from local_transcribe.framework.plugins import Turn

def merge_turn(a: List[Turn], b: List[Turn]) -> List[Turn]:
    """
    Merge two turn lists (already labeled) by start time.
    Preserves overlaps (no forced stitching).
    """
    merged = sorted(a + b, key=lambda x: (x.start, x.end))
    # Optional: enforce tiny non-negative monotonicity
    eps = 1e-3
    last_end = 0.0
    for t in merged:
        if t.start < last_end - eps:
            t.start = max(0.0, last_end - eps)
        if t.end < t.start:
            t.end = t.start + eps
        last_end = max(last_end, t.end)
    return merged
