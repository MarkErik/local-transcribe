"""
Diff analysis engine for comparing transcripts.

Uses sequence matching to identify:
- Exact matches
- Insertions (words in one transcript but not the other)
- Deletions
- Substitutions (different words at same position)
"""

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from collections import Counter


class DiffType(Enum):
    EQUAL = "equal"
    INSERT = "insert"      # Word exists only in transcript B
    DELETE = "delete"      # Word exists only in transcript A
    REPLACE = "replace"    # Word differs between A and B


@dataclass
class DiffSegment:
    """A segment of the diff result."""
    diff_type: DiffType
    words_a: list[str]  # Words from transcript A
    words_b: list[str]  # Words from transcript B
    position_a: int     # Starting position in transcript A
    position_b: int     # Starting position in transcript B
    
    @property
    def length_a(self) -> int:
        return len(self.words_a)
    
    @property
    def length_b(self) -> int:
        return len(self.words_b)


@dataclass 
class DiffResult:
    """Complete diff result with segments and statistics."""
    segments: list[DiffSegment]
    words_a: list[str]
    words_b: list[str]
    
    # Statistics
    total_words_a: int = 0
    total_words_b: int = 0
    matching_words: int = 0
    inserted_words: int = 0
    deleted_words: int = 0
    replaced_words_a: int = 0
    replaced_words_b: int = 0
    
    # Computed metrics
    similarity_ratio: float = 0.0
    word_error_rate: float = 0.0
    
    # Detailed analysis
    common_substitutions: list[tuple[str, str, int]] = field(default_factory=list)
    unique_to_a: list[tuple[str, int]] = field(default_factory=list)
    unique_to_b: list[tuple[str, int]] = field(default_factory=list)


def compute_diff(words_a: list[str], words_b: list[str]) -> DiffResult:
    """
    Compute the diff between two word lists.
    
    Uses SequenceMatcher to find the optimal alignment between transcripts.
    """
    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    opcodes = matcher.get_opcodes()
    
    segments = []
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            segments.append(DiffSegment(
                diff_type=DiffType.EQUAL,
                words_a=words_a[i1:i2],
                words_b=words_b[j1:j2],
                position_a=i1,
                position_b=j1,
            ))
        elif tag == "replace":
            segments.append(DiffSegment(
                diff_type=DiffType.REPLACE,
                words_a=words_a[i1:i2],
                words_b=words_b[j1:j2],
                position_a=i1,
                position_b=j1,
            ))
        elif tag == "insert":
            segments.append(DiffSegment(
                diff_type=DiffType.INSERT,
                words_a=[],
                words_b=words_b[j1:j2],
                position_a=i1,
                position_b=j1,
            ))
        elif tag == "delete":
            segments.append(DiffSegment(
                diff_type=DiffType.DELETE,
                words_a=words_a[i1:i2],
                words_b=[],
                position_a=i1,
                position_b=j1,
            ))
    
    result = DiffResult(
        segments=segments,
        words_a=words_a,
        words_b=words_b,
        total_words_a=len(words_a),
        total_words_b=len(words_b),
    )
    
    # Compute statistics
    _compute_statistics(result)
    
    return result


def _compute_statistics(result: DiffResult) -> None:
    """Compute statistics for the diff result."""
    matching = 0
    inserted = 0
    deleted = 0
    replaced_a = 0
    replaced_b = 0
    
    substitution_pairs = []
    
    for seg in result.segments:
        if seg.diff_type == DiffType.EQUAL:
            matching += seg.length_a
        elif seg.diff_type == DiffType.INSERT:
            inserted += seg.length_b
        elif seg.diff_type == DiffType.DELETE:
            deleted += seg.length_a
        elif seg.diff_type == DiffType.REPLACE:
            replaced_a += seg.length_a
            replaced_b += seg.length_b
            # Track substitution pairs for analysis
            for wa, wb in zip(seg.words_a, seg.words_b):
                substitution_pairs.append((wa, wb))
    
    result.matching_words = matching
    result.inserted_words = inserted
    result.deleted_words = deleted
    result.replaced_words_a = replaced_a
    result.replaced_words_b = replaced_b
    
    # Similarity ratio (0-1)
    total = max(result.total_words_a, result.total_words_b)
    if total > 0:
        result.similarity_ratio = matching / total
    
    # Word Error Rate (WER) - standard ASR metric
    # WER = (S + D + I) / N where N is reference length
    # Since we don't know which is "reference", use average
    ref_length = (result.total_words_a + result.total_words_b) / 2
    if ref_length > 0:
        errors = deleted + inserted + max(replaced_a, replaced_b)
        result.word_error_rate = errors / ref_length
    
    # Common substitutions analysis
    sub_counter = Counter(substitution_pairs)
    result.common_substitutions = [
        (pair[0], pair[1], count) 
        for pair, count in sub_counter.most_common(20)
    ]
    
    # Words unique to each transcript
    words_a_set = set(result.words_a)
    words_b_set = set(result.words_b)
    
    unique_a = words_a_set - words_b_set
    unique_b = words_b_set - words_a_set
    
    # Count occurrences of unique words
    a_counter = Counter(result.words_a)
    b_counter = Counter(result.words_b)
    
    result.unique_to_a = sorted(
        [(w, a_counter[w]) for w in unique_a],
        key=lambda x: -x[1]
    )[:20]
    
    result.unique_to_b = sorted(
        [(w, b_counter[w]) for w in unique_b],
        key=lambda x: -x[1]
    )[:20]


def get_context_around_diff(
    result: DiffResult, 
    segment_index: int, 
    context_words: int = 5
) -> dict:
    """
    Get context around a specific diff segment.
    
    Returns words before and after the diff from both transcripts.
    """
    if segment_index < 0 or segment_index >= len(result.segments):
        return {}
    
    segment = result.segments[segment_index]
    
    # Get context from transcript A
    start_a = max(0, segment.position_a - context_words)
    end_a = min(len(result.words_a), segment.position_a + segment.length_a + context_words)
    
    # Get context from transcript B
    start_b = max(0, segment.position_b - context_words)
    end_b = min(len(result.words_b), segment.position_b + segment.length_b + context_words)
    
    return {
        "segment": segment,
        "context_a": {
            "before": result.words_a[start_a:segment.position_a],
            "diff": segment.words_a,
            "after": result.words_a[segment.position_a + segment.length_a:end_a],
        },
        "context_b": {
            "before": result.words_b[start_b:segment.position_b],
            "diff": segment.words_b,
            "after": result.words_b[segment.position_b + segment.length_b:end_b],
        },
    }


def generate_aligned_html(result: DiffResult) -> tuple[str, str]:
    """
    Generate HTML representations of both transcripts with diff highlighting.
    
    Returns (html_a, html_b) with spans for different diff types.
    """
    html_a_parts = []
    html_b_parts = []
    
    for seg in result.segments:
        if seg.diff_type == DiffType.EQUAL:
            text_a = " ".join(seg.words_a)
            text_b = " ".join(seg.words_b)
            html_a_parts.append(f'<span class="equal">{text_a}</span>')
            html_b_parts.append(f'<span class="equal">{text_b}</span>')
        
        elif seg.diff_type == DiffType.INSERT:
            text_b = " ".join(seg.words_b)
            html_a_parts.append('<span class="gap">⋯</span>')
            html_b_parts.append(f'<span class="insert">{text_b}</span>')
        
        elif seg.diff_type == DiffType.DELETE:
            text_a = " ".join(seg.words_a)
            html_a_parts.append(f'<span class="delete">{text_a}</span>')
            html_b_parts.append('<span class="gap">⋯</span>')
        
        elif seg.diff_type == DiffType.REPLACE:
            text_a = " ".join(seg.words_a)
            text_b = " ".join(seg.words_b)
            html_a_parts.append(f'<span class="replace">{text_a}</span>')
            html_b_parts.append(f'<span class="replace">{text_b}</span>')
    
    return " ".join(html_a_parts), " ".join(html_b_parts)


def generate_unified_diff(result: DiffResult, context_lines: int = 3) -> str:
    """
    Generate a unified diff-style text output.
    """
    lines = []
    
    # Group words into "lines" of ~10 words for readable diff
    words_per_line = 10
    
    lines_a = []
    lines_b = []
    
    for i in range(0, len(result.words_a), words_per_line):
        lines_a.append(" ".join(result.words_a[i:i+words_per_line]))
    
    for i in range(0, len(result.words_b), words_per_line):
        lines_b.append(" ".join(result.words_b[i:i+words_per_line]))
    
    diff = difflib.unified_diff(
        lines_a, lines_b,
        fromfile="Transcript A",
        tofile="Transcript B",
        lineterm="",
        n=context_lines,
    )
    
    return "\n".join(diff)
