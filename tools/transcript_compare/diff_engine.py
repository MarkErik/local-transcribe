"""
Diff analysis engine for comparing transcripts.

Uses sequence matching to identify:
- Exact matches
- Insertions (words in one transcript but not the other)
- Deletions
- Substitutions (different words at same position)

Also provides detailed analysis:
- Repeated words/phrases detection
- N-gram analysis
- Filler word detection
- Word frequency analysis
"""

import difflib
import re
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


@dataclass
class RepetitionInfo:
    """Information about a repeated word or phrase."""
    text: str
    count: int
    positions: list[int]  # Starting positions in word list
    is_consecutive: bool  # True if it's a stutter (consecutive)


@dataclass
class TranscriptAnalysis:
    """Detailed analysis of a single transcript."""
    # Repetitions
    repeated_words: list[RepetitionInfo] = field(default_factory=list)
    repeated_phrases: list[RepetitionInfo] = field(default_factory=list)
    consecutive_repetitions: list[RepetitionInfo] = field(default_factory=list)  # Stutters
    
    # Filler words
    filler_words: dict[str, int] = field(default_factory=dict)
    total_filler_count: int = 0
    filler_percentage: float = 0.0
    
    # Word frequency
    word_frequency: list[tuple[str, int]] = field(default_factory=list)
    
    # N-grams (common phrases)
    bigrams: list[tuple[str, int]] = field(default_factory=list)
    trigrams: list[tuple[str, int]] = field(default_factory=list)
    
    # Statistics
    total_words: int = 0
    unique_words: int = 0
    vocabulary_richness: float = 0.0  # unique/total
    avg_word_length: float = 0.0
    total_characters: int = 0


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


# Common filler words and hesitation markers
FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "er", "err", "ah", "ahh",
    "like", "you know", "i mean", "basically", "actually",
    "literally", "right", "so", "well", "okay", "ok",
    "kind of", "sort of", "kinda", "sorta",
    "hmm", "hm", "mm", "mhm", "uh-huh", "yeah", "yep",
}

# Single-word fillers for easy detection
SINGLE_FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "er", "err", "ah", "ahh",
    "like", "basically", "actually", "literally", "right",
    "so", "well", "okay", "ok", "hmm", "hm", "mm", "mhm",
    "yeah", "yep",
}


def analyze_transcript(words: list[str]) -> TranscriptAnalysis:
    """
    Perform detailed analysis of a single transcript.
    
    Detects:
    - Repeated words (words appearing multiple times)
    - Repeated phrases (2-4 word sequences appearing multiple times)
    - Consecutive repetitions (stutters like "the the" or "I I I")
    - Filler words
    - Word frequency
    - N-grams
    """
    analysis = TranscriptAnalysis()
    analysis.total_words = len(words)
    
    if not words:
        return analysis
    
    # Normalize words for analysis
    normalized = [w.lower().strip() for w in words]
    
    # Word frequency
    word_counter = Counter(normalized)
    analysis.word_frequency = word_counter.most_common(30)
    analysis.unique_words = len(word_counter)
    analysis.vocabulary_richness = analysis.unique_words / analysis.total_words if analysis.total_words > 0 else 0
    
    # Character statistics
    analysis.total_characters = sum(len(w) for w in words)
    analysis.avg_word_length = analysis.total_characters / analysis.total_words if analysis.total_words > 0 else 0
    
    # Find repeated words (appearing more than once)
    repeated = [(word, count) for word, count in word_counter.items() 
                if count > 1 and len(word) > 2]  # Ignore tiny words
    repeated.sort(key=lambda x: -x[1])
    
    for word, count in repeated[:20]:
        positions = [i for i, w in enumerate(normalized) if w == word]
        analysis.repeated_words.append(RepetitionInfo(
            text=word,
            count=count,
            positions=positions,
            is_consecutive=False
        ))
    
    # Find consecutive repetitions (stutters)
    _find_consecutive_repetitions(normalized, analysis)
    
    # Find repeated phrases (bigrams, trigrams)
    _analyze_ngrams(normalized, analysis)
    
    # Filler word analysis
    _analyze_fillers(normalized, analysis)
    
    return analysis


def _find_consecutive_repetitions(words: list[str], analysis: TranscriptAnalysis) -> None:
    """Find consecutive word repetitions (stutters)."""
    i = 0
    while i < len(words):
        word = words[i]
        count = 1
        start = i
        
        # Count consecutive occurrences
        while i + count < len(words) and words[i + count] == word:
            count += 1
        
        if count > 1:
            analysis.consecutive_repetitions.append(RepetitionInfo(
                text=word,
                count=count,
                positions=[start],
                is_consecutive=True
            ))
        
        i += count
    
    # Also look for repeated short phrases (2-3 words)
    for phrase_len in [2, 3]:
        i = 0
        while i < len(words) - phrase_len * 2 + 1:
            phrase = tuple(words[i:i + phrase_len])
            next_phrase = tuple(words[i + phrase_len:i + phrase_len * 2])
            
            if phrase == next_phrase:
                phrase_text = " ".join(phrase)
                count = 2
                # Check for more consecutive repetitions
                while (i + phrase_len * (count + 1) <= len(words) and
                       tuple(words[i + phrase_len * count:i + phrase_len * (count + 1)]) == phrase):
                    count += 1
                
                analysis.consecutive_repetitions.append(RepetitionInfo(
                    text=phrase_text,
                    count=count,
                    positions=[i],
                    is_consecutive=True
                ))
                i += phrase_len * count
            else:
                i += 1


def _analyze_ngrams(words: list[str], analysis: TranscriptAnalysis) -> None:
    """Analyze bigrams and trigrams."""
    if len(words) < 2:
        return
    
    # Bigrams
    bigrams = [" ".join(words[i:i+2]) for i in range(len(words) - 1)]
    bigram_counter = Counter(bigrams)
    # Filter to phrases that appear more than once
    analysis.bigrams = [(phrase, count) for phrase, count in bigram_counter.most_common(20)
                        if count > 1]
    
    # Trigrams
    if len(words) >= 3:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counter = Counter(trigrams)
        analysis.trigrams = [(phrase, count) for phrase, count in trigram_counter.most_common(20)
                             if count > 1]
    
    # Find repeated phrases (non-consecutive)
    for phrase, count in analysis.bigrams + analysis.trigrams:
        if count > 2:  # Only notable repetitions
            phrase_words = phrase.split()
            positions = []
            for i in range(len(words) - len(phrase_words) + 1):
                if words[i:i+len(phrase_words)] == phrase_words:
                    positions.append(i)
            
            analysis.repeated_phrases.append(RepetitionInfo(
                text=phrase,
                count=count,
                positions=positions,
                is_consecutive=False
            ))


def _analyze_fillers(words: list[str], analysis: TranscriptAnalysis) -> None:
    """Analyze filler words and hesitation markers."""
    filler_counts = {}
    
    for word in words:
        if word in SINGLE_FILLER_WORDS:
            filler_counts[word] = filler_counts.get(word, 0) + 1
    
    # Also check for multi-word fillers
    text = " ".join(words)
    for filler in FILLER_WORDS:
        if " " in filler:  # Multi-word filler
            count = text.lower().count(filler)
            if count > 0:
                filler_counts[filler] = count
    
    analysis.filler_words = dict(sorted(filler_counts.items(), key=lambda x: -x[1]))
    analysis.total_filler_count = sum(filler_counts.values())
    analysis.filler_percentage = (analysis.total_filler_count / len(words) * 100) if words else 0


def compare_analyses(analysis_a: TranscriptAnalysis, analysis_b: TranscriptAnalysis) -> dict:
    """
    Compare two transcript analyses and highlight differences.
    
    Returns a dictionary with comparison insights.
    """
    comparison = {
        "vocabulary_comparison": {
            "richness_a": round(analysis_a.vocabulary_richness * 100, 2),
            "richness_b": round(analysis_b.vocabulary_richness * 100, 2),
            "unique_words_a": analysis_a.unique_words,
            "unique_words_b": analysis_b.unique_words,
        },
        "filler_comparison": {
            "percentage_a": round(analysis_a.filler_percentage, 2),
            "percentage_b": round(analysis_b.filler_percentage, 2),
            "total_a": analysis_a.total_filler_count,
            "total_b": analysis_b.total_filler_count,
        },
        "repetition_comparison": {
            "stutters_a": len(analysis_a.consecutive_repetitions),
            "stutters_b": len(analysis_b.consecutive_repetitions),
            "repeated_phrases_a": len(analysis_a.repeated_phrases),
            "repeated_phrases_b": len(analysis_b.repeated_phrases),
        },
        "length_comparison": {
            "avg_word_length_a": round(analysis_a.avg_word_length, 2),
            "avg_word_length_b": round(analysis_b.avg_word_length, 2),
            "total_chars_a": analysis_a.total_characters,
            "total_chars_b": analysis_b.total_characters,
        }
    }
    
    return comparison


def serialize_analysis(analysis: TranscriptAnalysis) -> dict:
    """Convert TranscriptAnalysis to JSON-serializable dict."""
    return {
        "total_words": analysis.total_words,
        "unique_words": analysis.unique_words,
        "vocabulary_richness": round(analysis.vocabulary_richness * 100, 2),
        "avg_word_length": round(analysis.avg_word_length, 2),
        "total_characters": analysis.total_characters,
        "filler_words": analysis.filler_words,
        "total_filler_count": analysis.total_filler_count,
        "filler_percentage": round(analysis.filler_percentage, 2),
        "word_frequency": [
            {"word": w, "count": c} for w, c in analysis.word_frequency[:15]
        ],
        "repeated_words": [
            {"text": r.text, "count": r.count, "positions": r.positions[:5]}
            for r in analysis.repeated_words[:10]
        ],
        "consecutive_repetitions": [
            {"text": r.text, "count": r.count, "position": r.positions[0] if r.positions else 0}
            for r in analysis.consecutive_repetitions[:15]
        ],
        "repeated_phrases": [
            {"text": r.text, "count": r.count}
            for r in analysis.repeated_phrases[:10]
        ],
        "bigrams": [
            {"phrase": p, "count": c} for p, c in analysis.bigrams[:10]
        ],
        "trigrams": [
            {"phrase": p, "count": c} for p, c in analysis.trigrams[:10]
        ],
    }


def find_similar_word_pairs(result: DiffResult) -> list[dict]:
    """
    Find word pairs from substitutions that are phonetically or visually similar.
    These are likely transcription errors or homophones.
    
    Returns list of dicts with word_a, word_b, similarity, count.
    """
    similar_pairs = []
    
    for seg in result.segments:
        if seg.diff_type != DiffType.REPLACE:
            continue
            
        # Compare individual word pairs in the replacement
        for wa, wb in zip(seg.words_a, seg.words_b):
            wa_lower = wa.lower()
            wb_lower = wb.lower()
            
            # Skip if they're the same after normalization
            if wa_lower == wb_lower:
                continue
            
            # Calculate similarity metrics
            similarity_score = _calculate_word_similarity(wa_lower, wb_lower)
            
            if similarity_score >= 0.5:  # At least 50% similar
                similar_pairs.append({
                    "word_a": wa,
                    "word_b": wb,
                    "similarity": round(similarity_score * 100, 1),
                    "likely_type": _classify_difference(wa_lower, wb_lower)
                })
    
    # Deduplicate and count
    pair_counts = Counter((p["word_a"].lower(), p["word_b"].lower()) for p in similar_pairs)
    
    unique_pairs = []
    seen = set()
    for pair in similar_pairs:
        key = (pair["word_a"].lower(), pair["word_b"].lower())
        if key not in seen:
            seen.add(key)
            pair["count"] = pair_counts[key]
            unique_pairs.append(pair)
    
    # Sort by count, then by similarity
    unique_pairs.sort(key=lambda x: (-x["count"], -x["similarity"]))
    
    return unique_pairs[:20]


def _calculate_word_similarity(word_a: str, word_b: str) -> float:
    """Calculate similarity between two words using multiple methods."""
    if not word_a or not word_b:
        return 0.0
    
    # Method 1: Levenshtein-based ratio
    seq_ratio = difflib.SequenceMatcher(None, word_a, word_b).ratio()
    
    # Method 2: Shared character ratio
    chars_a = set(word_a)
    chars_b = set(word_b)
    shared = len(chars_a & chars_b)
    total = len(chars_a | chars_b)
    char_ratio = shared / total if total > 0 else 0
    
    # Method 3: Length similarity
    len_ratio = min(len(word_a), len(word_b)) / max(len(word_a), len(word_b))
    
    # Weighted combination
    return (seq_ratio * 0.6) + (char_ratio * 0.25) + (len_ratio * 0.15)


def _classify_difference(word_a: str, word_b: str) -> str:
    """Classify the type of difference between two words."""
    if len(word_a) == len(word_b):
        # Same length - likely single character difference
        diff_chars = sum(1 for a, b in zip(word_a, word_b) if a != b)
        if diff_chars == 1:
            return "single_char"
        elif diff_chars == 2:
            return "double_char"
    
    # Check for common suffixes/prefixes differing
    if word_a.startswith(word_b) or word_b.startswith(word_a):
        return "suffix_diff"
    if word_a.endswith(word_b) or word_b.endswith(word_a):
        return "prefix_diff"
    
    # Check for possible homophones (common patterns)
    homophone_patterns = [
        ("their", "there", "they're"),
        ("your", "you're"),
        ("its", "it's"),
        ("to", "too", "two"),
        ("than", "then"),
        ("affect", "effect"),
        ("weather", "whether"),
        ("right", "write"),
        ("hear", "here"),
        ("know", "no"),
    ]
    
    for pattern in homophone_patterns:
        if word_a in pattern and word_b in pattern:
            return "homophone"
    
    # Check for common transcription confusions
    if abs(len(word_a) - len(word_b)) <= 2:
        return "similar_sound"
    
    return "other"
