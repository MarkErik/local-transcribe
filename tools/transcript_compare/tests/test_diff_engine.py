"""
Tests for diff engine functionality.
"""

import pytest
from ..diff_engine import (
    compute_diff, DiffResult, DiffType, DiffSegment,
    analyze_transcript, TranscriptAnalysis, RepetitionInfo,
    generate_aligned_html, generate_unified_diff,
    compare_analyses, serialize_analysis, find_similar_word_pairs,
    FILLER_WORDS, SINGLE_FILLER_WORDS
)


class TestDiffComputation:
    """Test diff computation logic."""

    def test_compute_diff_identical(self):
        """Test diff of identical transcripts."""
        words_a = ["hello", "world", "test"]
        words_b = ["hello", "world", "test"]

        result = compute_diff(words_a, words_b)

        assert isinstance(result, DiffResult)
        assert len(result.segments) == 1
        assert result.segments[0].diff_type == DiffType.EQUAL
        assert result.similarity_ratio == 1.0
        assert result.word_error_rate == 0.0
        assert result.matching_words == 3
        assert result.inserted_words == 0
        assert result.deleted_words == 0
        assert result.replaced_words_a == 0

    def test_compute_diff_insertion(self):
        """Test diff with insertions."""
        words_a = ["hello", "world"]
        words_b = ["hello", "beautiful", "world"]

        result = compute_diff(words_a, words_b)

        assert result.inserted_words == 1
        assert result.matching_words == 2
        assert result.similarity_ratio > 0.5

        # Check segments
        assert len(result.segments) == 3  # equal, insert, equal
        assert result.segments[1].diff_type == DiffType.INSERT
        assert result.segments[1].words_b == ["beautiful"]

    def test_compute_diff_deletion(self):
        """Test diff with deletions."""
        words_a = ["hello", "boring", "world"]
        words_b = ["hello", "world"]

        result = compute_diff(words_a, words_b)

        assert result.deleted_words == 1
        assert result.matching_words == 2

        # Check segments
        assert any(seg.diff_type == DiffType.DELETE for seg in result.segments)

    def test_compute_diff_replacement(self):
        """Test diff with replacements."""
        words_a = ["hello", "bad", "world"]
        words_b = ["hello", "good", "world"]

        result = compute_diff(words_a, words_b)

        assert result.replaced_words_a == 1
        assert result.replaced_words_b == 1
        assert result.matching_words == 2

        # Check segments
        assert any(seg.diff_type == DiffType.REPLACE for seg in result.segments)

    def test_compute_diff_empty(self):
        """Test diff with empty transcripts."""
        result = compute_diff([], [])

        assert result.total_words_a == 0
        assert result.total_words_b == 0
        assert result.similarity_ratio == 0.0  # Special case
        assert result.word_error_rate == 0.0

    def test_compute_diff_statistics(self):
        """Test statistical calculations."""
        words_a = ["the", "quick", "brown", "fox"]
        words_b = ["the", "fast", "brown", "dog"]

        result = compute_diff(words_a, words_b)

        # Should have 2 matches, 2 replacements
        assert result.matching_words == 2
        assert result.replaced_words_a == 2
        assert result.replaced_words_b == 2

        # Similarity should be reasonable
        assert 0.4 < result.similarity_ratio < 0.7


class TestDiffSegments:
    """Test DiffSegment functionality."""

    def test_diff_segment_properties(self):
        """Test DiffSegment property calculations."""
        segment = DiffSegment(
            diff_type=DiffType.REPLACE,
            words_a=["bad"],
            words_b=["good"],
            position_a=1,
            position_b=1
        )

        assert segment.length_a == 1
        assert segment.length_b == 1

    def test_diff_segment_empty(self):
        """Test DiffSegment with empty words."""
        segment = DiffSegment(
            diff_type=DiffType.INSERT,
            words_a=[],
            words_b=["new", "words"],
            position_a=0,
            position_b=0
        )

        assert segment.length_a == 0
        assert segment.length_b == 2


class TestTranscriptAnalysis:
    """Test transcript analysis functionality."""

    def test_analyze_transcript_basic(self):
        """Test basic transcript analysis."""
        words = ["hello", "world", "hello", "test"]

        analysis = analyze_transcript(words)

        assert isinstance(analysis, TranscriptAnalysis)
        assert analysis.total_words == 4
        assert analysis.unique_words == 3  # hello appears twice
        assert analysis.vocabulary_richness == 3/4

    def test_analyze_transcript_repetitions(self):
        """Test repetition detection."""
        words = ["the", "the", "quick", "quick", "brown", "fox"]

        analysis = analyze_transcript(words)

        # Should detect consecutive repetitions
        assert len(analysis.consecutive_repetitions) >= 1
        assert any(r.text == "the" and r.count == 2 for r in analysis.consecutive_repetitions)
        assert any(r.text == "quick" and r.count == 2 for r in analysis.consecutive_repetitions)

    def test_analyze_transcript_filler_words(self):
        """Test filler word detection."""
        words = ["um", "hello", "uh", "world", "like", "test"]

        analysis = analyze_transcript(words)

        assert analysis.total_filler_count == 3  # um, uh, like
        assert "um" in analysis.filler_words
        assert "uh" in analysis.filler_words
        assert "like" in analysis.filler_words

    def test_analyze_transcript_ngrams(self):
        """Test n-gram analysis."""
        words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "the", "quick", "brown"]

        analysis = analyze_transcript(words)

        # Should find some bigrams and trigrams
        assert len(analysis.bigrams) > 0

        # "the quick brown" should appear
        trigram_texts = [b[0] for b in analysis.trigrams]
        assert "the quick brown" in trigram_texts

    def test_analyze_transcript_empty(self):
        """Test analysis of empty transcript."""
        analysis = analyze_transcript([])

        assert analysis.total_words == 0
        assert analysis.unique_words == 0
        assert analysis.vocabulary_richness == 0.0
        assert analysis.avg_word_length == 0.0


class TestRepetitionInfo:
    """Test RepetitionInfo dataclass."""

    def test_repetition_info_consecutive(self):
        """Test consecutive repetition info."""
        rep = RepetitionInfo(
            text="the",
            count=3,
            positions=[5],
            is_consecutive=True
        )

        assert rep.text == "the"
        assert rep.count == 3
        assert rep.is_consecutive

    def test_repetition_info_non_consecutive(self):
        """Test non-consecutive repetition info."""
        rep = RepetitionInfo(
            text="hello",
            count=2,
            positions=[1, 10],
            is_consecutive=False
        )

        assert not rep.is_consecutive
        assert len(rep.positions) == 2


class TestHTMLGeneration:
    """Test HTML generation functions."""

    def test_generate_aligned_html(self):
        """Test aligned HTML generation."""
        words_a = ["hello", "world"]
        words_b = ["hello", "beautiful", "world"]

        result = compute_diff(words_a, words_b)
        html_a, html_b = generate_aligned_html(result)

        # Should contain HTML spans
        assert "<span class=\"equal\">" in html_a
        assert "<span class=\"insert\">" in html_b
        assert "beautiful" in html_b

    def test_generate_unified_diff(self):
        """Test unified diff generation."""
        words_a = ["hello", "world"]
        words_b = ["hello", "beautiful", "world"]

        result = compute_diff(words_a, words_b)
        diff_text = generate_unified_diff(result)

        assert "--- Transcript A" in diff_text
        assert "+++ Transcript B" in diff_text
        assert "hello beautiful world" in diff_text


class TestAnalysisComparison:
    """Test analysis comparison functionality."""

    def test_compare_analyses(self):
        """Test comparison of two analyses."""
        analysis_a = TranscriptAnalysis(
            total_words=100,
            unique_words=80,
            filler_words={"um": 5, "uh": 3},
            total_filler_count=8,
            consecutive_repetitions=[RepetitionInfo("the", 2, [1], True)]
        )

        analysis_b = TranscriptAnalysis(
            total_words=95,
            unique_words=75,
            filler_words={"um": 2},
            total_filler_count=2,
            consecutive_repetitions=[]
        )

        comparison = compare_analyses(analysis_a, analysis_b)

        assert "vocabulary_comparison" in comparison
        assert "filler_comparison" in comparison
        assert "repetition_comparison" in comparison

        # Filler reduction should be reflected in the counts
        assert comparison["filler_comparison"]["total_a"] == 8
        assert comparison["filler_comparison"]["total_b"] == 2

    def test_serialize_analysis(self):
        """Test analysis serialization."""
        analysis = TranscriptAnalysis(
            total_words=50,
            unique_words=40,
            vocabulary_richness=0.8,
            avg_word_length=4.5,
            filler_words={"um": 2, "uh": 1},
            total_filler_count=3,
            repeated_words=[RepetitionInfo("the", 5, [1, 2, 3, 4, 5], False)],
            consecutive_repetitions=[RepetitionInfo("very", 3, [10], True)]
        )

        serialized = serialize_analysis(analysis)

        assert serialized["total_words"] == 50
        assert serialized["unique_words"] == 40
        assert serialized["vocabulary_richness"] == 80.0  # percentage
        assert len(serialized["repeated_words"]) > 0
        assert len(serialized["consecutive_repetitions"]) > 0


class TestSimilarWordPairs:
    """Test similar word pair detection."""

    def test_find_similar_word_pairs(self):
        """Test finding similar word pairs."""
        # Create a diff result with some replacements
        words_a = ["hello", "teh", "world", "recieve"]
        words_b = ["hello", "the", "world", "receive"]

        result = compute_diff(words_a, words_b)
        similar_pairs = find_similar_word_pairs(result)

        # Should find similar pairs
        assert len(similar_pairs) > 0

        # Check that pairs are found
        pair_texts = [(p["word_a"], p["word_b"]) for p in similar_pairs]
        assert ("teh", "the") in pair_texts or ("recieve", "receive") in pair_texts

    def test_find_similar_word_pairs_no_similar(self):
        """Test with completely different words."""
        words_a = ["cat", "dog"]
        words_b = ["elephant", "giraffe"]

        result = compute_diff(words_a, words_b)
        similar_pairs = find_similar_word_pairs(result)

        # Should not find highly similar pairs
        assert len(similar_pairs) == 0


class TestConstants:
    """Test constant definitions."""

    def test_filler_words_defined(self):
        """Test that filler word constants are defined."""
        assert isinstance(FILLER_WORDS, set)
        assert isinstance(SINGLE_FILLER_WORDS, set)

        # Check some common fillers are included
        assert "um" in SINGLE_FILLER_WORDS
        assert "uh" in SINGLE_FILLER_WORDS
        assert "like" in SINGLE_FILLER_WORDS

        # Check multi-word fillers
        assert any(" " in filler for filler in FILLER_WORDS)

    def test_filler_words_overlap(self):
        """Test that single fillers are subset of all fillers."""
        assert SINGLE_FILLER_WORDS.issubset(FILLER_WORDS)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_transcript(self):
        """Test with a very long transcript."""
        # Create long transcripts
        words_a = ["word"] * 1000
        words_b = ["word"] * 995 + ["different"] * 5

        result = compute_diff(words_a, words_b)

        assert result.total_words_a == 1000
        assert result.total_words_b == 1000
        assert result.similarity_ratio < 1.0

    def test_unicode_words(self):
        """Test with unicode characters."""
        words_a = ["héllo", "wörld"]
        words_b = ["hello", "world"]

        result = compute_diff(words_a, words_b)

        assert result.replaced_words_a == 2
        assert result.matching_words == 0

    def test_case_sensitivity(self):
        """Test case sensitivity in diff."""
        words_a = ["Hello", "World"]
        words_b = ["hello", "world"]

        result = compute_diff(words_a, words_b)

        # Should be considered different (case-sensitive)
        assert result.matching_words == 0
        assert result.replaced_words_a == 2