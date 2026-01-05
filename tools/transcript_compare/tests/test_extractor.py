"""
Tests for transcript extraction functionality.
"""

import json
import tempfile
from pathlib import Path
import pytest

from ..extractor import (
    detect_format, extract_words_word_level, extract_words_chunk_based,
    extract_words_segment_based, extract_from_file, extract_from_text,
    ExtractedTranscript, extract_script_from_text, ExtractedScript,
    is_script_format
)


class TestFormatDetection:
    """Test format detection logic."""

    def test_detect_word_level_format(self):
        """Test detection of word-level JSON format."""
        data = {
            "metadata": {"duration": 120.5},
            "words": [
                {"text": "hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0}
            ]
        }
        assert detect_format(data) == "word-level"

    def test_detect_chunk_based_format(self):
        """Test detection of chunk-based format."""
        data = [
            {"chunk_id": 1, "words": ["hello", "world"]},
            {"chunk_id": 2, "words": ["how", "are", "you"]}
        ]
        assert detect_format(data) == "chunk-based"

    def test_detect_segment_based_format(self):
        """Test detection of segment-based format."""
        data = {
            "segments": [
                {"text": "hello world", "words": ["hello", "world"], "start_s": 0.0}
            ]
        }
        assert detect_format(data) == "segment-based"

    def test_detect_unknown_format(self):
        """Test detection returns unknown for unrecognized formats."""
        data = {"random": "data"}
        assert detect_format(data) == "unknown"


class TestWordLevelExtraction:
    """Test word-level format extraction."""

    def test_extract_words_word_level_basic(self):
        """Test basic word extraction from word-level format."""
        data = {
            "metadata": {"source": "test"},
            "words": [
                {"text": "Hello", "start": 0.0, "end": 0.5},
                {"text": "world", "start": 0.5, "end": 1.0},
                {"text": "!", "start": 1.0, "end": 1.1}
            ]
        }
        words, metadata = extract_words_word_level(data)

        # Should lowercase and clean whitespace, but keep punctuation as separate words
        assert words == ["hello", "world", "!"]
        assert metadata == {"source": "test"}

    def test_extract_words_word_level_empty_text(self):
        """Test handling of empty or whitespace-only text."""
        data = {
            "words": [
                {"text": "hello"},
                {"text": ""},  # Empty
                {"text": "   "},  # Whitespace only
                {"text": "world"}
            ]
        }
        words, metadata = extract_words_word_level(data)
        assert words == ["hello", "world"]

    def test_extract_words_word_level_no_metadata(self):
        """Test extraction when metadata is missing."""
        data = {
            "words": [{"text": "test"}]
        }
        words, metadata = extract_words_word_level(data)
        assert words == ["test"]
        assert metadata == {}


class TestChunkBasedExtraction:
    """Test chunk-based format extraction."""

    def test_extract_words_chunk_based_basic(self):
        """Test basic chunk-based extraction."""
        data = [
            {"chunk_id": 1, "words": ["hello", "world"]},
            {"chunk_id": 2, "words": ["how", "are", "you"]}
        ]
        words, metadata = extract_words_chunk_based(data)

        assert words == ["hello", "world", "how", "are", "you"]
        assert metadata == {"total_chunks": 2}

    def test_extract_words_chunk_based_mixed_formats(self):
        """Test chunk-based with mixed word formats."""
        data = [
            {"chunk_id": 1, "words": ["hello", {"text": "world"}]},
            {"chunk_id": 2, "words": [{"text": "test"}]}
        ]
        words, metadata = extract_words_chunk_based(data)

        assert words == ["hello", "world", "test"]
        assert metadata == {"total_chunks": 2}


class TestSegmentBasedExtraction:
    """Test segment-based format extraction."""

    def test_extract_words_segment_based_basic(self):
        """Test basic segment-based extraction."""
        data = {
            "segments": [
                {
                    "text": "hello world",
                    "words": ["hello", "world"],
                    "start_s": 0.0
                },
                {
                    "text": "how are you",
                    "words": ["how", "are", "you"],
                    "start_s": 2.0
                }
            ],
            "speaker_id": "test_speaker",
            "audio_file": "test.wav"
        }
        words, metadata = extract_words_segment_based(data)

        assert words == ["hello", "world", "how", "are", "you"]
        assert metadata == {
            "total_segments": 2,
            "speaker_id": "test_speaker",
            "audio_file": "test.wav"
        }


class TestFileExtraction:
    """Test file-based extraction."""

    def test_extract_from_file_word_level(self):
        """Test extraction from word-level JSON file."""
        data = {
            "metadata": {"test": True},
            "words": [{"text": "hello"}, {"text": "world"}]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()

            try:
                result = extract_from_file(f.name)
                assert isinstance(result, ExtractedTranscript)
                assert result.words == ["hello", "world"]
                assert result.format_type == "word-level"
                assert result.metadata == {"test": True}
            finally:
                Path(f.name).unlink()

    def test_extract_from_file_nonexistent(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            extract_from_file("nonexistent.json")

    def test_extract_from_file_wrong_extension(self):
        """Test error handling for wrong file extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test")
            f.flush()

            try:
                with pytest.raises(ValueError, match="Expected JSON file"):
                    extract_from_file(f.name)
            finally:
                Path(f.name).unlink()


class TestTextExtraction:
    """Test plain text extraction."""

    def test_extract_from_text_basic(self):
        """Test basic text extraction."""
        text = "Hello world, how are you?"
        result = extract_from_text(text, "test_source")

        assert isinstance(result, ExtractedTranscript)
        assert result.words == ["hello", "world", "how", "are", "you"]
        assert result.source_file == "test_source"
        assert result.format_type == "plain-text"

    def test_extract_from_text_punctuation(self):
        """Test punctuation handling."""
        text = "Hello! World? (Test) [item] {thing} … - – —"
        result = extract_from_text(text)

        # Should remove all punctuation
        assert "hello" in result.words
        assert "world" in result.words
        assert "test" in result.words
        assert "item" in result.words
        assert "thing" in result.words
        # Should not contain punctuation characters
        assert not any(word in result.words for word in ["!", "?", "(", ")", "[", "]", "{", "}", "…", "-", "–", "—"])


class TestScriptExtraction:
    """Test script format extraction."""

    def test_extract_script_from_text_basic(self):
        """Test basic script extraction."""
        script_text = """Duration: 120.5s | Turns: 3 | Speakers: 2

SPEAKER1:
   (0.00s) Hello world
---
SPEAKER2:
   (5.50s) How are you doing
---
SPEAKER1:
   (10.25s) I'm doing well, thank you
---
"""

        result = extract_script_from_text(script_text, "test.script.txt")

        assert isinstance(result, ExtractedScript)
        assert len(result.turns) == 3
        assert result.turns[0].speaker == "SPEAKER1"
        assert result.turns[0].timestamp == 0.00
        assert result.turns[0].words == ["hello", "world"]
        assert result.metadata["duration"] == "120.5s"

    def test_extract_script_from_text_with_interjections(self):
        """Test script extraction with interjections."""
        script_text = """SPEAKER1:
   (0.00s) Hello [SPEAKER2: (0.50s) Hi there] world
---
"""

        result = extract_script_from_text(script_text)

        assert len(result.turns) == 1
        turn = result.turns[0]
        assert turn.speaker == "SPEAKER1"
        assert len(turn.interjections) == 1
        assert turn.interjections[0] == ("SPEAKER2", 0.50, "Hi there")
        assert turn.words == ["hello", "world"]  # Interjection removed from main text

    def test_extract_script_from_text_complex_formatting(self):
        """Test script extraction with complex formatting."""
        script_text = """SPEAKER1:
   (0.00s) Hello world
   How are you today?
---
SPEAKER2:
   (5.50s) I'm doing well
   Thanks for asking
---
"""

        result = extract_script_from_text(script_text)

        assert len(result.turns) == 2
        assert result.turns[0].words == ["hello", "world", "how", "are", "you", "today"]
        assert result.turns[1].words == ["i'm", "doing", "well", "thanks", "for", "asking"]


class TestScriptFormatDetection:
    """Test script format detection."""

    def test_is_script_format_positive(self):
        """Test detection of script format files."""
        # Create a temporary script file
        script_content = """CONVERSATION TRANSCRIPT

SPEAKER1:
   (0.00s) Hello world
---
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.script.txt', delete=False) as f:
            f.write(script_content)
            f.flush()

            try:
                assert is_script_format(f.name)
            finally:
                Path(f.name).unlink()

    def test_is_script_format_negative(self):
        """Test non-script files are not detected as script format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Just some regular text")
            f.flush()

            try:
                assert not is_script_format(f.name)
            finally:
                Path(f.name).unlink()


class TestExtractedTranscript:
    """Test ExtractedTranscript dataclass."""

    def test_properties(self):
        """Test ExtractedTranscript properties."""
        transcript = ExtractedTranscript(
            words=["hello", "world", "test"],
            source_file="test.json",
            format_type="word-level",
            metadata={"duration": 10.5}
        )

        assert transcript.word_count == 3
        assert transcript.text == "hello world test"

    def test_empty_transcript(self):
        """Test empty transcript properties."""
        transcript = ExtractedTranscript(
            words=[],
            source_file="empty.json",
            format_type="word-level"
        )

        assert transcript.word_count == 0
        assert transcript.text == ""


class TestExtractedScript:
    """Test ExtractedScript dataclass."""

    def test_properties(self):
        """Test ExtractedScript properties."""
        from ..extractor import ScriptTurn

        turns = [
            ScriptTurn("SPEAKER1", 0.0, "Hello world"),
            ScriptTurn("SPEAKER2", 5.0, "How are you")
        ]

        script = ExtractedScript(
            turns=turns,
            source_file="test.script.txt",
            metadata={"duration": "10s"}
        )

        assert script.total_turns == 2
        assert script.total_words == 5  # hello world how are you
        assert set(script.speakers) == {"SPEAKER1", "SPEAKER2"}
        assert script.all_words == ["hello", "world", "how", "are", "you"]
        assert "Hello world How are you" in script.full_text