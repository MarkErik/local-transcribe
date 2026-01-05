"""
Tests for CLI interface.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from io import StringIO

from ..cli import (
    format_percentage, print_header, print_statistics,
    print_substitutions, print_unique_words, print_diff_segments,
    cmd_compare, cmd_web, main
)
from ..extractor import ExtractedTranscript
from ..diff_engine import DiffResult, DiffType, DiffSegment


@pytest.fixture
def sample_transcripts():
    """Create sample transcript data for testing."""
    data_a = {
        "metadata": {"duration": 10.5},
        "words": [
            {"text": "hello"},
            {"text": "world"},
            {"text": "test"}
        ]
    }
    data_b = {
        "metadata": {"duration": 9.8},
        "words": [
            {"text": "hello"},
            {"text": "beautiful"},
            {"text": "world"}
        ]
    }

    files = []
    for data in [data_a, data_b]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            files.append(f.name)

    yield files

    # Cleanup
    for f in files:
        Path(f).unlink()


class TestFormattingFunctions:
    """Test CLI formatting functions."""

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(85.5) == "85.5%"
        assert format_percentage(0.0) == "0.0%"
        assert format_percentage(100.0) == "100.0%"

    def test_print_header(self, capsys):
        """Test header printing."""
        print_header("Test Header", "*")
        captured = capsys.readouterr()
        assert "**************" in captured.out
        assert "Test Header" in captured.out


class TestStatisticsPrinting:
    """Test statistics printing functions."""

    def test_print_statistics(self, capsys):
        """Test statistics printing."""
        # Create a mock diff result
        result = DiffResult(
            segments=[],
            words_a=["hello", "world", "test"],
            words_b=["hello", "beautiful", "world"],
            total_words_a=3,
            total_words_b=3,
            matching_words=2,
            inserted_words=1,
            deleted_words=1,
            replaced_words_a=0,
            replaced_words_b=0,
            similarity_ratio=0.667,
            word_error_rate=0.333
        )

        print_statistics(result)
        captured = capsys.readouterr()

        assert "📊 COMPARISON STATISTICS" in captured.out
        assert "3" in captured.out  # word counts
        assert "2" in captured.out  # matching words
        assert "66.7%" in captured.out  # similarity
        assert "33.3%" in captured.out  # WER

    def test_print_substitutions(self, capsys):
        """Test substitution printing."""
        result = DiffResult(
            segments=[],
            words_a=[],
            words_b=[],
            common_substitutions=[
                ("bad", "good", 3),
                ("old", "new", 1)
            ]
        )

        print_substitutions(result)
        captured = capsys.readouterr()

        assert "🔄 COMMON SUBSTITUTIONS" in captured.out
        assert "bad" in captured.out
        assert "good" in captured.out
        assert "3" in captured.out

    def test_print_unique_words(self, capsys):
        """Test unique words printing."""
        result = DiffResult(
            segments=[],
            words_a=[],
            words_b=[],
            unique_to_a=[("unique_a", 2), ("only_a", 1)],
            unique_to_b=[("unique_b", 3)]
        )

        print_unique_words(result)
        captured = capsys.readouterr()

        assert "📝 WORDS UNIQUE TO EACH TRANSCRIPT" in captured.out
        assert "unique_a" in captured.out
        assert "unique_b" in captured.out
        assert "(2x)" in captured.out

    def test_print_diff_segments(self, capsys):
        """Test diff segments printing."""
        segments = [
            DiffSegment(DiffType.EQUAL, ["hello"], ["hello"], 0, 0),
            DiffSegment(DiffType.REPLACE, ["bad"], ["good"], 1, 1),
            DiffSegment(DiffType.INSERT, [], ["new"], 2, 2),
            DiffSegment(DiffType.DELETE, ["old"], [], 3, 3)
        ]

        result = DiffResult(
            segments=segments,
            words_a=["hello", "bad", "old"],
            words_b=["hello", "good", "new"],
            total_words_a=3,
            total_words_b=3
        )

        print_diff_segments(result, max_segments=10)
        captured = capsys.readouterr()

        assert "🔍 DETAILED DIFFERENCES" in captured.out
        assert "REPLACE" in captured.out
        assert "INSERT" in captured.out
        assert "DELETE" in captured.out


class TestCommandFunctions:
    """Test CLI command functions."""

    def test_cmd_compare_success(self, sample_transcripts, capsys):
        """Test successful comparison command."""
        # Mock args object
        class MockArgs:
            def __init__(self):
                self.file_a = sample_transcripts[0]
                self.file_b = sample_transcripts[1]
                self.detailed = False
                self.unified = False
                self.max_diffs = 30

        args = MockArgs()
        result = cmd_compare(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Loading transcript A" in captured.out
        assert "Loading transcript B" in captured.out
        assert "📊 COMPARISON STATISTICS" in captured.out

    def test_cmd_compare_file_not_found(self, capsys):
        """Test comparison with nonexistent file."""
        class MockArgs:
            def __init__(self):
                self.file_a = "/nonexistent/file_a.json"
                self.file_b = "/nonexistent/file_b.json"
                self.detailed = False
                self.unified = False
                self.max_diffs = 30

        args = MockArgs()
        result = cmd_compare(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    @patch('tools.transcript_compare.web_app.run_server')
    def test_cmd_web_success(self, mock_run_server, capsys):
        """Test web command."""
        class MockArgs:
            def __init__(self):
                self.port = 5050
                self.host = "127.0.0.1"
                self.debug = True
                self.file_a = None
                self.file_b = None
                self.audio = None

        args = MockArgs()
        result = cmd_web(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Starting web interface" in captured.out
        mock_run_server.assert_called_once_with(
            host="127.0.0.1", port=5050, debug=True
        )

    @patch('tools.transcript_compare.web_app.run_server')
    def test_cmd_web_with_preload_files(self, mock_run_server, sample_transcripts, capsys):
        """Test web command with preloaded files."""
        class MockArgs:
            def __init__(self):
                self.port = 5050
                self.host = "127.0.0.1"
                self.debug = True
                self.file_a = sample_transcripts[0]
                self.file_b = sample_transcripts[1]
                self.audio = None

        args = MockArgs()
        result = cmd_web(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Pre-loading transcript A" in captured.out
        assert "Pre-loading transcript B" in captured.out
        mock_run_server.assert_called_once()


class TestMainFunction:
    """Test main CLI function."""

    def test_main_compare_command(self):
        """Test main function with compare command."""
        with patch('sys.argv', ['transcript-compare', 'compare', 'file_a.json', 'file_b.json']):
            with patch('tools.transcript_compare.cli.cmd_compare') as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_main_web_command(self):
        """Test main function with web command."""
        with patch('sys.argv', ['transcript-compare', 'web']):
            with patch('tools.transcript_compare.cli.cmd_web') as mock_cmd:
                mock_cmd.return_value = 0
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_main_no_command(self, capsys):
        """Test main function with no command."""
        with patch('sys.argv', ['transcript-compare']):
            result = main()
            assert result == 0
            captured = capsys.readouterr()
            assert "usage:" in captured.out

    def test_main_invalid_command(self, capsys):
        """Test main function with invalid command."""
        with patch('sys.argv', ['transcript-compare', 'invalid']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
            captured = capsys.readouterr()
            assert "usage:" in captured.err


class TestErrorConditions:
    """Test error conditions."""

    def test_cmd_compare_invalid_json(self, capsys):
        """Test comparison with invalid JSON."""
        # Create invalid JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()

            try:
                class MockArgs:
                    def __init__(self):
                        self.file_a = f.name
                        self.file_b = f.name  # Same file
                        self.detailed = False
                        self.unified = False
                        self.max_diffs = 30

                args = MockArgs()
                result = cmd_compare(args)

                assert result == 1
                captured = capsys.readouterr()
                assert "Error" in captured.err

            finally:
                Path(f.name).unlink()


class TestIntegration:
    """Integration tests for CLI functionality."""

    def test_full_compare_workflow(self, sample_transcripts, capsys):
        """Test full comparison workflow."""
        class MockArgs:
            def __init__(self):
                self.file_a = sample_transcripts[0]
                self.file_b = sample_transcripts[1]
                self.detailed = True
                self.unified = True
                self.max_diffs = 10

        args = MockArgs()
        result = cmd_compare(args)

        assert result == 0
        captured = capsys.readouterr()

        # Check that all expected output sections are present
        assert "Loading transcript A" in captured.out
        assert "Loading transcript B" in captured.out
        assert "📊 COMPARISON STATISTICS" in captured.out
        assert " WORDS UNIQUE TO EACH TRANSCRIPT" in captured.out
        assert "🔍 DETAILED DIFFERENCES" in captured.out
        # Unified diff may or may not be present depending on data