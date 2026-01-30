#!/usr/bin/env python3
"""
Tests for CLI module functionality.

These tests ensure that CLI argument parsing, prompt helpers, and mode
determination work correctly before and after refactoring. They cover:
- Argument parsing
- Pipeline mode determination
- Prompt helper functions
- Output format filtering
- Configuration helpers
"""

import sys
import os
import argparse

import pytest

# Add parent directory to path for local_transcribe imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_transcribe.framework.cli import (
    parse_args,
    determine_pipeline_mode,
    apply_cli_implications,
    PipelineMode,
    get_available_writers,
    filter_incompatible_writers,
    list_stages,
)


# ==============================================================================
# Argument Parsing Tests
# ==============================================================================

class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_parse_args_default_values(self):
        """Test that default values are set correctly."""
        args = parse_args([])
        
        assert args.interactive is False
        assert args.log_level == "WARNING"
        assert args.audio_files is None
        assert args.outdir is None
        assert args.single_speaker_audio is False
        assert args.de_identify is False
        assert args.vad_pipeline is False

    def test_parse_args_interactive_flag(self):
        """Test -i/--interactive flag."""
        args = parse_args(["-i"])
        assert args.interactive is True
        
        args = parse_args(["--interactive"])
        assert args.interactive is True

    def test_parse_args_log_level(self):
        """Test -l/--log-level option."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            args = parse_args(["-l", level])
            assert args.log_level == level

    def test_parse_args_audio_files_single(self):
        """Test -a/--audio-files with single file."""
        args = parse_args(["-a", "audio.m4a"])
        assert args.audio_files == ["audio.m4a"]

    def test_parse_args_audio_files_multiple(self):
        """Test -a/--audio-files with multiple files."""
        args = parse_args(["-a", "int.m4a", "part.m4a"])
        assert args.audio_files == ["int.m4a", "part.m4a"]

    def test_parse_args_outdir(self):
        """Test -o/--outdir option."""
        args = parse_args(["-o", "/output/dir"])
        assert args.outdir == "/output/dir"

    def test_parse_args_single_speaker(self):
        """Test -s/--single-speaker-audio flag."""
        args = parse_args(["-s"])
        assert args.single_speaker_audio is True

    def test_parse_args_num_speakers(self):
        """Test -n/--num-speakers option."""
        args = parse_args(["-n", "3"])
        assert args.num_speakers == 3

    def test_parse_args_system_capability(self):
        """Test -x/--system option."""
        for system in ["cuda", "mps", "cpu"]:
            args = parse_args(["-x", system])
            assert args.system == system

    def test_parse_args_de_identify(self):
        """Test -d/--de-identify flag."""
        args = parse_args(["-d"])
        assert args.de_identify is True

    def test_parse_args_provider_options(self):
        """Test provider-related options."""
        args = parse_args([
            "--transcriber-provider", "granite",
            "--transcriber-model", "granite-8b",
            "--aligner-provider", "mfa",
            "--diarization-provider", "pyannote",
        ])
        
        assert args.transcriber_provider == "granite"
        assert args.transcriber_model == "granite-8b"
        assert args.aligner_provider == "mfa"
        assert args.diarization_provider == "pyannote"

    def test_parse_args_llm_urls(self):
        """Test LLM URL options with defaults."""
        args = parse_args([])
        assert args.llm_de_identifier_url == "http://0.0.0.0:8080"
        assert args.llm_transcript_cleanup_url == "http://0.0.0.0:8080"
        assert args.remote_transcriber_url == "http://0.0.0.0:7070"

    def test_parse_args_llm_urls_custom(self):
        """Test custom LLM URL options."""
        args = parse_args([
            "--llm-de-identifier-url", "http://custom:9090",
            "--llm-transcript-cleanup-url", "http://other:9091",
            "--remote-transcriber-url", "http://remote:7071",
        ])
        
        assert args.llm_de_identifier_url == "http://custom:9090"
        assert args.llm_transcript_cleanup_url == "http://other:9091"
        assert args.remote_transcriber_url == "http://remote:7071"

    def test_parse_args_only_final_transcript(self):
        """Test --only-final-transcript flag."""
        args = parse_args(["--only-final-transcript"])
        assert args.only_final_transcript is True

    def test_parse_args_list_plugins(self):
        """Test --list-plugins flag."""
        args = parse_args(["--list-plugins"])
        assert args.list_plugins is True

    def test_parse_args_reentry_options(self):
        """Test pipeline re-entry options."""
        args = parse_args([
            "--from-diarized-json", "/path/to/checkpoint.json",
            "--audio-for-video", "/path/to/audio.m4a",
            "--mode", "vad_split_audio",
            "--speaker-map", "SPEAKER_00=Interviewer,SPEAKER_01=Participant",
            "--dry-run",
        ])
        
        assert args.from_diarized_json == "/path/to/checkpoint.json"
        assert args.audio_for_video == "/path/to/audio.m4a"
        assert args.mode == "vad_split_audio"
        assert args.speaker_map == "SPEAKER_00=Interviewer,SPEAKER_01=Participant"
        assert args.dry_run is True

    def test_parse_args_vad_pipeline(self):
        """Test --vad-pipeline flag."""
        args = parse_args(["--vad-pipeline"])
        assert args.vad_pipeline is True

    def test_parse_args_enable_cleanup(self):
        """Test --enable-cleanup flag."""
        args = parse_args(["--enable-cleanup"])
        assert args.enable_cleanup is True

    def test_parse_args_llm_url_tracking(self):
        """Test that LLM URL explicit setting is tracked."""
        # Not explicitly set
        args = parse_args([])
        assert args._llm_de_identifier_url_set is False
        
        # Explicitly set
        args = parse_args(["--llm-de-identifier-url", "http://test:8080"])
        assert args._llm_de_identifier_url_set is True


# ==============================================================================
# Pipeline Mode Determination Tests
# ==============================================================================

class TestPipelineModeDetermination:
    """Test pipeline mode determination logic."""

    def test_determine_mode_single_speaker(self):
        """Test single speaker mode detection."""
        args = argparse.Namespace(
            single_speaker_audio=True,
            audio_files=["audio.m4a"],
            vad_pipeline=False,
        )
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.SINGLE_SPEAKER

    def test_determine_mode_combined_audio_no_files(self):
        """Test combined audio mode with no files (defaults)."""
        args = argparse.Namespace(
            single_speaker_audio=False,
            audio_files=None,
            vad_pipeline=False,
        )
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.COMBINED_AUDIO

    def test_determine_mode_combined_audio_single_file(self):
        """Test combined audio mode with single file."""
        args = argparse.Namespace(
            single_speaker_audio=False,
            audio_files=["audio.m4a"],
            vad_pipeline=False,
        )
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.COMBINED_AUDIO

    def test_determine_mode_split_audio(self):
        """Test split audio mode with multiple files."""
        args = argparse.Namespace(
            single_speaker_audio=False,
            audio_files=["int.m4a", "part.m4a"],
            vad_pipeline=False,
        )
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.SPLIT_AUDIO

    def test_determine_mode_vad_split_audio(self):
        """Test VAD split audio mode with multiple files and flag."""
        args = argparse.Namespace(
            single_speaker_audio=False,
            audio_files=["int.m4a", "part.m4a"],
            vad_pipeline=True,
        )
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.VAD_SPLIT_AUDIO

    def test_single_speaker_overrides_vad_flag(self):
        """Test that single speaker mode takes precedence over VAD flag."""
        args = argparse.Namespace(
            single_speaker_audio=True,
            audio_files=["int.m4a", "part.m4a"],
            vad_pipeline=True,  # This should be ignored
        )
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.SINGLE_SPEAKER


# ==============================================================================
# CLI Implications Tests
# ==============================================================================

class TestCLIImplications:
    """Test apply_cli_implications function."""

    def test_apply_implications_returns_args(self):
        """Test that apply_cli_implications returns the args object."""
        args = argparse.Namespace(some_option=True)
        
        result = apply_cli_implications(args)
        
        assert result is args


# ==============================================================================
# Pipeline Mode Constants Tests
# ==============================================================================

class TestPipelineModeConstants:
    """Test PipelineMode class constants."""

    def test_pipeline_mode_values(self):
        """Test that PipelineMode has expected values."""
        assert PipelineMode.SINGLE_SPEAKER == "single_speaker_audio"
        assert PipelineMode.COMBINED_AUDIO == "combined_audio"
        assert PipelineMode.SPLIT_AUDIO == "split_audio"
        assert PipelineMode.VAD_SPLIT_AUDIO == "vad_split_audio"


# ==============================================================================
# Output Writer Filtering Tests (using mock registry)
# ==============================================================================

class MockRegistry:
    """Mock registry for testing output writer filtering."""
    
    def list_output_writers_with_metadata(self):
        return {
            "timestamped-txt": {
                "description": "Timestamped text format",
                "supported_modes": ["combined_audio", "split_audio", "vad_split_audio"],
            },
            "html-timeline": {
                "description": "Interactive HTML viewer",
                "supported_modes": ["combined_audio", "split_audio", "vad_split_audio"],
            },
            "csv": {
                "description": "CSV format",
                "supported_modes": ["single_speaker_audio", "combined_audio", "split_audio", "vad_split_audio"],
            },
            "video": {
                "description": "Video with subtitles",
                "supported_modes": ["combined_audio", "split_audio"],  # Not VAD
            },
            "srt": {
                "description": "SRT subtitles (internal)",
                "supported_modes": ["combined_audio", "split_audio", "vad_split_audio"],
            },
        }


class TestOutputWriterFiltering:
    """Test output writer filtering functions."""

    def test_get_available_writers_vad_mode(self):
        """Test getting available writers for VAD mode."""
        registry = MockRegistry()
        
        writers = get_available_writers("vad_split_audio", registry, exclude_internal=True)
        
        assert "timestamped-txt" in writers
        assert "html-timeline" in writers
        assert "csv" in writers
        assert "video" not in writers  # Not supported in VAD mode
        assert "srt" not in writers  # Internal, excluded

    def test_get_available_writers_combined_mode(self):
        """Test getting available writers for combined audio mode."""
        registry = MockRegistry()
        
        writers = get_available_writers("combined_audio", registry, exclude_internal=True)
        
        assert "timestamped-txt" in writers
        assert "video" in writers  # Supported in combined mode
        assert "srt" not in writers  # Internal

    def test_get_available_writers_include_internal(self):
        """Test getting available writers with internal writers included."""
        registry = MockRegistry()
        
        writers = get_available_writers("vad_split_audio", registry, exclude_internal=False)
        
        assert "srt" in writers  # Not excluded

    def test_filter_incompatible_writers(self):
        """Test filtering incompatible writer selections."""
        registry = MockRegistry()
        selected = ["timestamped-txt", "video", "csv"]
        
        compatible, removed = filter_incompatible_writers(selected, "vad_split_audio", registry)
        
        assert "timestamped-txt" in compatible
        assert "csv" in compatible
        assert "video" in removed  # Not supported in VAD mode

    def test_filter_incompatible_writers_unknown(self):
        """Test filtering unknown writer selections."""
        registry = MockRegistry()
        selected = ["timestamped-txt", "unknown-writer"]
        
        compatible, removed = filter_incompatible_writers(selected, "combined_audio", registry)
        
        assert "timestamped-txt" in compatible
        assert "unknown-writer" in removed

    def test_filter_incompatible_all_compatible(self):
        """Test filtering when all selections are compatible."""
        registry = MockRegistry()
        selected = ["timestamped-txt", "csv"]
        
        compatible, removed = filter_incompatible_writers(selected, "vad_split_audio", registry)
        
        assert len(compatible) == 2
        assert len(removed) == 0


# ==============================================================================
# Comprehensive Argument Combination Tests
# ==============================================================================

class TestArgumentCombinations:
    """Test various argument combinations."""

    def test_typical_vad_pipeline_args(self):
        """Test typical VAD pipeline arguments."""
        args = parse_args([
            "-a", "interviewer.m4a", "participant.m4a",
            "-o", "/output",
            "--vad-pipeline",
            "-d",
            "-i",
        ])
        
        assert len(args.audio_files) == 2
        assert args.outdir == "/output"
        assert args.vad_pipeline is True
        assert args.de_identify is True
        assert args.interactive is True
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.VAD_SPLIT_AUDIO

    def test_typical_single_speaker_args(self):
        """Test typical single speaker arguments."""
        args = parse_args([
            "-a", "single_speaker.m4a",
            "-o", "/output",
            "-s",
            "-d",
        ])
        
        assert args.single_speaker_audio is True
        assert args.de_identify is True
        
        mode = determine_pipeline_mode(args)
        assert mode == PipelineMode.SINGLE_SPEAKER

    def test_remote_transcriber_args(self):
        """Test remote transcriber arguments."""
        args = parse_args([
            "-a", "audio.m4a",
            "-o", "/output",
            "--transcriber-provider", "remote",
            "--remote-transcriber-url", "http://server:7070",
        ])
        
        assert args.transcriber_provider == "remote"
        assert args.remote_transcriber_url == "http://server:7070"


# ==============================================================================
# Run tests directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
