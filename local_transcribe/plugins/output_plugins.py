#!/usr/bin/env python3
"""
Output writer plugin implementations.
"""

from typing import List

from local_transcribe.core.plugins import OutputWriter, Turn, registry
from local_transcribe.output_writers.txt_writer import write_timestamped_txt, write_plain_txt, write_asr_words
from local_transcribe.output_writers.srt_vtt import write_srt, write_vtt
from local_transcribe.output_writers.csv_writer import write_conversation_csv
from local_transcribe.output_writers.markdown_writer import write_conversation_markdown


class TimestampedTextWriter(OutputWriter):
    """Output writer for timestamped text format."""

    @property
    def name(self) -> str:
        return "timestamped-txt"

    @property
    def description(self) -> str:
        return "Timestamped text format with speaker labels"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(self, turns: List[Turn], output_path: str) -> None:
        """Write turns to timestamped text file."""
        write_timestamped_txt(turns, output_path)


class PlainTextWriter(OutputWriter):
    """Output writer for plain text format."""

    @property
    def name(self) -> str:
        return "plain-txt"

    @property
    def description(self) -> str:
        return "Plain text format without timestamps"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(self, turns: List[Turn], output_path: str) -> None:
        """Write turns to plain text file."""
        write_plain_txt(turns, output_path)


class SRTWriter(OutputWriter):
    """Output writer for SRT subtitle format."""

    @property
    def name(self) -> str:
        return "srt"

    @property
    def description(self) -> str:
        return "SRT subtitle format"

    @property
    def supported_formats(self) -> List[str]:
        return [".srt"]

    def write(self, turns: List[Turn], output_path: str) -> None:
        """Write turns to SRT file."""
        write_srt(turns, output_path)


class VTTWriter(OutputWriter):
    """Output writer for WebVTT subtitle format."""

    @property
    def name(self) -> str:
        return "vtt"

    @property
    def description(self) -> str:
        return "WebVTT subtitle format"

    @property
    def supported_formats(self) -> List[str]:
        return [".vtt"]

    def write(self, turns: List[Turn], output_path: str) -> None:
        """Write turns to VTT file."""
        write_vtt(turns, output_path)


class CSVWriter(OutputWriter):
    """Output writer for CSV format."""

    @property
    def name(self) -> str:
        return "csv"

    @property
    def description(self) -> str:
        return "CSV format with conversation data"

    @property
    def supported_formats(self) -> List[str]:
        return [".csv"]

    def write(self, turns: List[Turn], output_path: str) -> None:
        """Write turns to CSV file."""
        write_conversation_csv(turns, output_path)


class MarkdownWriter(OutputWriter):
    """Output writer for Markdown format."""

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Markdown format with speaker formatting"

    @property
    def supported_formats(self) -> List[str]:
        return [".md"]

    def write(self, turns: List[Turn], output_path: str) -> None:
        """Write turns to Markdown file."""
        write_conversation_markdown(turns, output_path)


def register_output_plugins():
    """Register output writer plugins."""
    registry.register_output_writer(TimestampedTextWriter())
    registry.register_output_writer(PlainTextWriter())
    registry.register_output_writer(SRTWriter())
    registry.register_output_writer(VTTWriter())
    registry.register_output_writer(CSVWriter())
    registry.register_output_writer(MarkdownWriter())


# Auto-register on import
register_output_plugins()