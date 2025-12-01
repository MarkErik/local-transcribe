#!/usr/bin/env python3
"""
Dialogue Script Writer for hierarchical transcripts.

This writer produces screenplay/stage play style output that preserves
the hierarchical structure of TranscriptFlow, showing interjections
inline within the dialogue flow.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pathlib import Path
import textwrap

from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow
from local_transcribe.providers.file_writers.format_utils import (
    format_timestamp,
    format_duration,
    format_speaker_name
)


def write_dialogue_script(transcript: TranscriptFlow, path: str | Path) -> None:
    """
    Write a TranscriptFlow as a dialogue script with inline interjections.
    
    Format follows screenplay conventions:
    - Speaker names in CAPS
    - Timestamps in parentheses
    - Interjections in brackets within the flow
    - Word wrapping with proper indentation
    
    Args:
        transcript: TranscriptFlow object with hierarchical turn structure
        path: Output file path
    """
    path = Path(path)
    lines = []
    
    # Configuration
    LINE_WIDTH = 75
    SPEAKER_WIDTH = 12
    INDENT = " " * SPEAKER_WIDTH
    SEPARATOR = "-" * LINE_WIDTH
    
    # Extract data from TranscriptFlow
    if not hasattr(transcript, 'turns') or not hasattr(transcript, 'metadata'):
        raise ValueError("Expected TranscriptFlow object with 'turns' and 'metadata' attributes")
    
    turns = transcript.turns
    metadata = transcript.metadata
    conversation_metrics = getattr(transcript, 'conversation_metrics', {})
    
    # Header
    lines.append("CONVERSATION TRANSCRIPT")
    lines.append("=" * len("CONVERSATION TRANSCRIPT"))
    lines.append("")
    
    # Summary line
    duration = conversation_metrics.get('total_duration', metadata.get('duration', 0))
    total_turns = conversation_metrics.get('total_turns', len(turns))
    speakers = metadata.get('speakers', [])
    speaker_str = ", ".join(format_speaker_name(s) for s in speakers) if speakers else "Unknown"
    
    # Format duration as MM:SS
    total_seconds = int(duration)
    mins = total_seconds // 60
    secs = total_seconds % 60
    duration_str = f"{mins}:{secs:02d}"
    
    lines.append(f"Duration: {duration_str} | Turns: {total_turns} | Speakers: {speaker_str}")
    lines.append("")
    lines.append(SEPARATOR)
    lines.append("")
    
    # Process each turn
    for turn in turns:
        speaker = getattr(turn, 'primary_speaker', 'Unknown')
        start = getattr(turn, 'start', 0)
        text = getattr(turn, 'text', '')
        interjections = getattr(turn, 'interjections', [])
        
        # Format speaker name (uppercase, fixed width)
        speaker_label = format_speaker_name(speaker).upper()
        
        if interjections:
            # Turn with interjections - need to interleave them in the text
            _write_turn_with_interjections(
                lines, speaker_label, start, text, interjections,
                LINE_WIDTH, SPEAKER_WIDTH, INDENT
            )
        else:
            # Simple turn without interjections
            _write_simple_turn(
                lines, speaker_label, start, text,
                LINE_WIDTH, SPEAKER_WIDTH, INDENT
            )
        
        lines.append("")
        lines.append(SEPARATOR)
        lines.append("")
    
    # Write to file
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_simple_turn(
    lines: List[str],
    speaker: str,
    start: float,
    text: str,
    line_width: int,
    speaker_width: int,
    indent: str
) -> None:
    """Write a simple turn without interjections."""
    # First line: SPEAKER: (timestamp) text...
    timestamp = f"({start:.2f}s)"
    first_line_prefix = f"{speaker}: {timestamp} "
    
    # Calculate remaining width for text on first line
    first_line_text_width = line_width - len(first_line_prefix)
    
    # Wrap the text
    wrapped = textwrap.wrap(text, width=line_width - len(indent))
    
    if wrapped:
        # First line with speaker and timestamp
        lines.append(f"{speaker}: {timestamp} {wrapped[0]}")
        
        # Subsequent lines with indent
        for line in wrapped[1:]:
            lines.append(f"{indent}{line}")


def _write_turn_with_interjections(
    lines: List[str],
    speaker: str,
    start: float,
    text: str,
    interjections: List[Any],
    line_width: int,
    speaker_width: int,
    indent: str
) -> None:
    """
    Write a turn with interjections interleaved in the text flow.
    
    Since we don't have word-level timestamps in the text, we show
    interjections after the main text block with a (continues) marker
    if there's more content implied.
    """
    # Sort interjections by start time
    sorted_interjections = sorted(interjections, key=lambda x: getattr(x, 'start', 0))
    
    # First line: SPEAKER: (timestamp) text...
    timestamp = f"({start:.2f}s)"
    
    # Wrap the main text
    wrapped = textwrap.wrap(text, width=line_width - len(indent))
    
    if wrapped:
        # First line with speaker and timestamp
        lines.append(f"{speaker}: {timestamp} {wrapped[0]}")
        
        # Subsequent lines with indent
        for line in wrapped[1:]:
            lines.append(f"{indent}{line}")
    
    # Add interjections
    lines.append("")
    for ij in sorted_interjections:
        ij_speaker = format_speaker_name(getattr(ij, 'speaker', 'Unknown')).upper()
        ij_start = getattr(ij, 'start', 0)
        ij_text = getattr(ij, 'text', '')
        
        interjection_line = f"{indent}[{ij_speaker}: ({ij_start:.2f}s) {ij_text}]"
        lines.append(interjection_line)


class DialogueScriptWriter(OutputWriter):
    """Writer for dialogue script format (screenplay style)."""
    
    @property
    def name(self) -> str:
        return "dialogue-script"
    
    @property
    def description(self) -> str:
        return "Screenplay-style dialogue format with inline interjections"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]
    
    def write(self, turns: TranscriptFlow, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """
        Write transcript to dialogue script format.
        
        Args:
            turns: TranscriptFlow object
            output_path: Path to write the output file
            word_segments: Optional word segments (not used for this format)
            **kwargs: Additional options
        """
        write_dialogue_script(turns, output_path)


# Register the writer
registry.register_output_writer(DialogueScriptWriter())
