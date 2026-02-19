#!/usr/bin/env python3
"""
NVivo Script Writer for hierarchical transcripts.

This writer produces a simplified transcript format suitable for NVivo
qualitative analysis software, with no timestamps, no interjections,
and clean formatting.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pathlib import Path

from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow
from local_transcribe.providers.file_writers.format_utils import format_speaker_name


def write_nvivo_transcript(transcript: TranscriptFlow, path: str | Path) -> None:
    """
    Write a TranscriptFlow as a simplified NVivo-compatible transcript.
    
    Format:
    - Title: INTERVIEW TRANSCRIPT
    - No duration/turns/speakers summary line
    - No timestamps
    - No interjections (completely ignored)
    - No separator dashes between blocks
    - Speaker labels on their own line
    - Text follows on next line without indentation
    - One blank line between conversation blocks
    
    Args:
        transcript: TranscriptFlow object with hierarchical turn structure
        path: Output file path
    """
    path = Path(path)
    lines = []
    
    # Extract data from TranscriptFlow
    if not hasattr(transcript, 'turns') or not hasattr(transcript, 'metadata'):
        raise ValueError("Expected TranscriptFlow object with 'turns' and 'metadata' attributes")
    
    turns = transcript.turns
    
    # Header
    title = "INTERVIEW TRANSCRIPT"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    
    # Process each turn
    for turn in turns:
        speaker = getattr(turn, 'primary_speaker', 'Unknown')
        text = getattr(turn, 'text', '')
        # Interjections are intentionally ignored
        
        # Format speaker name (title case)
        speaker_label = format_speaker_name(speaker)
        
        # Write speaker label
        lines.append(f"{speaker_label}:")
        
        # Write text without indentation, no timestamp
        lines.append(text)
        
        # One blank line between blocks
        lines.append("")
    
    # Write to file (UTF-8 as specified)
    path.write_text("\n".join(lines), encoding="utf-8")


class NVivoScriptWriter(OutputWriter):
    """Writer for NVivo-compatible interview transcript format."""
    
    @property
    def name(self) -> str:
        return "nvivo-script"
    
    @property
    def description(self) -> str:
        return "NVivo-compatible interview transcript format (no timestamps, no interjections)"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]
    
    def write(self, turns: TranscriptFlow, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """
        Write transcript to NVivo script format.
        
        Args:
            turns: TranscriptFlow object
            output_path: Path to write the output file
            word_segments: Optional word segments (not used for this format)
            **kwargs: Additional options
        """
        write_nvivo_transcript(turns, output_path)


# Register the writer
registry.register_output_writer(NVivoScriptWriter())
