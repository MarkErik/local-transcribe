#!/usr/bin/env python3
"""
Annotated Markdown Writer for hierarchical transcripts.

This writer produces rich Markdown output that preserves the hierarchical
structure of TranscriptFlow, showing interjections inline within turns
and including conversation metrics.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pathlib import Path

from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow
from local_transcribe.providers.file_writers.format_utils import (
    format_timestamp,
    format_duration,
    format_speaker_name,
    get_interjection_verb,
    format_percentage,
    wrap_text
)


def write_annotated_markdown(transcript: TranscriptFlow, path: str | Path) -> None:
    """
    Write a TranscriptFlow as annotated Markdown with interjections inline.
    
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
    metadata = transcript.metadata
    conversation_metrics = getattr(transcript, 'conversation_metrics', {})
    speaker_statistics = getattr(transcript, 'speaker_statistics', {})
    
    # Header
    lines.append("# Conversation Transcript")
    lines.append("")
    
    # Metadata section
    lines.append("## Summary")
    lines.append("")
    
    duration = conversation_metrics.get('total_duration', metadata.get('duration', 0))
    lines.append(f"- **Duration:** {format_duration(duration)}")
    lines.append(f"- **Total Turns:** {conversation_metrics.get('total_turns', len(turns))}")
    lines.append(f"- **Total Interjections:** {conversation_metrics.get('total_interjections', 0)}")
    
    speakers = metadata.get('speakers', [])
    if speakers:
        speaker_str = ", ".join(format_speaker_name(s) for s in speakers)
        lines.append(f"- **Speakers:** {speaker_str}")
    
    # Speaker statistics
    if speaker_statistics:
        lines.append("")
        lines.append("### Speaker Statistics")
        lines.append("")
        lines.append("| Speaker | Turns | Words | Avg Duration | Avg WPM |")
        lines.append("|---------|-------|-------|--------------|---------|")
        
        for speaker, stats in speaker_statistics.items():
            turns_count = stats.get('total_turns', 0)
            words_count = stats.get('total_words', 0)
            avg_duration = stats.get('avg_turn_duration', 0)
            avg_wpm = stats.get('avg_speaking_rate', 0)
            
            lines.append(
                f"| {format_speaker_name(speaker)} | {turns_count} | {words_count} | "
                f"{avg_duration:.1f}s | {avg_wpm:.0f} |"
            )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    
    # Process each turn
    for turn in turns:
        turn_id = getattr(turn, 'turn_id', 0)
        speaker = getattr(turn, 'primary_speaker', 'Unknown')
        start = getattr(turn, 'start', 0)
        end = getattr(turn, 'end', 0)
        text = getattr(turn, 'text', '')
        interjections = getattr(turn, 'interjections', [])
        flow_continuity = getattr(turn, 'flow_continuity', 1.0)
        turn_type = getattr(turn, 'turn_type', 'monologue')
        word_count = getattr(turn, 'word_count', 0)
        speaking_rate = getattr(turn, 'speaking_rate', 0)
        
        # Turn header
        lines.append(f"### Turn {turn_id} ({format_timestamp(start, 'seconds')} - {format_timestamp(end, 'seconds')})")
        lines.append("")
        
        # Speaker and text
        # For turns with interjections, we need to show them inline
        if interjections:
            lines.append(f"**{format_speaker_name(speaker)}:** {text}")
            lines.append("")
            
            # Sort interjections by start time
            sorted_interjections = sorted(interjections, key=lambda x: getattr(x, 'start', 0))
            
            for ij in sorted_interjections:
                ij_speaker = getattr(ij, 'speaker', 'Unknown')
                ij_start = getattr(ij, 'start', 0)
                ij_text = getattr(ij, 'text', '')
                ij_type = getattr(ij, 'interjection_type', 'unclear')
                
                verb = get_interjection_verb(ij_type)
                lines.append(f"  → *[{format_timestamp(ij_start, 'seconds')}] {format_speaker_name(ij_speaker)} {verb}: \"{ij_text}\"*")
            
            lines.append("")
        else:
            lines.append(f"**{format_speaker_name(speaker)}:** {text}")
            lines.append("")
        
        # Turn metadata footer
        flow_str = format_percentage(flow_continuity)
        lines.append(f"*Flow: {flow_str} continuous | Type: {turn_type} | {word_count} words @ {speaking_rate:.1f} wpm*")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Write to file
    path.write_text("\n".join(lines), encoding="utf-8")


class AnnotatedMarkdownWriter(OutputWriter):
    """Writer for annotated Markdown format with hierarchical structure."""
    
    @property
    def name(self) -> str:
        return "markdown"
    
    @property
    def description(self) -> str:
        return "Rich Markdown format with inline interjections and conversation metrics"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".md"]
    
    def write(self, turns: TranscriptFlow, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """
        Write transcript to annotated Markdown format.
        
        Args:
            turns: TranscriptFlow object with hierarchical turn structure
            output_path: Path to write the output file
            word_segments: Optional word segments (not used for this format)
            **kwargs: Additional options
        """
        write_annotated_markdown(turns, output_path)


# Register the writer (replaces old markdown writer)
registry.register_output_writer(AnnotatedMarkdownWriter())
