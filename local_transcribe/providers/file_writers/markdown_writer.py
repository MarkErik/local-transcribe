#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Optional, Any
from pathlib import Path
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn, registry, WordSegment


def _extract_turns_as_dicts(transcript: Any) -> List[Dict]:
    """
    Extract turns as dictionaries from various transcript formats.
    
    Handles:
    - TranscriptFlow (new hierarchical format)
    - List of Turn objects
    - List of HierarchicalTurn objects
    - List of dictionaries
    
    Returns list of dicts with 'speaker', 'start', 'end', 'text' keys.
    """
    # Handle TranscriptFlow
    if hasattr(transcript, 'turns') and hasattr(transcript, 'metadata'):
        # This is a TranscriptFlow object
        turns = transcript.turns
        result = []
        for t in turns:
            # HierarchicalTurn uses primary_speaker
            speaker = getattr(t, 'primary_speaker', None) or getattr(t, 'speaker', 'Unknown')
            result.append({
                "speaker": speaker,
                "start": t.start,
                "end": t.end,
                "text": t.text
            })
        return result
    
    # Handle list of turns
    if isinstance(transcript, list):
        result = []
        for t in transcript:
            if isinstance(t, dict):
                result.append(t)
            elif hasattr(t, 'primary_speaker'):
                # HierarchicalTurn
                result.append({
                    "speaker": t.primary_speaker,
                    "start": t.start,
                    "end": t.end,
                    "text": t.text
                })
            elif hasattr(t, 'speaker'):
                # Turn object
                result.append({
                    "speaker": t.speaker,
                    "start": t.start,
                    "end": t.end,
                    "text": t.text
                })
            else:
                # Unknown format, try to extract what we can
                result.append({
                    "speaker": str(getattr(t, 'speaker', getattr(t, 'primary_speaker', 'Unknown'))),
                    "start": float(getattr(t, 'start', 0)),
                    "end": float(getattr(t, 'end', 0)),
                    "text": str(getattr(t, 'text', ''))
                })
        return result
    
    # Fallback - return empty list
    return []


def write_conversation_markdown(turns: List[Dict], path: str | Path) -> None:
    
    # Group consecutive turns by speaker (same logic as CSV writer)
    grouped_turns = []
    current_speaker = None
    current_text = ""
    
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "").strip()
        
        if not text:
            continue
            
        if speaker == current_speaker:
            # Same speaker, append to current text
            current_text += " " + text
        else:
            # Different speaker, save previous turn and start new one
            if current_speaker is not None:
                grouped_turns.append({
                    "speaker": current_speaker,
                    "text": current_text.strip()
                })
            current_speaker = speaker
            current_text = text
    
    # Don't forget the last turn
    if current_speaker is not None and current_text:
        grouped_turns.append({
            "speaker": current_speaker,
            "text": current_text.strip()
        })
    
    # Write markdown table
    lines = []
    
    # Add title
    lines.append("# Conversation Transcript")
    lines.append("")
    
    # Add table header
    lines.append("| Interviewer | Participant |")
    lines.append("|-------------|-------------|")
    
    # Add table rows
    for turn in grouped_turns:
        speaker = turn["speaker"]
        text = turn["text"]
        
        # Escape pipe characters in markdown
        text = text.replace("|", "\\|")
        
        if speaker.lower() == "interviewer":
            lines.append(f"| {text} | |")
        elif speaker.lower() == "participant":
            lines.append(f"| | {text} |")
        else:
            # Handle unexpected speaker names
            lines.append(f"| {text} | |")  # Default to interviewer column
    
    # Write to file
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Plugin class
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn, registry
from typing import List


class MarkdownWriter(OutputWriter):
    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Markdown format for conversation"

    @property
    def supported_formats(self) -> List[str]:
        return [".md"]

    def write(self, turns: Any, output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Extract turns as dicts from any supported format
        turn_dicts = _extract_turns_as_dicts(turns)
        write_conversation_markdown(turn_dicts, output_path)


# Register the writer
registry.register_output_writer(MarkdownWriter())
