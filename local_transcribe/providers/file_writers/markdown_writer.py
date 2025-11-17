#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn, registry, WordSegment


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

    def write(self, turns: List[Turn], output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Convert Turn to dict for compatibility
        turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
        write_conversation_markdown(turn_dicts, output_path)


# Register the writer
registry.register_output_writer(MarkdownWriter())
