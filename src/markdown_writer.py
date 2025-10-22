# src/markdown_writer.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path


def write_conversation_markdown(turns: List[Dict], path: str | Path) -> None:
    """
    Write a two-column markdown table with Interviewer and Participant columns.
    Each turn gets its own row, with the other column empty.
    Consecutive turns from the same speaker are merged into one cell.
    
    Args:
        turns: List of turn dictionaries with 'speaker' and 'text' keys
        path: Output file path for the markdown file
    """
    path = Path(path)
    
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