#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import csv


def write_conversation_csv(turns: List[Dict], path: str | Path) -> None:
    """
    Write a two-column CSV with Interviewer and Participant columns.
    Each turn gets its own row, with the other column empty.
    Consecutive turns from the same speaker are merged into one cell.
    
    Args:
        turns: List of turn dictionaries with 'speaker' and 'text' keys
        path: Output file path for the CSV
    """
    path = Path(path)
    
    # Group consecutive turns by speaker
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
    
    # Write CSV
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Interviewer', 'Participant'])
        
        # Write data rows
        for turn in grouped_turns:
            speaker = turn["speaker"]
            text = turn["text"]
            
            if speaker.lower() == "interviewer":
                writer.writerow([text, ""])
            elif speaker.lower() == "participant":
                writer.writerow(["", text])
            else:
                # Handle unexpected speaker names
                writer.writerow([text, ""])  # Default to interviewer column
