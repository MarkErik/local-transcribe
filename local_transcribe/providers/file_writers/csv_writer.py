#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Optional, Any
from pathlib import Path
import csv
from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow


def _extract_turns_as_dicts(transcript_flow: TranscriptFlow) -> List[Dict]:
    """
    Extract turns as dictionaries from TranscriptFlow.
    
    Returns list of dicts with 'speaker', 'start', 'end', 'text' keys.
    """
    turns = transcript_flow.turns
    result = []
    for t in turns:
        # HierarchicalTurn uses primary_speaker
        speaker = t.primary_speaker
        result.append({
            "speaker": speaker,
            "start": t.start,
            "end": t.end,
            "text": t.text
        })
    return result


def write_conversation_csv(turns: List[Dict], path: str | Path) -> None:
    
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


# Plugin class
class CSVWriter(OutputWriter):
    @property
    def name(self) -> str:
        return "csv"

    @property
    def description(self) -> str:
        return "CSV format with conversation data"

    @property
    def supported_formats(self) -> List[str]:
        return [".csv"]

    def write(self, turns: TranscriptFlow, output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Extract turns as dicts from TranscriptFlow
        turn_dicts = _extract_turns_as_dicts(turns)
        write_conversation_csv(turn_dicts, output_path)


# Register the writer
registry.register_output_writer(CSVWriter())
