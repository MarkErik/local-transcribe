# src/csv_writer.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import csv
import logging

from config import is_debug_enabled, is_info_enabled

logger = logging.getLogger(__name__)


def write_conversation_csv(turns: List[Dict], path: str | Path, include_cross_talk: bool = False) -> None:
    """
    Write a CSV with Interviewer and Participant columns.
    Each turn gets its own row, with the other column empty.
    Consecutive turns from the same speaker are merged into one cell.
    
    Args:
        turns: List of turn dictionaries with 'speaker' and 'text' keys,
               optionally including 'cross_talk_present' and 'confidence'
        path: Output file path for the CSV
        include_cross_talk: Whether to include cross-talk columns in output.
                           When False (default), maintains original CSV format for
                           backward compatibility. When True, adds 'cross_talk'
                           and 'confidence' columns.
    """
    path = Path(path)
    
    try:
        if is_info_enabled():
            logger.info(f"Writing CSV to {path} with cross-talk columns: {include_cross_talk}")
        
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
            if include_cross_talk:
                writer.writerow(['Interviewer', 'Participant', 'cross_talk', 'confidence'])
            else:
                writer.writerow(['Interviewer', 'Participant'])
            
            # Write data rows
            for turn in grouped_turns:
                speaker = turn["speaker"]
                text = turn["text"]
                
                # Get cross-talk information if available
                cross_talk_value = ""
                confidence_value = ""
                
                if include_cross_talk:
                    try:
                        # Check if turn has cross-talk information
                        cross_talk_present = turn.get("cross_talk_present", False)
                        confidence = turn.get("confidence", 1.0)
                        
                        cross_talk_value = str(cross_talk_present)
                        confidence_value = f"{confidence:.3f}"  # Format to 3 decimal places
                        
                    except Exception as e:
                        if is_info_enabled():
                            logger.warning(f"Error processing cross-talk data for turn: {e}")
                        # Use default values if there's an error
                        cross_talk_value = "False"
                        confidence_value = "1.000"
                
                if speaker.lower() == "interviewer":
                    if include_cross_talk:
                        writer.writerow([text, "", cross_talk_value, confidence_value])
                    else:
                        writer.writerow([text, ""])
                elif speaker.lower() == "participant":
                    if include_cross_talk:
                        writer.writerow(["", text, cross_talk_value, confidence_value])
                    else:
                        writer.writerow(["", text])
                else:
                    # Handle unexpected speaker names
                    if include_cross_talk:
                        writer.writerow([text, "", cross_talk_value, confidence_value])
                    else:
                        writer.writerow([text, ""])  # Default to interviewer column
            
            if is_info_enabled():
                logger.info(f"Successfully wrote CSV with {len(grouped_turns)} turns to {path}")
            
    except Exception as e:
        if is_info_enabled():
            logger.error(f"Error writing CSV to {path}: {e}")
        raise


# Backward compatibility alias - ensure existing code continues to work
def write_csv(turns: List[Dict], output_path: str, include_cross_talk: bool = False) -> None:
    """
    Write CSV output with optional cross-talk columns.
    
    This is an alias for write_conversation_csv to maintain backward compatibility.
    
    Args:
        turns: List of turn dictionaries with speaker and text information
        output_path: Output file path for the CSV
        include_cross_talk: Whether to include cross-talk columns (default: False)
    """
    write_conversation_csv(turns, output_path, include_cross_talk)