#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
import json

from local_transcribe.framework.plugin_interfaces import OutputWriter, WordWriter, Turn, registry, WordSegment


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


def write_word_segments_json(words: List[Union[WordSegment, Dict]], path: str | Path) -> None:
    """Write word-level results as JSON with detailed timing information."""
    path = Path(path)
    
    # Convert WordSegment objects to dictionaries for JSON serialization
    word_data = []
    for w in words:
        if hasattr(w, 'text'):
            # WordSegment object
            word_dict = {
                "text": w.text,
                "start": w.start,
                "end": w.end,
                "speaker": w.speaker
            }
        else:
            # Dictionary
            word_dict = {
                "text": w.get("text", ""),
                "start": w.get("start", 0),
                "end": w.get("end", 0),
                "speaker": w.get("speaker")
            }
        
        word_data.append(word_dict)
    
    # Create the JSON structure
    json_data = {
        "metadata": {
            "total_words": len(word_data),
            "format_version": "1.0"
        },
        "words": word_data
    }
    
    path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_turns_json(transcript: Any, path: str | Path) -> None:
    """Write turn-level results as JSON with timing and speaker information.
    
    Handles TranscriptFlow (with full hierarchical structure) or legacy Turn lists.
    """
    path = Path(path)
    
    # Check if this is a TranscriptFlow with full structure
    if hasattr(transcript, 'to_dict') and hasattr(transcript, 'metadata'):
        # TranscriptFlow - write full hierarchical structure
        json_data = transcript.to_dict()
        path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    
    # Legacy format - extract turns as dicts
    turns = _extract_turns_as_dicts(transcript)
    
    # Convert to turn data format
    turn_data = []
    for t in turns:
        turn_dict = {
            "speaker": t.get("speaker", "Unknown"),
            "start": t.get("start", 0),
            "end": t.get("end", 0),
            "text": t.get("text", ""),
            "word_count": len(t.get("text", "").split()) if t.get("text") else 0
        }
        turn_data.append(turn_dict)
    
    # Create the JSON structure
    json_data = {
        "metadata": {
            "total_turns": len(turn_data),
            "total_words": sum(t["word_count"] for t in turn_data),
            "speakers": list(set(t["speaker"] for t in turn_data if t["speaker"])),
            "format_version": "1.0"
        },
        "turns": turn_data
    }
    
    path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")


# Plugin classes
class WordSegmentsJsonWriter(WordWriter):
    @property
    def name(self) -> str:
        return "word-segments-json"

    @property
    def description(self) -> str:
        return "Word-level output with detailed timing information in JSON format"

    @property
    def supported_formats(self) -> List[str]:
        return [".json"]

    def write(self, words: List[WordSegment], output_path: str) -> None:
        write_word_segments_json(words, output_path)


class TurnsJsonWriter(OutputWriter):
    @property
    def name(self) -> str:
        return "turns-json"

    @property
    def description(self) -> str:
        return "Turn-level output with speaker and timing information in JSON format"

    @property
    def supported_formats(self) -> List[str]:
        return [".json"]

    def write(self, turns: Any, output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Handles both TranscriptFlow and legacy Turn lists
        write_turns_json(turns, output_path)


# Register the file_writers
registry.register_word_writer(WordSegmentsJsonWriter())
registry.register_output_writer(TurnsJsonWriter())