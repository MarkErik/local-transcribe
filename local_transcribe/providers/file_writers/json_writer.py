#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Union, Any, Optional
from pathlib import Path
import json

from local_transcribe.framework.plugin_interfaces import OutputWriter, WordWriter, Turn, registry, WordSegment


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
            "total_duration": max((w["end"] for w in word_data if w["end"]), default=0),
            "format_version": "1.0"
        },
        "words": word_data
    }
    
    path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_turns_json(turns: List[Union[Turn, Dict]], path: str | Path) -> None:
    """Write turn-level results as JSON with timing and speaker information."""
    path = Path(path)
    
    # Convert Turn objects to dictionaries for JSON serialization
    turn_data = []
    for t in turns:
        if hasattr(t, 'speaker'):
            # Turn object
            turn_dict = {
                "speaker": t.speaker,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "word_count": len(t.text.split()) if t.text else 0
            }
        else:
            # Dictionary
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
            "total_duration": max((t["end"] for t in turn_data if t["end"]), default=0),
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

    def write(self, turns: List[Turn], output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Convert Turn to dict for compatibility
        turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
        write_turns_json(turn_dicts, output_path)


# Register the file_writers
registry.register_word_writer(WordSegmentsJsonWriter())
registry.register_output_writer(TurnsJsonWriter())