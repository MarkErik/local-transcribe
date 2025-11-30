#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
from local_transcribe.framework.plugin_interfaces import WordSegment


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


def _fmt_ts(t: float) -> str:
    """Format seconds -> 00:00:00.000 for human-readable timestamps."""
    if t < 0:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_timestamped_txt(turns: List[Dict], path: str | Path) -> None:
    """Write a text file with [timestamp] Speaker: text format."""
    path = Path(path)
    lines: list[str] = []
    for t in turns:
        ts = _fmt_ts(float(t["start"]))
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "").strip()
        lines.append(f"[{ts}] {speaker}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plain_txt(turns: List[Dict], path: str | Path) -> None:
    """Write a plain transcript without timestamps, merging consecutive segments from the same speaker into paragraphs."""
    path = Path(path)
    lines: list[str] = []
    
    if not turns:
        path.write_text("", encoding="utf-8")
        return

    current_speaker = turns[0].get("speaker", "Unknown")
    current_paragraph_text = []

    for t in turns:
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "").strip()

        if speaker == current_speaker:
            current_paragraph_text.append(text)
        else:
            # Speaker changed, finalize the previous speaker's paragraph
            if current_paragraph_text:  # Ensure there's text to write
                lines.append(f"{current_speaker}: {' '.join(current_paragraph_text)}")
            
            # Start a new paragraph for the new speaker
            current_speaker = speaker
            current_paragraph_text = [text]
    
    # After the loop, add the last speaker's paragraph
    if current_paragraph_text:
        lines.append(f"{current_speaker}: {' '.join(current_paragraph_text)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_word_segments(words: List[Union[WordSegment, Dict]], path: str | Path) -> None:
    """Write word-level results as plain text without timestamps or speaker labels."""
    path = Path(path)
    lines: list[str] = []
    for w in words:
        # Handle both dict and WordSegment
        if hasattr(w, 'text'):
            text = w.text.strip()
        else:
            text = w.get("text", "").strip()
        if text:  # Only add non-empty text
            lines.append(text)
    path.write_text(" ".join(lines) + "\n", encoding="utf-8")


# Plugin classes
from local_transcribe.framework.plugin_interfaces import OutputWriter, WordWriter, Turn, registry, WordSegment
from typing import List


class WordSegmentsWriter(WordWriter):
    @property
    def name(self) -> str:
        return "word-segments"

    @property
    def description(self) -> str:
        return "Raw word output without timestamps or speakers"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(self, words: List[WordSegment], output_path: str) -> None:
        write_word_segments(words, output_path)


class TimestampedTextWriter(OutputWriter):
    @property
    def name(self) -> str:
        return "timestamped-txt"

    @property
    def description(self) -> str:
        return "Timestamped text format with speaker labels"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(self, turns: Any, output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Extract turns as dicts from any supported format
        turn_dicts = _extract_turns_as_dicts(turns)
        write_timestamped_txt(turn_dicts, output_path)


class PlainTextWriter(OutputWriter):
    @property
    def name(self) -> str:
        return "plain-txt"

    @property
    def description(self) -> str:
        return "Plain text format without timestamps"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(self, turns: Any, output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Extract turns as dicts from any supported format
        turn_dicts = _extract_turns_as_dicts(turns)
        write_plain_txt(turn_dicts, output_path)


# Register the file_writers
registry.register_output_writer(TimestampedTextWriter())
registry.register_output_writer(PlainTextWriter())
registry.register_word_writer(WordSegmentsWriter())
