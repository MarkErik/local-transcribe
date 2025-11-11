#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict
from pathlib import Path


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


def write_asr_words(words: List[Dict], path: str | Path) -> None:
    """Write ASR word-level results as plain text without timestamps or speaker labels."""
    path = Path(path)
    lines: list[str] = []
    for w in words:
        text = w.get("text", "").strip()
        if text:  # Only add non-empty text
            lines.append(text)
    path.write_text(" ".join(lines) + "\n", encoding="utf-8")


# Plugin classes
from local_transcribe.framework.plugins import OutputWriter, Turn, registry
from typing import List


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

    def write(self, turns: List[Turn], output_path: str) -> None:
        # Convert Turn to dict for compatibility
        turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
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

    def write(self, turns: List[Turn], output_path: str) -> None:
        # Convert Turn to dict for compatibility
        turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
        write_plain_txt(turn_dicts, output_path)


# Register the writers
registry.register_output_writer(TimestampedTextWriter())
registry.register_output_writer(PlainTextWriter())
