"""
Extract words from different transcript JSON formats.

Supported formats:
1. word-level: {"metadata": {...}, "words": [{"text": "word", "start": 0.0, "end": 0.1, "speaker": "..."}, ...]}
2. chunk-based: [{"chunk_id": 1, "words": ["word1", "word2", ...]}, ...]
3. segment-based: {"segments": [{"text": "...", "words": ["word1", ...], "start_s": 0.0, ...}, ...]}
4. script-based: .script.txt files with speaker turns and timestamps
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedTranscript:
    """Container for extracted transcript data."""
    words: list[str]
    source_file: str
    format_type: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.words)
    
    @property
    def text(self) -> str:
        return " ".join(self.words)


def detect_format(data: dict | list) -> str:
    """Detect the JSON format type."""
    if isinstance(data, list):
        # Check if it's chunk-based format
        if data and isinstance(data[0], dict) and "chunk_id" in data[0]:
            return "chunk-based"
        return "unknown"
    
    if isinstance(data, dict):
        # Check for word-level format (granite-silero-mfa style)
        if "words" in data and "metadata" in data:
            if data["words"] and isinstance(data["words"][0], dict) and "text" in data["words"][0]:
                return "word-level"
        
        # Check for segment-based format (vad-granite style)
        if "segments" in data:
            if data["segments"] and isinstance(data["segments"][0], dict) and "words" in data["segments"][0]:
                return "segment-based"
        
        # Check for simple word-level format without metadata
        if "words" in data and isinstance(data["words"], list):
            if data["words"] and isinstance(data["words"][0], dict) and "text" in data["words"][0]:
                return "word-level"
    
    return "unknown"


def extract_words_word_level(data: dict) -> tuple[list[str], dict]:
    """Extract words from word-level format."""
    words = [w["text"].lower().strip() for w in data["words"] if w.get("text", "").strip()]
    metadata = data.get("metadata", {})
    return words, metadata


def extract_words_chunk_based(data: list) -> tuple[list[str], dict]:
    """Extract words from chunk-based format."""
    words = []
    for chunk in data:
        chunk_words = chunk.get("words", [])
        for w in chunk_words:
            if isinstance(w, str) and w.strip():
                words.append(w.lower().strip())
            elif isinstance(w, dict) and w.get("text", "").strip():
                words.append(w["text"].lower().strip())
    
    metadata = {"total_chunks": len(data)}
    return words, metadata


def extract_words_segment_based(data: dict) -> tuple[list[str], dict]:
    """Extract words from segment-based format."""
    words = []
    for segment in data.get("segments", []):
        segment_words = segment.get("words", [])
        for w in segment_words:
            if isinstance(w, str) and w.strip():
                words.append(w.lower().strip())
            elif isinstance(w, dict) and w.get("text", "").strip():
                words.append(w["text"].lower().strip())
    
    metadata = {
        "total_segments": data.get("total_segments", len(data.get("segments", []))),
        "speaker_id": data.get("speaker_id"),
        "audio_file": data.get("audio_file"),
    }
    return words, metadata


def extract_from_file(filepath: str | Path) -> ExtractedTranscript:
    """Extract words from a transcript JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not filepath.suffix.lower() == ".json":
        raise ValueError(f"Expected JSON file, got: {filepath.suffix}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    format_type = detect_format(data)
    
    if format_type == "word-level":
        words, metadata = extract_words_word_level(data)
    elif format_type == "chunk-based":
        words, metadata = extract_words_chunk_based(data)
    elif format_type == "segment-based":
        words, metadata = extract_words_segment_based(data)
    else:
        raise ValueError(f"Unknown JSON format in file: {filepath}")
    
    return ExtractedTranscript(
        words=words,
        source_file=str(filepath),
        format_type=format_type,
        metadata=metadata,
    )


def extract_from_text(text: str, source_name: str = "text") -> ExtractedTranscript:
    """Extract words from plain text."""
    # Simple tokenization - split on whitespace and clean up
    words = []
    for word in text.split():
        # Remove punctuation from ends
        cleaned = word.lower().strip(".,!?;:\"'()[]{}…-–—")
        if cleaned:
            words.append(cleaned)
    
    return ExtractedTranscript(
        words=words,
        source_file=source_name,
        format_type="plain-text",
        metadata={},
    )


@dataclass
class ScriptTurn:
    """A single turn in a script transcript."""
    speaker: str
    timestamp: float
    text: str
    interjections: list[tuple[str, float, str]] = field(default_factory=list)
    
    @property
    def words(self) -> list[str]:
        """Extract words from the turn text."""
        result = []
        for word in self.text.split():
            cleaned = word.lower().strip(".,!?;:\"'()[]{}…-–—")
            if cleaned:
                result.append(cleaned)
        return result


@dataclass
class ExtractedScript:
    """Container for extracted script transcript data."""
    turns: list[ScriptTurn]
    source_file: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def total_turns(self) -> int:
        return len(self.turns)
    
    @property
    def total_words(self) -> int:
        return sum(len(t.words) for t in self.turns)
    
    @property
    def speakers(self) -> list[str]:
        return list(set(t.speaker for t in self.turns))
    
    @property
    def all_words(self) -> list[str]:
        """Get all words from all turns in order."""
        words = []
        for turn in self.turns:
            words.extend(turn.words)
        return words
    
    @property
    def full_text(self) -> str:
        """Get all text from all turns."""
        return " ".join(t.text for t in self.turns)


def extract_script_from_file(filepath: str | Path) -> ExtractedScript:
    """Extract turns from a .script.txt file."""
    import re
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    return extract_script_from_text(content, str(filepath))


def extract_script_from_text(content: str, source_name: str = "text") -> ExtractedScript:
    """Extract turns from script-formatted text."""
    import re
    
    lines = content.split("\n")
    turns = []
    metadata = {}
    
    # Parse header for metadata
    for i, line in enumerate(lines[:10]):
        if line.startswith("Duration:"):
            # Extract duration, turns, speakers from header
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Duration:"):
                    metadata["duration"] = part.replace("Duration:", "").strip()
                elif part.startswith("Turns:"):
                    metadata["total_turns"] = part.replace("Turns:", "").strip()
                elif part.startswith("Speakers:"):
                    metadata["speakers"] = part.replace("Speakers:", "").strip()
    
    # Find turn blocks (separated by dashed lines)
    turn_pattern = re.compile(
        r'^([A-Z][A-Z0-9_\s]+):\s*\n'  # Speaker name
        r'\s*\((\d+\.?\d*)s\)\s*'       # Timestamp
        r'(.+?)(?=\n-{3,}|\n[A-Z][A-Z0-9_\s]+:\s*\n|\Z)',  # Content until next separator
        re.MULTILINE | re.DOTALL
    )
    
    # Interjection pattern (inline speaker notes)
    interjection_pattern = re.compile(
        r'\[([A-Z][A-Z0-9_]+):\s*\((\d+\.?\d*)s\)\s*([^\]]*)\]'
    )
    
    for match in turn_pattern.finditer(content):
        speaker = match.group(1).strip()
        timestamp = float(match.group(2))
        raw_text = match.group(3).strip()
        
        # Extract interjections from the text
        interjections = []
        for interj_match in interjection_pattern.finditer(raw_text):
            interj_speaker = interj_match.group(1)
            interj_time = float(interj_match.group(2))
            interj_text = interj_match.group(3).strip()
            interjections.append((interj_speaker, interj_time, interj_text))
        
        # Clean up the text (remove interjections, normalize whitespace)
        clean_text = interjection_pattern.sub("", raw_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Remove tab characters that are often at the start of continuation lines
        clean_text = clean_text.replace("\t", " ")
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if clean_text:  # Only add turns with actual content
            turns.append(ScriptTurn(
                speaker=speaker,
                timestamp=timestamp,
                text=clean_text,
                interjections=interjections,
            ))
    
    return ExtractedScript(
        turns=turns,
        source_file=source_name,
        metadata=metadata,
    )


def is_script_format(filepath: str | Path) -> bool:
    """Check if a file is in script format."""
    filepath = Path(filepath)
    
    # Check file extension
    if filepath.suffix.lower() in ['.txt']:
        if 'script' in filepath.stem.lower():
            return True
        # Check content for script format markers
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.read(500)
            return 'CONVERSATION TRANSCRIPT' in header or bool(re.search(r'^[A-Z]+:\s*\n\s*\(\d+\.?\d*s\)', header, re.MULTILINE))
        except Exception:
            pass
    return False
