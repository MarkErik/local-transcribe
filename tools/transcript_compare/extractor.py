"""
Extract words from different transcript JSON formats.

Supported formats:
1. word-level: {"metadata": {...}, "words": [{"text": "word", "start": 0.0, "end": 0.1, "speaker": "..."}, ...]}
2. chunk-based: [{"chunk_id": 1, "words": ["word1", "word2", ...]}, ...]
3. segment-based: {"segments": [{"text": "...", "words": ["word1", ...], "start_s": 0.0, ...}, ...]}
"""

import json
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
