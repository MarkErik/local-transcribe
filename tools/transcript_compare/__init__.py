"""
Transcript Comparison Tool

A tool to analyze and report differences between transcripts from different
transcription methods/settings.

Usage (CLI):
    uv run python -m tools.transcript_compare.cli compare file_a.json file_b.json
    uv run python -m tools.transcript_compare.cli compare file_a.json file_b.json --detailed
    uv run python -m tools.transcript_compare.cli web
    uv run python -m tools.transcript_compare.cli web --file-a file_a.json --file-b file_b.json

Usage (Python):
    from tools.transcript_compare.extractor import extract_from_file
    from tools.transcript_compare.diff_engine import compute_diff
    
    transcript_a = extract_from_file("file_a.json")
    transcript_b = extract_from_file("file_b.json")
    result = compute_diff(transcript_a.words, transcript_b.words)
    print(f"Similarity: {result.similarity_ratio * 100:.1f}%")
"""

from .extractor import extract_from_file, extract_from_text, ExtractedTranscript
from .diff_engine import compute_diff, DiffResult, DiffType, DiffSegment

__all__ = [
    "extract_from_file",
    "extract_from_text",
    "ExtractedTranscript",
    "compute_diff",
    "DiffResult",
    "DiffType",
    "DiffSegment",
]
