#!/usr/bin/env python3
"""
Transcript Comparison Tool - Entry Point

Compare and analyze differences between transcripts from different 
transcription methods.

Usage:
    uv run python compare_transcripts.py compare file_a.json file_b.json
    uv run python compare_transcripts.py compare file_a.json file_b.json --detailed
    uv run python compare_transcripts.py web
    uv run python compare_transcripts.py web --file-a file_a.json --file-b file_b.json --audio audio.m4a
"""

import sys
from tools.transcript_compare.cli import main

if __name__ == "__main__":
    sys.exit(main())
