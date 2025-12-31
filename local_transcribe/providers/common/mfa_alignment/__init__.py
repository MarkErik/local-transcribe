"""
MFA Alignment Module

A modular package for Montreal Forced Aligner (MFA) word alignment operations.

This package provides:
- TextGrid parsing for MFA output files
- Word similarity calculations for alignment
- Dynamic programming sequence alignment
- Word text replacement with original transcripts
- Main alignment engine that orchestrates all operations

Usage:
    from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine
    
    engine = MFAAlignmentEngine(logger)
    word_dicts = engine.parse_textgrid_to_word_dicts(textgrid_path, transcript, start_time, end_time, speaker)
"""

from local_transcribe.providers.common.mfa_alignment.alignment_engine import MFAAlignmentEngine
from local_transcribe.providers.common.mfa_alignment.textgrid_parser import TextGridParser
from local_transcribe.providers.common.mfa_alignment.word_similarity import WordSimilarity
from local_transcribe.providers.common.mfa_alignment.sequence_alignment import SequenceAligner
from local_transcribe.providers.common.mfa_alignment.word_replacement import WordReplacer
from local_transcribe.providers.common.mfa_alignment.fallback_alignment import FallbackAligner

__all__ = [
    'MFAAlignmentEngine',
    'TextGridParser',
    'WordSimilarity',
    'SequenceAligner',
    'WordReplacer',
    'FallbackAligner',
]
