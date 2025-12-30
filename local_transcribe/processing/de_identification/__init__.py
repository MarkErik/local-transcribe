#!/usr/bin/env python3
"""
LLM-based de-identification for removing people's names from transcripts.

This package provides a two-pass de-identification system:
1. First pass: General name detection and replacement with [REDACTED]
2. Second pass: Targeted review using names discovered across all speakers

Usage:
    from local_transcribe.processing.de_identification import (
        DeIdentificationOrchestrator,
        de_identify,  # Convenience function
    )
    
    # Simple usage
    result = de_identify(segments, llm_url="http://localhost:8080")
    
    # With orchestrator for more control
    orchestrator = DeIdentificationOrchestrator(
        llm_url="http://localhost:8080",
        intermediate_dir=Path("./output")
    )
    result = orchestrator.de_identify(segments)
    
    # Multi-speaker processing
    results = orchestrator.de_identify_multi_speaker({
        "Interviewer": interviewer_segments,
        "Participant": participant_segments,
    })
"""

# Core data structures
from .core import (
    DeIdentificationConfig,
    DeIdentificationResult,
    WordReplacement,
    DiscoveredName,
    ValidationResult,
    Chunk,
    DEFAULT_CONFIG,
    normalize_text_for_comparison,
    words_equivalent,
    format_timestamp,
)

# Main orchestrator
from .orchestrator import (
    DeIdentificationOrchestrator,
    de_identify,
)

# Individual passes (for advanced usage)
from .first_pass import (
    de_identify_first_pass,
    de_identify_text_first_pass,
    FirstPassResult,
    FIRST_PASS_SYSTEM_PROMPT,
)

from .second_pass import (
    de_identify_second_pass,
    build_global_name_list,
    build_global_name_list_from_dicts,
    SecondPassResult,
    get_second_pass_system_prompt,
)

# LLM client (for advanced usage)
from .llm_client import LLMDeIdentifierClient

# Validation (for advanced usage)
from .validation import (
    validate_first_pass_output,
    validate_second_pass_output,
)

# Chunking (for advanced usage)
from .chunking import (
    chunk_word_segments,
    chunk_plain_text,
    merge_processed_text_chunks,
)

# Audit logging
from .audit import AuditLogger

# Debug output
from .debug import DebugFileWriter


__all__ = [
    # Main API
    'DeIdentificationOrchestrator',
    'de_identify',
    
    # Data structures
    'DeIdentificationConfig',
    'DeIdentificationResult',
    'WordReplacement',
    'DiscoveredName',
    'ValidationResult',
    'Chunk',
    'DEFAULT_CONFIG',
    
    # Utilities
    'normalize_text_for_comparison',
    'words_equivalent',
    'format_timestamp',
    
    # First pass
    'de_identify_first_pass',
    'de_identify_text_first_pass',
    'FirstPassResult',
    'FIRST_PASS_SYSTEM_PROMPT',
    
    # Second pass
    'de_identify_second_pass',
    'build_global_name_list',
    'build_global_name_list_from_dicts',
    'SecondPassResult',
    'get_second_pass_system_prompt',
    
    # LLM client
    'LLMDeIdentifierClient',
    
    # Validation
    'validate_first_pass_output',
    'validate_second_pass_output',
    
    # Chunking
    'chunk_word_segments',
    'chunk_plain_text',
    'merge_processed_text_chunks',
    
    # Audit & Debug
    'AuditLogger',
    'DebugFileWriter',
]
