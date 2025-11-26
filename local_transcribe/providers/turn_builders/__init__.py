#!/usr/bin/env python3
"""
Turn builder providers.

Available providers:
- combined_audio_turn_builder: For single audio files with diarization
- split_audio_turn_builder: Rule-based turn builder for split audio mode
- split_audio_llm_turn_builder: LLM-enhanced turn builder for split audio mode
"""

# Import data structures for external use
from .split_audio_data_structures import (
    RawSegment,
    InterjectionSegment,
    HierarchicalTurn,
    EnrichedTranscript,
    TurnBuilderConfig
)

# Import all turn builder modules to register them
from . import combined_audio_turn_builder
from . import split_audio_turn_builder
from . import split_audio_llm_turn_builder
