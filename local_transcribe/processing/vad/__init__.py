#!/usr/bin/env python3
"""
VAD (Voice Activity Detection) module for split-audio pipeline.

This module provides VAD-based turn building for processing per-speaker
audio files into conversation transcripts. It uses Silero VAD segments
as authoritative timing and builds turns/blocks by merging nearby VAD segments.

Available components:
- VADSegment: Single VAD-detected speech segment
- VADBlock: Merged block of contiguous VAD segments (a turn)
- VADBlockBuilderConfig: Configuration for merging VAD segments
- VADBlockBuilder: Builds conversation blocks from per-speaker VAD segments
- SileroVADProcessor: Wrapper for Silero VAD
- VADASRProcessor: Processes VAD blocks through ASR with chunking
"""

from local_transcribe.processing.vad.data_structures import (
    VADSegment,
    VADBlock,
    VADBlockBuilderConfig,
    ASRChunk,
)

from local_transcribe.processing.vad.silero_vad import SileroVADProcessor

from local_transcribe.processing.vad.vad_block_builder import VADBlockBuilder

from local_transcribe.processing.vad.vad_asr_processor import VADASRProcessor

from local_transcribe.processing.vad.vad_audit import (
    write_vad_audit,
    write_turn_building_audit,
)


__all__ = [
    # Data structures
    'VADSegment',
    'VADBlock',
    'VADBlockBuilderConfig',
    'ASRChunk',
    # Processors
    'SileroVADProcessor',
    'VADBlockBuilder',
    'VADASRProcessor',
    # Audit utilities
    'write_vad_audit',
    'write_turn_building_audit',
]
