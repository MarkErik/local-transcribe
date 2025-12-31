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
- CombinedSegment: Combined segment for ASR processing
- SegmentCombinationConfig: Configuration for intelligent segmentation
- ASRChunk: Audio chunk prepared for ASR

- segment_for_asr: Stateless function for intelligent segment combination/splitting
- VADSegmenter: Convenience class for intelligent segmentation
- VADBlockBuilder: Builds conversation blocks from per-speaker VAD segments
- VADASRProcessor: Processes VAD blocks through ASR with chunking

- SileroVADProcessor: Legacy wrapper for Silero VAD (use providers.vad.SileroVADProvider instead)
"""

# Import from unified types module
from local_transcribe.processing.vad.types import (
    VADSegment,
    VADBlock,
    VADBlockBuilderConfig,
    ASRChunk,
    CombinedSegment,
    SegmentCombinationConfig,
)

# Import segmenter
from local_transcribe.processing.vad.segmenter import (
    segment_for_asr,
    VADSegmenter,
)

# Import block builder
from local_transcribe.processing.vad.vad_block_builder import VADBlockBuilder

# Import ASR processor
from local_transcribe.processing.vad.vad_asr_processor import VADASRProcessor

# Import legacy processor for backward compatibility
from local_transcribe.processing.vad.silero_vad import SileroVADProcessor

# Import audit utilities
from local_transcribe.processing.vad.vad_audit import (
    write_vad_audit,
    write_turn_building_audit,
    write_asr_chunks_audit,
)


__all__ = [
    # Types
    'VADSegment',
    'VADBlock',
    'VADBlockBuilderConfig',
    'ASRChunk',
    'CombinedSegment',
    'SegmentCombinationConfig',
    # Segmentation
    'segment_for_asr',
    'VADSegmenter',
    # Processors
    'SileroVADProcessor',  # Legacy - use providers.vad.SileroVADProvider
    'VADBlockBuilder',
    'VADASRProcessor',
    # Audit utilities
    'write_vad_audit',
    'write_turn_building_audit',
    'write_asr_chunks_audit',
]

