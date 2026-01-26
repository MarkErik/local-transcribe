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
- SegmentCombinationConfig: Configuration for segmentation
- ASRChunk: Audio chunk prepared for ASR

Segmentation (see segmenter.py, segment_combiner.py, segment_splitter.py):
- segment_for_asr: Main entry point for segment combination/splitting
- VADSegmenter: Convenience class for segmentation
- combine_segments: Combine nearby VAD segments based on gap analysis
- split_long_segments: Split long segments at natural boundaries

- VADBlockBuilder: Builds conversation blocks from per-speaker VAD segments
- VADASRProcessor: Processes VAD blocks through ASR with chunking

For VAD detection, use providers.vad.SileroVADProvider.
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

# Import segmenter (orchestrates combination and splitting)
from local_transcribe.processing.vad.segmenter import (
    segment_for_asr,
    VADSegmenter,
)

# Import segment combiner
from local_transcribe.processing.vad.segment_combiner import (
    combine_segments,
    should_combine_segments,
)

# Import segment splitter
from local_transcribe.processing.vad.segment_splitter import (
    split_long_segments,
    split_long_segments_recursive,
    find_best_split_points,
    find_force_split_points,
    calculate_split_score,
)

# Import block builder
from local_transcribe.processing.vad.vad_block_builder import VADBlockBuilder

# Import ASR processor
from local_transcribe.processing.vad.vad_asr_processor import VADASRProcessor

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
    # Segmentation (main entry point)
    'segment_for_asr',
    'VADSegmenter',
    # Segment combination
    'combine_segments',
    'should_combine_segments',
    # Segment splitting
    'split_long_segments',
    'split_long_segments_recursive',
    'find_best_split_points',
    'find_force_split_points',
    'calculate_split_score',
    # Processors
    'VADBlockBuilder',
    'VADASRProcessor',
    # Audit utilities
    'write_vad_audit',
    'write_turn_building_audit',
    'write_asr_chunks_audit',
]
