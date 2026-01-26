#!/usr/bin/env python3
"""
VAD segment combination and splitting for ASR.

This module provides the main entry point for VAD segmentation, combining
the segment_combiner and segment_splitter modules. It orchestrates the
combination and splitting of raw VAD segments into ASR-ready chunks.

For implementation details, see:
- segment_combiner.py: Functions for combining nearby VAD segments
- segment_splitter.py: Functions for splitting long segments at natural boundaries
"""

from typing import List, Optional
import logging

from local_transcribe.processing.vad.types import (
    VADSegment,
    CombinedSegment,
    SegmentCombinationConfig,
)
from local_transcribe.processing.vad.segment_combiner import combine_segments
from local_transcribe.processing.vad.segment_splitter import (
    split_long_segments,
    split_long_segments_recursive,
)
from local_transcribe.lib.program_logger import get_logger

logger = logging.getLogger(__name__)


def segment_for_asr(
    raw_segments: List[VADSegment],
    max_segment_duration: float = 30.0,
    config: Optional[SegmentCombinationConfig] = None
) -> List[CombinedSegment]:
    """Combine and split raw VAD segments into ASR-ready chunks.
    
    Uses pause analysis to split at natural boundaries.
    This is the main entry point for the segmentation logic.
    
    Args:
        raw_segments: Raw VAD segments from provider
        max_segment_duration: Maximum chunk duration (from transcriber capability)
        config: Optional custom configuration (uses interview defaults if None)
        
    Returns:
        List of CombinedSegment ready for ASR processing
    """
    if not raw_segments:
        return []
    
    # Create config if not provided
    if config is None:
        config = SegmentCombinationConfig(max_segment_duration=max_segment_duration)
    else:
        config.max_segment_duration = max_segment_duration
    
    # Step 1: Initial combination based on gap analysis
    combined = combine_segments(raw_segments, config)
    
    # Step 2: Split long segments at natural boundaries
    final = split_long_segments(combined, config)
    
    # Step 3: Second pass for remaining long segments
    final = split_long_segments_recursive(final, config)
    
    logger.info(f"Final result: {len(final)} segments after combination and splitting")
    
    return final


# Convenience class for stateful usage (optional)
class VADSegmenter:
    """Convenience class for VAD segmentation with stored configuration.
    
    This class wraps the stateless functions for easier use when the
    same configuration is used multiple times.
    """
    
    def __init__(
        self,
        max_segment_duration: float = 30.0,
        config: Optional[SegmentCombinationConfig] = None
    ):
        """Initialize the segmenter.
        
        Args:
            max_segment_duration: Maximum chunk duration (from transcriber capability)
            config: Optional custom configuration (uses interview defaults if None)
        """
        self.logger = get_logger()
        
        if config is None:
            self.config = SegmentCombinationConfig(
                max_segment_duration=max_segment_duration
            )
        else:
            self.config = config
            self.config.max_segment_duration = max_segment_duration
    
    def segment_for_asr(
        self,
        raw_segments: List[VADSegment]
    ) -> List[CombinedSegment]:
        """Combine and split raw VAD segments into ASR-ready chunks.
        
        Args:
            raw_segments: Raw VAD segments from provider
            
        Returns:
            List of CombinedSegment ready for ASR processing
        """
        return segment_for_asr(
            raw_segments,
            max_segment_duration=self.config.max_segment_duration,
            config=self.config
        )
