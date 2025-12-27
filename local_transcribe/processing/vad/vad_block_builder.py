#!/usr/bin/env python3
"""
VAD Block Builder for building conversation blocks from VAD segments.

This module merges nearby VAD segments into blocks/turns, interleaves
blocks from all speakers, and classifies interjections and overlaps.
"""

from typing import List, Dict, Optional
from local_transcribe.processing.vad.data_structures import VADSegment, VADBlock, VADBlockBuilderConfig
from local_transcribe.lib.program_logger import log_progress, log_debug, log_completion


class VADBlockBuilder:
    """
    Builds conversation blocks/turns from per-speaker VAD segments.
    
    The builder performs:
    1. Merging nearby VAD segments per speaker into preliminary blocks
    2. Sorting all blocks by start time
    3. Detecting overlaps between speakers
    4. Classifying interjections vs. floor-taking turns
    """
    
    def __init__(self, config: Optional[VADBlockBuilderConfig] = None):
        """
        Initialize the block builder.
        
        Args:
            config: Configuration for merging and classification thresholds
        """
        self.config = config or VADBlockBuilderConfig()
    
    def build_blocks(
        self,
        all_vad_segments: Dict[str, List[VADSegment]]
    ) -> List[VADBlock]:
        """
        Build blocks from all speakers' VAD segments.
        
        Process:
        1. Merge nearby segments per-speaker into preliminary blocks
        2. Sort all blocks by start time
        3. Detect overlaps between speakers
        4. Classify interjections vs. floor-taking turns
        
        Args:
            all_vad_segments: Mapping of speaker_id to their VAD segments
            
        Returns:
            List of VADBlock sorted by start time
        """
        log_progress("Building conversation blocks from VAD segments")
        
        # 1. Merge segments per speaker
        all_blocks: List[VADBlock] = []
        block_id = 0
        
        for speaker_id, segments in all_vad_segments.items():
            speaker_blocks = self._merge_speaker_segments(segments, speaker_id, start_block_id=block_id)
            all_blocks.extend(speaker_blocks)
            block_id += len(speaker_blocks)
            log_debug(f"Created {len(speaker_blocks)} blocks for {speaker_id}")
        
        # 2. Sort by start time
        all_blocks.sort(key=lambda b: b.start_s)
        
        # Re-assign block IDs after sorting
        for i, block in enumerate(all_blocks):
            block.block_id = i
        
        # 3. Detect overlaps
        self._detect_overlaps(all_blocks)
        
        # 4. Classify interjections
        self._classify_interjections(all_blocks)
        
        # Summary stats
        total_blocks = len(all_blocks)
        interjection_count = sum(1 for b in all_blocks if b.is_interjection)
        overlap_count = sum(1 for b in all_blocks if b.overlap_with)
        
        log_completion(
            f"Block building complete: {total_blocks} blocks, "
            f"{interjection_count} interjections, {overlap_count} with overlaps"
        )
        
        return all_blocks
    
    def _merge_speaker_segments(
        self,
        segments: List[VADSegment],
        speaker_id: str,
        start_block_id: int = 0
    ) -> List[VADBlock]:
        """
        Merge nearby VAD segments for one speaker into blocks.
        
        Segments are merged if the gap between them is less than
        merge_gap_threshold_ms.
        
        Args:
            segments: VAD segments for one speaker
            speaker_id: Speaker identifier
            start_block_id: Starting block ID for numbering
            
        Returns:
            List of VADBlock for this speaker
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_s)
        
        merge_gap_s = self.config.merge_gap_threshold_ms / 1000.0
        
        blocks: List[VADBlock] = []
        current_start = sorted_segments[0].start_s
        current_end = sorted_segments[0].end_s
        current_segment_ids = [sorted_segments[0].segment_id]
        
        for i in range(1, len(sorted_segments)):
            segment = sorted_segments[i]
            gap = segment.start_s - current_end
            
            if gap <= merge_gap_s:
                # Merge with current block
                current_end = max(current_end, segment.end_s)
                current_segment_ids.append(segment.segment_id)
            else:
                # Create block from accumulated segments
                block = VADBlock(
                    block_id=start_block_id + len(blocks),
                    speaker_id=speaker_id,
                    start_s=current_start,
                    end_s=current_end,
                    source_segment_ids=current_segment_ids,
                )
                blocks.append(block)
                
                # Start new accumulation
                current_start = segment.start_s
                current_end = segment.end_s
                current_segment_ids = [segment.segment_id]
        
        # Don't forget the last block
        block = VADBlock(
            block_id=start_block_id + len(blocks),
            speaker_id=speaker_id,
            start_s=current_start,
            end_s=current_end,
            source_segment_ids=current_segment_ids,
        )
        blocks.append(block)
        
        return blocks
    
    def _detect_overlaps(self, blocks: List[VADBlock]) -> None:
        """
        Detect and annotate overlapping blocks.
        
        Two blocks overlap if they have temporal overlap greater than
        overlap_threshold_ms and are from different speakers.
        
        Args:
            blocks: List of blocks sorted by start time (modified in place)
        """
        overlap_threshold_s = self.config.overlap_threshold_ms / 1000.0
        
        for i, block in enumerate(blocks):
            overlapping_ids: List[int] = []
            
            # Check against subsequent blocks that might overlap
            for j in range(i + 1, len(blocks)):
                other = blocks[j]
                
                # If other block starts after this block ends, no more overlaps possible
                if other.start_s >= block.end_s:
                    break
                
                # Only count overlaps with different speakers
                if other.speaker_id == block.speaker_id:
                    continue
                
                # Calculate overlap
                overlap_start = max(block.start_s, other.start_s)
                overlap_end = min(block.end_s, other.end_s)
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration >= overlap_threshold_s:
                    overlapping_ids.append(other.block_id)
                    
                    # Also mark the other block as overlapping with this one
                    if other.overlap_with is None:
                        other.overlap_with = []
                    if block.block_id not in other.overlap_with:
                        other.overlap_with.append(block.block_id)
            
            if overlapping_ids:
                block.overlap_with = overlapping_ids
    
    def _classify_interjections(self, blocks: List[VADBlock]) -> None:
        """
        Classify short overlapping segments as interjections.
        
        A block is classified as an interjection if:
        - It has overlap with another speaker's block
        - Its duration is <= interjection_max_duration_ms
        - It occurs within interjection_window_ms of another speaker's block start
        
        Args:
            blocks: List of blocks (modified in place)
        """
        max_duration_s = self.config.interjection_max_duration_ms / 1000.0
        window_s = self.config.interjection_window_ms / 1000.0
        
        for block in blocks:
            # Only consider blocks that overlap with other speakers
            if not block.overlap_with:
                continue
            
            # Check duration criterion
            if block.duration_s > max_duration_s:
                continue
            
            # Check if this appears to be a backchannel/interjection
            # (short utterance during another speaker's turn)
            for overlapping_id in block.overlap_with:
                other_block = next((b for b in blocks if b.block_id == overlapping_id), None)
                if other_block is None:
                    continue
                
                # If the other block is longer and this block is short,
                # this block is likely an interjection
                if other_block.duration_s > block.duration_s:
                    # Check if this block starts within window of other's start
                    # or if it's a brief backchannel during a longer turn
                    time_since_other_start = block.start_s - other_block.start_s
                    
                    if time_since_other_start >= -window_s:  # Allow small negative (started just before)
                        block.is_interjection = True
                        break
    
    def get_blocks_for_speaker(
        self,
        blocks: List[VADBlock],
        speaker_id: str
    ) -> List[VADBlock]:
        """
        Filter blocks to get only those for a specific speaker.
        
        Args:
            blocks: All blocks
            speaker_id: Speaker to filter by
            
        Returns:
            List of blocks for the specified speaker
        """
        return [b for b in blocks if b.speaker_id == speaker_id]
    
    def get_non_interjection_blocks(
        self,
        blocks: List[VADBlock]
    ) -> List[VADBlock]:
        """
        Get blocks that are not interjections.
        
        Args:
            blocks: All blocks
            
        Returns:
            List of primary (non-interjection) blocks
        """
        return [b for b in blocks if not b.is_interjection]
