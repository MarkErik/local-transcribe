#!/usr/bin/env python3
"""
Debug output utilities for chunk stitching.

This module provides functionality for saving debug information during
the chunk stitching process, including input chunks, stitch steps, and output.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from local_transcribe.processing.chunk_stitching.core import (
    ChunkStitcherConfig,
    ChunkInfo,
    OverlapResult,
)
from local_transcribe.processing.chunk_stitching.word_utils import get_word_texts, has_timestamps
from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_debug


class StitcherDebugWriter:
    """
    Handles debug output for the chunk stitching process.
    
    Creates timestamped directories and saves JSON files documenting
    each step of the stitching process.
    """
    
    def __init__(
        self,
        intermediate_dir: Optional[Path],
        config: ChunkStitcherConfig,
        use_timestamped_subdir: bool = True,
    ):
        """
        Initialize the debug writer.
        
        Args:
            intermediate_dir: Base directory for debug output
            config: Stitcher configuration
            use_timestamped_subdir: Whether to create a timestamped subdirectory
        """
        self.config = config
        self.debug_dir: Optional[Path] = None
        
        if intermediate_dir:
            if use_timestamped_subdir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.debug_dir = Path(intermediate_dir) / "chunk_stitching" / "stitcher_debug" / timestamp
            else:
                self.debug_dir = Path(intermediate_dir)
            
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Chunk stitcher debug output: {self.debug_dir}")
    
    @property
    def enabled(self) -> bool:
        """Whether debug output is enabled."""
        return self.debug_dir is not None
    
    def should_save(self, chunk_count: int) -> bool:
        """
        Determine if debug files should be saved.
        
        Args:
            chunk_count: Number of chunks to process
            
        Returns:
            True if debug files should be saved
        """
        if not self.enabled:
            return False
        
        # Skip debug for single chunks if configured
        if self.config.skip_single_chunk_debug and chunk_count <= 1:
            return False
        
        return True
    
    def save_input(self, chunks: List[Dict[str, Any]], words_have_timestamps: bool) -> None:
        """
        Save input chunks for debugging.
        
        Args:
            chunks: List of chunk dictionaries
            words_have_timestamps: Whether words have timestamp information
        """
        if not self.debug_dir:
            return
        
        json_data = {
            'total_chunks': len(chunks),
            'has_timestamps': words_have_timestamps,
            'chunks': []
        }
        
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.get('chunk_id'),
                'word_count': len(chunk['words'])
            }
            if words_have_timestamps and chunk['words']:
                chunk_data['words'] = chunk['words']
                chunk_data['time_range'] = {
                    'start': chunk['words'][0].get('start', 0),
                    'end': chunk['words'][-1].get('end', 0)
                }
            else:
                chunk_data['words'] = chunk['words']
            json_data['chunks'].append(chunk_data)
        
        with open(self.debug_dir / "00_input_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def save_stitch_step(
        self,
        step_num: int,
        chunk1_info: ChunkInfo,
        chunk2_info: ChunkInfo,
        overlap_result: OverlapResult,
        result_word_count: int,
        words_from_chunk1: int,
        words_from_chunk2: int,
        overlapping_words: List[str],
        result_time_range: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save debug info for a single stitching step.
        
        Args:
            step_num: Step number (0-indexed)
            chunk1_info: Info about the first chunk
            chunk2_info: Info about the second chunk
            overlap_result: Result from overlap detection
            result_word_count: Number of words in the result
            words_from_chunk1: Words contributed from chunk1
            words_from_chunk2: Words contributed from chunk2
            overlapping_words: The actual overlapping words
            result_time_range: Time range of result (if timestamps available)
        """
        if not self.debug_dir:
            return
        
        overlap_info = overlap_result.to_dict()
        overlap_info['overlapping_words'] = overlapping_words
        
        result_info: Dict[str, Any] = {
            'word_count': result_word_count,
            'words_from_chunk1': words_from_chunk1,
            'words_from_chunk2': words_from_chunk2
        }
        if result_time_range:
            result_info['time_range'] = result_time_range
        
        json_data = {
            'step': step_num,
            'chunk1': chunk1_info.to_dict(),
            'chunk2': chunk2_info.to_dict(),
            'overlap_detection': overlap_info,
            'result': result_info
        }
        
        with open(self.debug_dir / f"step_{step_num:03d}_stitch.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def save_output(
        self, 
        result: Union[str, List[WordSegment]], 
        words_have_timestamps: bool
    ) -> None:
        """
        Save final stitched output for debugging.
        
        Args:
            result: The stitched result (string or list of WordSegments)
            words_have_timestamps: Whether the output has timestamps
        """
        if not self.debug_dir:
            return
        
        if words_have_timestamps and isinstance(result, list):
            words_list = [
                {'text': w.text, 'start': w.start, 'end': w.end, 'speaker': w.speaker} 
                for w in result
            ]
            json_data: Dict[str, Any] = {
                'total_words': len(words_list), 
                'has_timestamps': True, 
                'words': words_list
            }
        else:
            words = result.split() if isinstance(result, str) else result
            json_data = {
                'total_words': len(words), 
                'has_timestamps': False, 
                'text': result
            }
        
        with open(self.debug_dir / "99_final_output.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def save_session_summary(
        self,
        chunks: List[Dict[str, Any]],
        words_have_timestamps: bool,
        total_stitch_steps: int,
        final_word_count: int,
    ) -> None:
        """
        Save session summary for debugging.
        
        Args:
            chunks: Original input chunks
            words_have_timestamps: Whether words have timestamps
            total_stitch_steps: Number of stitching steps performed
            final_word_count: Number of words in final output
        """
        if not self.debug_dir:
            return
        
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'min_overlap_ratio': self.config.min_overlap_ratio,
                'similarity_threshold': self.config.similarity_threshold,
                'sequence_alignment_window': self.config.sequence_alignment_window,
                'min_sequence_match_length': self.config.min_sequence_match_length
            },
            'input': {
                'total_chunks': len(chunks),
                'has_timestamps': words_have_timestamps,
                'total_input_words': sum(len(c['words']) for c in chunks)
            },
            'processing': {
                'stitch_steps': total_stitch_steps
            },
            'output': {
                'final_word_count': final_word_count
            }
        }
        
        with open(self.debug_dir / "session_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def capture_chunk_info(
    chunk: List[Any],
    chunk_id: Optional[Any],
    words_have_timestamps: bool,
    is_first_chunk: bool = True,
) -> ChunkInfo:
    """
    Capture info about a chunk for debug output.
    
    Args:
        chunk: The chunk's word list
        chunk_id: ID of the chunk (if any)
        words_have_timestamps: Whether words have timestamps
        is_first_chunk: If True, capture last 10 words; if False, capture first 10
        
    Returns:
        ChunkInfo with captured data
    """
    info = ChunkInfo(
        chunk_id=chunk_id if not is_first_chunk else None,
        word_count=len(chunk),
    )
    
    if words_have_timestamps:
        if is_first_chunk:
            sample = chunk[-10:] if len(chunk) >= 10 else chunk
            info.last_10_words = [w.get('text', '') for w in sample]
        else:
            sample = chunk[:10] if len(chunk) >= 10 else chunk
            info.first_10_words = [w.get('text', '') for w in sample]
        
        if chunk:
            info.time_range = {
                'start': chunk[0].get('start', 0),
                'end': chunk[-1].get('end', 0)
            }
    else:
        if is_first_chunk:
            info.last_10_words = chunk[-10:] if len(chunk) >= 10 else list(chunk)
        else:
            info.first_10_words = chunk[:10] if len(chunk) >= 10 else list(chunk)
    
    return info
