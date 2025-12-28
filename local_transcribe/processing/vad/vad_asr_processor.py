#!/usr/bin/env python3
"""
VAD ASR Processor for transcribing VAD blocks with chunking and stitching.

This module handles the transcription of VAD blocks through an ASR provider,
with automatic chunking of long blocks and stitching of overlapping results.
"""

import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import soundfile as sf

from local_transcribe.processing.vad.data_structures import VADBlock, ASRChunk
from local_transcribe.processing.chunk_stitcher import ChunkStitcher
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment
from local_transcribe.lib.program_logger import log_progress, log_debug, log_completion, get_logger, get_output_context
from local_transcribe.lib.audio_processor import load_audio_as_array


class VADASRProcessor:
    """
    Processes VAD blocks through ASR with chunking and stitching.
    
    Handles long blocks by splitting them into chunks with overlap,
    transcribing each chunk, and stitching the results together.
    """
    
    # Chunk size parameters from implementation plan
    MAX_CHUNK_DURATION_S = 30.0
    OVERLAP_DURATION_S = 4.0
    MIN_CHUNK_DURATION_S = 7.0  # Minimum viable chunk size
    
    # Minimum audio duration for ASR (seconds)
    # Short audio will be padded with silence to reach this threshold
    MIN_AUDIO_DURATION_S = 1.0
    
    # Standard sample rate for ASR
    SAMPLE_RATE = 16000
    
    def __init__(
        self,
        transcriber_provider: TranscriberProvider,
        models_dir: Optional[Path] = None,
        intermediate_dir: Optional[Path] = None,
        remote_granite_url: Optional[str] = None,
    ):
        """
        Initialize the VAD ASR processor.
        
        Args:
            transcriber_provider: ASR provider to use for transcription
            models_dir: Path to model cache directory
            intermediate_dir: Path for intermediate/debug files
            remote_granite_url: URL for remote Granite server (if using remote)
        """
        self.transcriber = transcriber_provider
        self.models_dir = models_dir
        self.intermediate_dir = intermediate_dir
        self.remote_granite_url = remote_granite_url
        self.logger = get_logger()
        
        # Track chunk info for audit
        self._chunk_audit_data: List[Dict[str, Any]] = []
        
        # Debug directories - initialized on first use in process_blocks
        self.debug_enabled = False
        self.transcription_debug_dir: Optional[Path] = None
        self.stitching_debug_dir: Optional[Path] = None
        
        # Create chunk stitcher without intermediate_dir initially
        # We'll configure it properly in _setup_debug_directories
        self.chunk_stitcher = ChunkStitcher(intermediate_dir=None)
    
    def _setup_debug_directories(self) -> None:
        """
        Setup debug directories for the transcription run.
        
        Creates timestamped directories for transcription outputs and stitching debug.
        Only creates directories if DEBUG logging is enabled and intermediate_dir is set.
        """
        self.debug_enabled = get_output_context().should_log("DEBUG")
        
        if not self.debug_enabled or not self.intermediate_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create transcription debug directory for per-block ASR results
        self.transcription_debug_dir = Path(self.intermediate_dir) / "transcription" / f"vad_asr_{timestamp}"
        self.transcription_debug_dir.mkdir(parents=True, exist_ok=True)
        log_debug(f"Transcription debug output: {self.transcription_debug_dir}")
        
        # Create stitching debug directory for blocks that require chunk stitching
        self.stitching_debug_dir = Path(self.intermediate_dir) / "chunk_stitching" / f"vad_stitching_{timestamp}"
        # Don't create this directory yet - only create if we actually need stitching
        
    def _save_block_transcription_debug(
        self,
        block: VADBlock,
        text: str,
        was_chunked: bool,
        chunk_count: int = 1
    ) -> None:
        """
        Save debug output for a transcribed block.
        
        Args:
            block: The VAD block that was transcribed
            text: The transcription result
            was_chunked: Whether the block required chunking
            chunk_count: Number of chunks if chunking was needed
        """
        if not self.debug_enabled or not self.transcription_debug_dir:
            return
        
        # Use block_id and source segment IDs for audit trail
        block_id_str = f"block_{block.block_id:03d}"
        segment_ids_str = "_".join(str(sid) for sid in block.source_segment_ids)
        
        # Save JSON with full details
        debug_data = {
            "block_id": block.block_id,
            "speaker_id": block.speaker_id,
            "source_segment_ids": block.source_segment_ids,
            "start_s": block.start_s,
            "end_s": block.end_s,
            "duration_s": block.duration_s,
            "was_chunked": was_chunked,
            "chunk_count": chunk_count,
            "transcription": {
                "text": text,
                "word_count": len(text.split()) if text else 0,
            }
        }
        
        json_path = self.transcription_debug_dir / f"{block_id_str}_segments_{segment_ids_str}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable text file
        txt_path = self.transcription_debug_dir / f"{block_id_str}_segments_{segment_ids_str}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"BLOCK {block.block_id} - ASR OUTPUT\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Speaker: {block.speaker_id}\n")
            f.write(f"Source VAD Segment IDs: {block.source_segment_ids}\n")
            f.write(f"Time: {block.start_s:.2f}s - {block.end_s:.2f}s ({block.duration_s:.2f}s)\n")
            f.write(f"Chunked: {'Yes (' + str(chunk_count) + ' chunks)' if was_chunked else 'No'}\n")
            f.write(f"Word count: {len(text.split()) if text else 0}\n")
            f.write(f"{'-' * 60}\n\n")
            f.write(text if text else "[No transcription]")
            f.write("\n")

    def process_blocks(
        self,
        blocks: List[VADBlock],
        speaker_audio_files: Dict[str, str],
        **kwargs
    ) -> List[VADBlock]:
        """
        Process multiple VAD blocks through ASR.
        
        Args:
            blocks: List of VAD blocks to transcribe
            speaker_audio_files: Mapping of speaker_id to audio file path
            **kwargs: Additional arguments passed to transcriber
            
        Returns:
            List of VADBlock with text field populated
        """
        log_progress(f"Transcribing {len(blocks)} VAD blocks")
        
        # Setup debug directories for this run
        self._setup_debug_directories()
        
        # Load audio data for each speaker
        speaker_audio: Dict[str, tuple] = {}  # speaker_id -> (audio_array, sample_rate)
        for speaker_id, audio_path in speaker_audio_files.items():
            audio_array, sr = load_audio_as_array(audio_path)
            speaker_audio[speaker_id] = (audio_array, sr)
            log_debug(f"Loaded audio for {speaker_id}: {len(audio_array)/sr:.1f}s")
        
        # Process each block
        blocks_needing_chunking = 0
        for i, block in enumerate(blocks):
            if block.duration_s > self.MAX_CHUNK_DURATION_S:
                blocks_needing_chunking += 1
        
        if blocks_needing_chunking > 0:
            log_progress(f"{blocks_needing_chunking} blocks require chunking (>{self.MAX_CHUNK_DURATION_S}s)")
        
        for i, block in enumerate(blocks):
            if (i + 1) % 10 == 0 or i == 0:
                log_progress(f"Processing block {i + 1}/{len(blocks)} ({block.speaker_id})")
            
            audio_data, sr = speaker_audio[block.speaker_id]
            
            try:
                text = self.process_block(block, audio_data, sr, **kwargs)
                block.text = text
            except Exception as e:
                self.logger.error(f"ASR failed for block {block.block_id}: {e}")
                raise RuntimeError(f"ASR failed for block {block.block_id} ({block.speaker_id}): {e}")
        
        log_completion(f"Transcription complete: {len(blocks)} blocks processed")
        return blocks
    
    def process_block(
        self,
        block: VADBlock,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        **kwargs
    ) -> str:
        """
        Process a single block through ASR.
        
        If block duration > 30s, splits into chunks with 4s overlap,
        then stitches results. Small final chunks (<7s) are merged with
        the previous chunk to avoid poor ASR quality on very short segments.
        
        Args:
            block: VAD block to transcribe
            audio_data: Full audio array for this speaker
            sample_rate: Sample rate of audio
            **kwargs: Additional arguments passed to transcriber
            
        Returns:
            Transcribed text for this block
        """
        # Extract audio segment for this block
        start_sample = int(block.start_s * sample_rate)
        end_sample = int(block.end_s * sample_rate)
        block_audio = audio_data[start_sample:end_sample]
        
        if block.duration_s <= self.MAX_CHUNK_DURATION_S:
            # Direct transcription for short blocks
            text = self._transcribe_audio(block_audio, sample_rate, block.speaker_id, block=block, **kwargs)
            # Save debug output for non-chunked block
            self._save_block_transcription_debug(block, text, was_chunked=False, chunk_count=1)
            return text
        
        # Long block - need to chunk
        log_debug(f"Block {block.block_id} duration {block.duration_s:.1f}s > {self.MAX_CHUNK_DURATION_S}s, chunking")
        
        # Split into chunks
        chunks = self._split_block_into_chunks(block, block_audio, sample_rate)
        
        # Transcribe each chunk
        chunk_results: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = self._transcribe_audio(chunk.audio_segment, sample_rate, chunk.speaker_id, block=block, chunk=chunk, **kwargs)
            chunk_results.append({
                "chunk_id": chunk.chunk_id,
                "words": text.split(),  # Simple word splitting for stitching
                "text": text,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
            })
            
            # Record for audit
            self._chunk_audit_data.append({
                "block_id": block.block_id,
                "chunk_id": chunk.chunk_id,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
                "duration_s": chunk.duration_s,
                "word_count": len(text.split()),
            })
        
        # Stitch results
        if len(chunk_results) == 1:
            result_text = chunk_results[0]["text"]
            self._save_block_transcription_debug(block, result_text, was_chunked=True, chunk_count=1)
            return result_text
        
        result_text = self._stitch_chunk_transcripts(chunk_results, block)
        # Save debug output for chunked block
        self._save_block_transcription_debug(block, result_text, was_chunked=True, chunk_count=len(chunks))
        return result_text
    
    def _split_block_into_chunks(
        self,
        block: VADBlock,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[ASRChunk]:
        """
        Split a long block into <=30s chunks with 4s overlap.
        
        Logic:
        1. Calculate effective chunk length (30s - 4s overlap = 26s advance)
        2. Iterate through audio creating chunks of MAX_CHUNK_DURATION_S
        3. Each subsequent chunk starts overlap_duration before previous chunk ends
        4. If final remaining chunk is < MIN_CHUNK_DURATION_S (7s):
           - Merge it with the previous chunk by extending the previous chunk
           - Do NOT create a separate small chunk
        
        Args:
            block: Source VAD block
            audio_data: Audio samples for this block
            sample_rate: Sample rate
            
        Returns:
            List of ASRChunk objects ready for transcription
        """
        total_samples = len(audio_data)
        total_duration = total_samples / sample_rate
        
        max_chunk_samples = int(self.MAX_CHUNK_DURATION_S * sample_rate)
        overlap_samples = int(self.OVERLAP_DURATION_S * sample_rate)
        min_chunk_samples = int(self.MIN_CHUNK_DURATION_S * sample_rate)
        
        # Effective stride between chunk starts
        stride_samples = max_chunk_samples - overlap_samples
        
        chunks: List[ASRChunk] = []
        chunk_id = 0
        chunk_start = 0
        
        while chunk_start < total_samples:
            chunk_end = min(chunk_start + max_chunk_samples, total_samples)
            chunk_audio = audio_data[chunk_start:chunk_end]
            chunk_samples = len(chunk_audio)
            
            # Check if this is a small final chunk that should be merged
            remaining_samples = total_samples - chunk_start
            is_final_chunk = chunk_end >= total_samples
            is_small_chunk = chunk_samples < min_chunk_samples
            
            if is_final_chunk and is_small_chunk and len(chunks) > 0:
                # Merge with previous chunk
                prev_chunk = chunks[-1]
                
                # Calculate how much new audio to add (non-overlapping portion)
                # The overlap region is at the start of this chunk
                if chunk_start > 0:
                    new_audio_start = overlap_samples if chunk_samples >= overlap_samples else 0
                    new_audio = chunk_audio[new_audio_start:]
                    
                    # Extend previous chunk's audio
                    extended_audio = np.concatenate([prev_chunk.audio_segment, new_audio])
                    
                    # Update previous chunk
                    prev_chunk.audio_segment = extended_audio
                    prev_chunk.end_s = block.start_s + chunk_end / sample_rate
                    
                    log_debug(
                        f"Merged small final chunk ({chunk_samples/sample_rate:.1f}s) "
                        f"into previous chunk, new duration: {len(extended_audio)/sample_rate:.1f}s"
                    )
                break
            
            # Create chunk
            chunk = ASRChunk(
                chunk_id=chunk_id,
                speaker_id=block.speaker_id,
                source_block_id=block.block_id,
                start_s=block.start_s + chunk_start / sample_rate,
                end_s=block.start_s + chunk_end / sample_rate,
                audio_segment=chunk_audio,
                overlap_start_s=block.start_s + (chunk_start + overlap_samples) / sample_rate if chunk_start > 0 else 0,
            )
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk position
            chunk_start += stride_samples
        
        log_debug(f"Split block {block.block_id} ({total_duration:.1f}s) into {len(chunks)} chunks")
        return chunks
    
    def _transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speaker_id: str,
        block: Optional[VADBlock] = None,
        chunk: Optional[ASRChunk] = None,
        **kwargs
    ) -> str:
        """
        Transcribe an audio segment using the ASR provider.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            speaker_id: Speaker identifier
            block: Optional VAD block context (for debug)
            chunk: Optional ASR chunk context (for debug)
            **kwargs: Additional arguments for transcriber
            
        Returns:
            Transcribed text
        """
        # Pad short audio segments with silence to meet minimum duration requirement
        audio_to_transcribe = self._pad_short_audio(audio_data, sample_rate)
        
        # Write audio to temp file (most ASR providers expect file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_to_transcribe, sample_rate)
        
        try:
            # Get device preference
            from local_transcribe.lib.system_capability_utils import get_system_capability
            device = get_system_capability()
            
            # Build transcriber kwargs
            transcriber_kwargs: Dict[str, Any] = {
                'device': device,
                'role': speaker_id,
            }
            
            # Add remote granite URL if available
            if self.remote_granite_url:
                transcriber_kwargs['use_remote_granite'] = True
                transcriber_kwargs['remote_granite_url'] = self.remote_granite_url
            
            if self.models_dir:
                transcriber_kwargs['models_dir'] = self.models_dir
            
            # Merge with passed kwargs
            transcriber_kwargs.update(kwargs)
            
            # Transcribe
            result = self.transcriber.transcribe(temp_path, **transcriber_kwargs)
            
            # Handle different return types
            if isinstance(result, str):
                return result
            elif isinstance(result, list):
                # Chunked output - stitch
                if result and isinstance(result[0], dict) and 'words' in result[0]:
                    stitched = self.chunk_stitcher.stitch_chunks(result)
                    if isinstance(stitched, str):
                        return stitched
                    elif isinstance(stitched, list):
                        return ' '.join(seg.text for seg in stitched)
                return ' '.join(str(r) for r in result)
            else:
                return str(result)
                
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def _pad_short_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Pad short audio segments with silence to meet minimum duration requirement.
        
        If the audio is shorter than MIN_AUDIO_DURATION_S, silence is added
        equally before and after the actual audio to reach the threshold.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate of audio
            
        Returns:
            Original audio if long enough, or padded audio if too short
        """
        current_duration = len(audio_data) / sample_rate
        
        if current_duration >= self.MIN_AUDIO_DURATION_S:
            return audio_data
        
        # Calculate padding needed
        target_samples = int(self.MIN_AUDIO_DURATION_S * sample_rate)
        padding_needed = target_samples - len(audio_data)
        
        # Split padding equally before and after
        pad_before = padding_needed // 2
        pad_after = padding_needed - pad_before
        
        log_debug(
            f"Padding short audio ({current_duration:.2f}s) with "
            f"{pad_before / sample_rate:.2f}s before and "
            f"{pad_after / sample_rate:.2f}s after to reach {self.MIN_AUDIO_DURATION_S}s"
        )
        
        # Create silence arrays and concatenate
        silence_before = np.zeros(pad_before, dtype=audio_data.dtype)
        silence_after = np.zeros(pad_after, dtype=audio_data.dtype)
        
        return np.concatenate([silence_before, audio_data, silence_after])
    
    def _stitch_chunk_transcripts(
        self,
        chunk_results: List[Dict[str, Any]],
        block: Optional[VADBlock] = None
    ) -> str:
        """
        Stitch overlapping chunk transcripts using ChunkStitcher.
        
        Args:
            chunk_results: List of chunk results with words
            block: The source VAD block (for debug context)
            
        Returns:
            Stitched transcript text
        """
        # Only enable stitching debug if we're actually stitching multiple chunks
        if self.debug_enabled and self.stitching_debug_dir and len(chunk_results) > 1:
            # Create the stitching debug directory if it doesn't exist
            self.stitching_debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a block-specific subdirectory for this stitching operation
            if block:
                block_debug_dir = self.stitching_debug_dir / f"block_{block.block_id:03d}"
                block_debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a new chunk stitcher with the block-specific debug directory
                # Use use_timestamped_debug_dir=False to use the directory directly
                block_stitcher = ChunkStitcher(
                    intermediate_dir=block_debug_dir,
                    skip_single_chunk_debug=True,
                    use_timestamped_debug_dir=False  # Use the directory directly
                )
                result = block_stitcher.stitch_chunks(chunk_results)
            else:
                result = self.chunk_stitcher.stitch_chunks(chunk_results)
        else:
            # Use default stitcher without debug output
            result = self.chunk_stitcher.stitch_chunks(chunk_results)
        
        if isinstance(result, str):
            return result
        elif isinstance(result, list):
            # List of WordSegments
            return ' '.join(seg.text for seg in result)
        
        return str(result)
    
    def get_chunk_audit_data(self) -> List[Dict[str, Any]]:
        """
        Get audit data for all chunks processed.
        
        Returns:
            List of chunk audit records
        """
        return self._chunk_audit_data
    
    def clear_chunk_audit_data(self) -> None:
        """Clear accumulated chunk audit data."""
        self._chunk_audit_data = []
