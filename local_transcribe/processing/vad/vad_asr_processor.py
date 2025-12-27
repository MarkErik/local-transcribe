#!/usr/bin/env python3
"""
VAD ASR Processor for transcribing VAD blocks with chunking and stitching.

This module handles the transcription of VAD blocks through an ASR provider,
with automatic chunking of long blocks and stitching of overlapping results.
"""

import tempfile
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import soundfile as sf

from local_transcribe.processing.vad.data_structures import VADBlock, ASRChunk
from local_transcribe.processing.chunk_stitcher import ChunkStitcher
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment
from local_transcribe.lib.program_logger import log_progress, log_debug, log_completion, get_logger
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
        self.chunk_stitcher = ChunkStitcher(intermediate_dir=intermediate_dir)
        self.logger = get_logger()
        
        # Track chunk info for audit
        self._chunk_audit_data: List[Dict[str, Any]] = []
    
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
            return self._transcribe_audio(block_audio, sample_rate, block.speaker_id, **kwargs)
        
        # Long block - need to chunk
        log_debug(f"Block {block.block_id} duration {block.duration_s:.1f}s > {self.MAX_CHUNK_DURATION_S}s, chunking")
        
        # Split into chunks
        chunks = self._split_block_into_chunks(block, block_audio, sample_rate)
        
        # Transcribe each chunk
        chunk_results: List[Dict[str, Any]] = []
        for chunk in chunks:
            text = self._transcribe_audio(chunk.audio_segment, sample_rate, chunk.speaker_id, **kwargs)
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
            return chunk_results[0]["text"]
        
        return self._stitch_chunk_transcripts(chunk_results)
    
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
        **kwargs
    ) -> str:
        """
        Transcribe an audio segment using the ASR provider.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            speaker_id: Speaker identifier
            **kwargs: Additional arguments for transcriber
            
        Returns:
            Transcribed text
        """
        # Write audio to temp file (most ASR providers expect file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_data, sample_rate)
        
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
    
    def _stitch_chunk_transcripts(
        self,
        chunk_results: List[Dict[str, Any]]
    ) -> str:
        """
        Stitch overlapping chunk transcripts using ChunkStitcher.
        
        Args:
            chunk_results: List of chunk results with words
            
        Returns:
            Stitched transcript text
        """
        # Use chunk_stitcher for overlap detection
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
