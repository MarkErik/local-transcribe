#!/usr/bin/env python3
"""
VAD Audit utilities for generating audit/debug files.

This module provides functions for writing VAD and turn building
audit files for debugging and analysis.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from local_transcribe.processing.vad.data_structures import VADSegment, VADBlock
from local_transcribe.lib.program_logger import log_intermediate_save


def write_vad_audit(
    vad_segments: Dict[str, List[VADSegment]],
    output_path: Path,
    run_id: str,
    speaker_audio_files: Optional[Dict[str, str]] = None,
    speaker_durations: Optional[Dict[str, float]] = None,
    vad_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write vad_audit_<run>.json with normalized per-speaker VAD data.
    
    Args:
        vad_segments: Mapping of speaker_id to their VAD segments
        output_path: Directory to write the audit file
        run_id: Unique run identifier
        speaker_audio_files: Optional mapping of speaker_id to audio file path
        speaker_durations: Optional mapping of speaker_id to audio duration
        vad_config: Optional VAD configuration used
        
    Returns:
        Path to the written audit file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audit_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "speakers": {},
    }
    
    for speaker_id, segments in vad_segments.items():
        speaker_data = {
            "audio_file": speaker_audio_files.get(speaker_id) if speaker_audio_files else None,
            "duration_s": speaker_durations.get(speaker_id) if speaker_durations else None,
            "vad_config": vad_config,
            "segments": [seg.to_dict() for seg in segments],
            "total_speech_s": sum(seg.duration_s for seg in segments),
            "segment_count": len(segments),
        }
        audit_data["speakers"][speaker_id] = speaker_data
    
    # Summary
    audit_data["summary"] = {
        "total_speakers": len(vad_segments),
        "total_segments": sum(len(segs) for segs in vad_segments.values()),
        "total_speech_s": sum(
            sum(seg.duration_s for seg in segs) 
            for segs in vad_segments.values()
        ),
    }
    
    audit_file = output_path / f"vad_audit_{run_id}.json"
    with open(audit_file, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    
    log_intermediate_save(str(audit_file), "VAD audit saved to")
    return audit_file


def write_turn_building_audit(
    blocks: List[VADBlock],
    output_path: Path,
    run_id: str,
    config: Optional[Dict[str, Any]] = None,
    asr_chunk_data: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Write turn_building_audit_<run>.json with block traces.
    
    Args:
        blocks: List of VAD blocks (turns)
        output_path: Directory to write the audit file
        run_id: Unique run identifier
        config: Block builder configuration used
        asr_chunk_data: Optional ASR chunk audit data
        
    Returns:
        Path to the written audit file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build block data with ASR chunk references
    block_data = []
    asr_chunks_by_block: Dict[int, List[Dict[str, Any]]] = {}
    
    if asr_chunk_data:
        for chunk in asr_chunk_data:
            block_id = chunk["block_id"]
            if block_id not in asr_chunks_by_block:
                asr_chunks_by_block[block_id] = []
            asr_chunks_by_block[block_id].append(chunk)
    
    for block in blocks:
        block_dict = block.to_dict()
        
        # Add ASR chunk info if available
        if block.block_id in asr_chunks_by_block:
            block_dict["asr_chunks"] = asr_chunks_by_block[block.block_id]
        
        # Add transcript preview
        if block.text:
            preview_length = 60
            block_dict["transcript_preview"] = (
                block.text[:preview_length] + "..." 
                if len(block.text) > preview_length 
                else block.text
            )
        
        block_data.append(block_dict)
    
    audit_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "blocks": block_data,
        "summary": {
            "total_blocks": len(blocks),
            "interjection_count": sum(1 for b in blocks if b.is_interjection),
            "overlap_count": sum(1 for b in blocks if b.overlap_with),
            "blocks_with_chunking": len(asr_chunks_by_block),
            "total_asr_chunks": sum(len(chunks) for chunks in asr_chunks_by_block.values()),
            "speakers": list(set(b.speaker_id for b in blocks)),
            "total_duration_s": sum(b.duration_s for b in blocks),
        },
    }
    
    audit_file = output_path / f"turn_building_audit_{run_id}.json"
    with open(audit_file, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    
    log_intermediate_save(str(audit_file), "Turn building audit saved to")
    return audit_file


def write_asr_chunks_audit(
    chunk_data: List[Dict[str, Any]],
    output_path: Path,
    run_id: str,
) -> Path:
    """
    Write asr_chunks_<run>.json with ASR chunking details.
    
    Args:
        chunk_data: List of chunk audit records
        output_path: Directory to write the audit file
        run_id: Unique run identifier
        
    Returns:
        Path to the written audit file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audit_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "chunks": chunk_data,
        "summary": {
            "total_chunks": len(chunk_data),
            "blocks_with_chunks": len(set(c["block_id"] for c in chunk_data)),
            "total_words": sum(c.get("word_count", 0) for c in chunk_data),
        },
    }
    
    audit_file = output_path / f"asr_chunks_{run_id}.json"
    with open(audit_file, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    
    log_intermediate_save(str(audit_file), "ASR chunks audit saved to")
    return audit_file
