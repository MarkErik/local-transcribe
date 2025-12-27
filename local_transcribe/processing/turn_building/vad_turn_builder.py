#!/usr/bin/env python3
"""
VAD Turn Builder for building conversation turns from split audio files.

This module implements the VAD-first approach for split audio transcription:
1. Run Silero VAD on each speaker's audio file
2. Build blocks by merging VAD segments per speaker
3. Interleave blocks from all speakers by timeline
4. Process each block through ASR (with chunking if >30s)
5. Classify interjections and annotate overlaps
6. Return TranscriptFlow with conversation turns
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from local_transcribe.processing.vad.data_structures import VADSegment, VADBlock, VADBlockBuilderConfig
from local_transcribe.processing.vad.silero_vad import SileroVADProcessor
from local_transcribe.processing.vad.vad_block_builder import VADBlockBuilder
from local_transcribe.processing.vad.vad_asr_processor import VADASRProcessor
from local_transcribe.processing.vad.vad_audit import (
    write_vad_audit,
    write_turn_building_audit,
    write_asr_chunks_audit,
)
from local_transcribe.processing.turn_building.turn_building_data_structures import (
    TranscriptFlow,
    HierarchicalTurn,
    InterjectionSegment,
)
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment
from local_transcribe.lib.program_logger import log_status, log_progress, log_completion, log_debug


def _validate_audio_durations(
    speaker_audio_files: Dict[str, str],
    tolerance_s: float = 1.0
) -> None:
    """
    Validate that all speaker audio files have approximately the same duration.
    
    Args:
        speaker_audio_files: Mapping of speaker_id to audio file path
        tolerance_s: Maximum allowed difference in duration (seconds)
        
    Raises:
        ValueError: If audio files have significantly different durations
    """
    from local_transcribe.lib.audio_processor import load_audio_as_array
    
    durations = {}
    for speaker_id, audio_path in speaker_audio_files.items():
        audio, sr = load_audio_as_array(audio_path)
        durations[speaker_id] = len(audio) / sr
    
    if len(durations) < 2:
        return
    
    duration_values = list(durations.values())
    max_diff = max(duration_values) - min(duration_values)
    
    if max_diff > tolerance_s:
        duration_str = ", ".join(f"{k}: {v:.1f}s" for k, v in durations.items())
        raise ValueError(
            f"Audio files have significantly different durations ({duration_str}). "
            f"Maximum difference of {max_diff:.1f}s exceeds tolerance of {tolerance_s}s. "
            "Ensure all audio files are from the same recording session."
        )


def _create_word_segments_from_block(
    block: VADBlock
) -> List[WordSegment]:
    """
    Create WordSegment objects from a VADBlock.
    
    Since VAD pipeline doesn't do word-level alignment, we create
    a single "word" segment for the entire block text.
    
    Args:
        block: VADBlock with text populated
        
    Returns:
        List of WordSegment (one per word, with block-level timing)
    """
    if not block.text:
        return []
    
    words = block.text.split()
    if not words:
        return []
    
    # Distribute timing across words (approximate)
    duration_per_word = block.duration_s / len(words) if words else 0
    segments = []
    
    for i, word in enumerate(words):
        start = block.start_s + i * duration_per_word
        end = start + duration_per_word
        
        segments.append(WordSegment(
            text=word,
            start=start,
            end=end,
            speaker=block.speaker_id,
        ))
    
    return segments


def _convert_blocks_to_transcript_flow(
    blocks: List[VADBlock],
    run_id: str,
) -> TranscriptFlow:
    """
    Convert VAD blocks into a TranscriptFlow.
    
    Primary blocks become HierarchicalTurns, interjections become
    InterjectionSegments embedded in the appropriate turns.
    
    Args:
        blocks: List of VADBlock with text populated
        run_id: Unique run identifier
        
    Returns:
        TranscriptFlow with conversation structure
    """
    turns: List[HierarchicalTurn] = []
    turn_id = 0
    
    # Separate primary blocks from interjections
    primary_blocks = [b for b in blocks if not b.is_interjection]
    interjection_blocks = [b for b in blocks if b.is_interjection]
    
    # Convert primary blocks to turns
    for block in primary_blocks:
        words = _create_word_segments_from_block(block)
        
        turn = HierarchicalTurn(
            turn_id=turn_id,
            primary_speaker=block.speaker_id,
            start=block.start_s,
            end=block.end_s,
            text=block.text,
            words=words,
            interjections=[],
        )
        turns.append(turn)
        turn_id += 1
    
    # Embed interjections into appropriate turns
    for ij_block in interjection_blocks:
        words = _create_word_segments_from_block(ij_block)
        
        interjection = InterjectionSegment(
            speaker=ij_block.speaker_id,
            start=ij_block.start_s,
            end=ij_block.end_s,
            text=ij_block.text,
            words=words,
            confidence=0.8,  # Default confidence for VAD-detected interjections
            interjection_type="acknowledgment",  # Default type
            interrupt_level="low",  # Default level
            classification_method="vad",
            likely_diarization_error=False,
        )
        
        # Find the turn this interjection belongs to
        for turn in turns:
            if turn.start <= ij_block.start_s <= turn.end:
                turn.interjections.append(interjection)
                break
        else:
            # If no overlapping turn, find the nearest preceding turn
            preceding_turns = [t for t in turns if t.end <= ij_block.start_s]
            if preceding_turns:
                preceding_turns[-1].interjections.append(interjection)
    
    # Sort interjections within each turn by time
    for turn in turns:
        turn.interjections.sort(key=lambda ij: ij.start)
        turn.recalculate_metrics()
    
    # Calculate conversation metrics
    speaker_stats: Dict[str, Dict[str, Any]] = {}
    for speaker in set(b.speaker_id for b in blocks):
        speaker_blocks = [b for b in blocks if b.speaker_id == speaker]
        speaker_stats[speaker] = {
            "total_blocks": len(speaker_blocks),
            "total_duration_s": sum(b.duration_s for b in speaker_blocks),
            "total_words": sum(len(b.text.split()) for b in speaker_blocks if b.text),
            "interjection_count": sum(1 for b in speaker_blocks if b.is_interjection),
        }
    
    transcript = TranscriptFlow(
        turns=turns,
        metadata={
            "run_id": run_id,
            "pipeline": "vad_split_audio",
            "timestamp": datetime.now().isoformat(),
            "total_blocks": len(blocks),
            "total_turns": len(turns),
            "total_interjections": len(interjection_blocks),
        },
        conversation_metrics={
            "total_duration_s": max(b.end_s for b in blocks) - min(b.start_s for b in blocks) if blocks else 0,
            "overlap_count": sum(1 for b in blocks if b.overlap_with),
        },
        speaker_statistics=speaker_stats,
    )
    
    return transcript


def build_turns_vad_split_audio(
    speaker_audio_files: Dict[str, str],
    transcriber_provider: TranscriberProvider,
    config: Optional[VADBlockBuilderConfig] = None,
    intermediate_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    vad_threshold: float = 0.5,
    remote_granite_url: Optional[str] = None,
    validate_durations: bool = True,
    **kwargs
) -> TranscriptFlow:
    """
    Build conversation turns using VAD-first approach for split audio.
    
    Pipeline:
    1. Run Silero VAD on each speaker's audio file
    2. Build blocks by merging VAD segments per speaker
    3. Interleave blocks from all speakers by timeline
    4. Process each block through ASR (with chunking if >30s)
    5. Classify interjections and annotate overlaps
    6. Return TranscriptFlow with conversation turns
    
    Args:
        speaker_audio_files: Mapping of speaker IDs to audio file paths
        transcriber_provider: ASR provider to use (e.g., granite)
        config: VAD block building configuration
        intermediate_dir: Path for intermediate/debug files
        models_dir: Path to model cache directory
        vad_threshold: VAD speech probability threshold (0-1)
        remote_granite_url: URL for remote Granite server
        validate_durations: Whether to validate audio file durations match
        **kwargs: Additional arguments passed to transcriber
        
    Returns:
        TranscriptFlow containing the conversation transcript
        
    Raises:
        ValueError: If audio files have significantly different durations
        RuntimeError: If ASR fails for any block
    """
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_status(f"Starting VAD-first pipeline for {len(speaker_audio_files)} speakers")
    
    # Validate audio durations
    if validate_durations:
        log_progress("Validating audio file durations")
        _validate_audio_durations(speaker_audio_files)
    
    # Create intermediate directory structure
    if intermediate_dir:
        intermediate_dir = Path(intermediate_dir)
        vad_dir = intermediate_dir / "vad"
        vad_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize VAD processor
    vad_processor = SileroVADProcessor(
        threshold=vad_threshold,
        models_dir=models_dir,
    )
    
    # 1. Run VAD on each speaker's audio
    log_status("Running VAD on speaker audio files")
    all_vad_segments: Dict[str, List[VADSegment]] = {}
    speaker_durations: Dict[str, float] = {}
    
    for speaker_id, audio_path in speaker_audio_files.items():
        log_progress(f"Running VAD on {speaker_id}")
        segments = vad_processor.process_audio(audio_path, speaker_id)
        all_vad_segments[speaker_id] = segments
        speaker_durations[speaker_id] = vad_processor.get_audio_duration(audio_path)
    
    # Write VAD audit
    if intermediate_dir:
        write_vad_audit(
            all_vad_segments,
            vad_dir,
            run_id,
            speaker_audio_files=speaker_audio_files,
            speaker_durations=speaker_durations,
            vad_config=vad_processor.config,
        )
    
    # 2. Build blocks from VAD segments
    log_status("Building conversation blocks from VAD segments")
    block_config = config or VADBlockBuilderConfig()
    block_builder = VADBlockBuilder(config=block_config)
    blocks = block_builder.build_blocks(all_vad_segments)
    
    # 3. Process blocks through ASR
    log_status(f"Transcribing {len(blocks)} blocks")
    asr_processor = VADASRProcessor(
        transcriber_provider=transcriber_provider,
        models_dir=models_dir,
        intermediate_dir=intermediate_dir,
        remote_granite_url=remote_granite_url,
    )
    
    blocks = asr_processor.process_blocks(
        blocks,
        speaker_audio_files,
        **kwargs
    )
    
    # Write turn building audit
    if intermediate_dir:
        chunk_data = asr_processor.get_chunk_audit_data()
        write_turn_building_audit(
            blocks,
            vad_dir,
            run_id,
            config=block_config.to_dict(),
            asr_chunk_data=chunk_data,
        )
        
        # Write ASR chunks audit if there were chunked blocks
        if chunk_data:
            write_asr_chunks_audit(chunk_data, vad_dir, run_id)
    
    # 4. Convert to TranscriptFlow
    log_status("Building final transcript")
    transcript = _convert_blocks_to_transcript_flow(blocks, run_id)
    
    log_completion(
        f"VAD pipeline complete: {transcript.total_turns} turns, "
        f"{transcript.total_interjections} interjections"
    )
    
    return transcript
