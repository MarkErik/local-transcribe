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

from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime

from local_transcribe.processing.vad.types import VADSegment, VADBlock, VADBlockBuilderConfig
from local_transcribe.providers.vad import SileroVADProvider
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
from local_transcribe.lib.program_logger import log_status, log_progress, log_completion, log_debug, log_intermediate_save


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


def _merge_consecutive_same_speaker_blocks(
    primary_blocks: List[VADBlock],
    interjection_blocks: List[VADBlock],
    max_gap_s: float = 2.0,
) -> List[tuple[VADBlock, List[int]]]:
    """
    Merge consecutive primary blocks from the same speaker when gap is small.
    
    This fixes fragmentation where a single thought is split across multiple
    VAD blocks due to brief pauses. Blocks separated only by interjections
    from the other speaker are also merged.
    
    Args:
        primary_blocks: List of primary (non-interjection) blocks, sorted by time
        interjection_blocks: List of interjection blocks
        max_gap_s: Maximum gap between blocks to allow merging (seconds)
        
    Returns:
        List of tuples: (merged VADBlock, list of source block IDs)
    """
    if not primary_blocks:
        return []
    
    # Sort by start time to ensure correct order
    sorted_blocks = sorted(primary_blocks, key=lambda b: b.start_s)
    
    merged: List[tuple[VADBlock, List[int]]] = []
    current = sorted_blocks[0]
    current_source_ids = [current.block_id]
    
    for i in range(1, len(sorted_blocks)):
        next_block = sorted_blocks[i]
        gap = next_block.start_s - current.end_s
        
        # Check if same speaker and gap is small enough
        if next_block.speaker_id == current.speaker_id and gap <= max_gap_s:
            # Check if only interjections fill the gap (not primary speech from other speaker)
            # Look for any primary blocks from OTHER speakers in the gap
            gap_has_other_primary = False
            for b in sorted_blocks[len(merged):i]:
                if (b.speaker_id != current.speaker_id and 
                    b.start_s >= current.end_s and 
                    b.end_s <= next_block.start_s):
                    gap_has_other_primary = True
                    break
            
            if not gap_has_other_primary:
                # Merge: extend current block to include next
                current = VADBlock(
                    block_id=current.block_id,
                    speaker_id=current.speaker_id,
                    start_s=current.start_s,
                    end_s=next_block.end_s,
                    source_segments=current.source_segments + next_block.source_segments,
                    is_interjection=False,
                    overlap_with=_merge_overlap_lists(current.overlap_with, next_block.overlap_with),
                    text=_merge_text(current.text, next_block.text),
                )
                # Track all source block IDs
                current_source_ids.append(next_block.block_id)
                continue
        
        # Can't merge - save current and start new
        merged.append((current, current_source_ids))
        current = next_block
        current_source_ids = [current.block_id]
    
    # Don't forget the last block
    merged.append((current, current_source_ids))
    
    return merged


def _merge_overlap_lists(
    list1: Optional[List[int]], 
    list2: Optional[List[int]]
) -> Optional[List[int]]:
    """Merge two overlap lists, removing duplicates."""
    if list1 is None and list2 is None:
        return None
    combined = set(list1 or []) | set(list2 or [])
    return sorted(combined) if combined else None


def _merge_text(text1: str, text2: str) -> str:
    """Merge two text strings with a space separator."""
    t1 = text1.strip() if text1 else ""
    t2 = text2.strip() if text2 else ""
    if t1 and t2:
        return f"{t1} {t2}"
    return t1 or t2


def _convert_blocks_to_transcript_flow(
    blocks: List[VADBlock],
    run_id: str,
    max_gap_to_merge_s: float = 2.0,
) -> TranscriptFlow:
    """
    Convert VAD blocks into a TranscriptFlow.
    
    Primary blocks become HierarchicalTurns, interjections become
    InterjectionSegments embedded in the appropriate turns.
    
    Consecutive same-speaker primary blocks with small gaps are merged
    to avoid fragmenting continuous thoughts.
    
    The source_block_ids field on each HierarchicalTurn tracks which VAD
    blocks contributed to that turn, enabling UI highlighting and edit
    resolution back to the original blocks.
    
    Args:
        blocks: List of VADBlock with text populated
        run_id: Unique run identifier
        max_gap_to_merge_s: Max gap between same-speaker blocks to merge (seconds)
        
    Returns:
        TranscriptFlow with conversation structure
    """
    turns: List[HierarchicalTurn] = []
    turn_id = 0
    
    # Separate primary blocks from interjections
    primary_blocks = [b for b in blocks if not b.is_interjection]
    interjection_blocks = [b for b in blocks if b.is_interjection]
    
    # Merge consecutive same-speaker primary blocks
    # Returns list of (merged_block, source_block_ids) tuples
    merged_primary_blocks = _merge_consecutive_same_speaker_blocks(
        primary_blocks, 
        interjection_blocks,
        max_gap_s=max_gap_to_merge_s,
    )
    
    # Convert merged primary blocks to turns
    for block, source_block_ids in merged_primary_blocks:
        words = _create_word_segments_from_block(block)
        
        turn = HierarchicalTurn(
            turn_id=turn_id,
            primary_speaker=block.speaker_id,
            start=block.start_s,
            end=block.end_s,
            text=block.text,
            words=words,
            interjections=[],
            source_block_ids=source_block_ids,  # Capture which VAD blocks contributed
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
    
    # Calculate total duration and extract unique speakers
    total_duration = max(b.end_s for b in blocks) - min(b.start_s for b in blocks) if blocks else 0
    unique_speakers = list(set(b.speaker_id for b in blocks))
    
    transcript = TranscriptFlow(
        turns=turns,
        metadata={
            "run_id": run_id,
            "pipeline": "vad_split_audio",
            "timestamp": datetime.now().isoformat(),
            "total_blocks": len(blocks),
            "total_turns": len(turns),
            "total_interjections": len(interjection_blocks),
            "duration": total_duration,
            "speakers": unique_speakers,
        },
        conversation_metrics={
            "total_duration": total_duration,
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
    validate_durations: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
        transcriber_provider: ASR provider to use (e.g., granite, remote)
        config: VAD block building configuration
        intermediate_dir: Path for intermediate/debug files
        models_dir: Path to model cache directory
        validate_durations: Whether to validate audio file durations match
        progress_callback: Optional callback for progress updates.
                          Called as: progress_callback(current_block, total_blocks, speaker_id)
                          This enables web UI to show per-block transcription progress.
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
    vad_dir: Optional[Path] = None
    if intermediate_dir:
        intermediate_dir = Path(intermediate_dir)
        vad_dir = intermediate_dir / "vad"
        vad_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize VAD provider
    vad_provider = SileroVADProvider(models_dir=models_dir)
    
    # 1. Run VAD on each speaker's audio
    log_status("Running VAD on speaker audio files")
    all_vad_segments: Dict[str, List[VADSegment]] = {}
    speaker_durations: Dict[str, float] = {}
    
    for speaker_id, audio_path in speaker_audio_files.items():
        log_progress(f"Running VAD on {speaker_id}")
        segments = vad_provider.detect_speech(audio_path, speaker_id)
        all_vad_segments[speaker_id] = segments
        speaker_durations[speaker_id] = vad_provider.get_audio_duration(audio_path)
    
    # Write VAD audit
    if vad_dir is not None:
        write_vad_audit(
            all_vad_segments,
            vad_dir,
            run_id,
            speaker_audio_files=speaker_audio_files,
            speaker_durations=speaker_durations,
            vad_config=vad_provider.config,
        )
    
    # 2. Build blocks from VAD segments
    log_status("Building conversation blocks from VAD segments")
    block_config = config or VADBlockBuilderConfig()
    block_builder = VADBlockBuilder(config=block_config)
    blocks = block_builder.build_blocks(all_vad_segments)
    
    # Save intermediate VAD blocks (before ASR processing)
    if vad_dir is not None:
        write_turn_building_audit(
            blocks,
            vad_dir,
            run_id,
            config=block_config.to_dict(),
            asr_chunk_data=None,  # No ASR chunking data yet
        )
        log_intermediate_save(str(vad_dir / f"turn_building_audit_{run_id}.json"), "Intermediate VAD blocks saved to")
    
    # 3. Process blocks through ASR
    log_status(f"Transcribing {len(blocks)} blocks")
    asr_processor = VADASRProcessor(
        transcriber_provider=transcriber_provider,
        models_dir=models_dir,
        intermediate_dir=intermediate_dir,
    )
    
    blocks = asr_processor.process_blocks(
        blocks,
        speaker_audio_files,
        progress_callback=progress_callback,
        **kwargs
    )
    
    # Write turn building audit
    if vad_dir is not None:
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
