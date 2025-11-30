#!/usr/bin/env python3
"""
Checkpoint loading and validation for pipeline re-entry.

This module provides utilities for loading intermediate pipeline outputs
(checkpoints) and validating their format for re-entry into the pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from local_transcribe.framework.plugin_interfaces import WordSegment


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails."""
    pass


class CheckpointWarning:
    """Represents a non-fatal warning during checkpoint validation."""
    
    def __init__(self, message: str, severity: str = "warning"):
        self.message = message
        self.severity = severity  # "warning" or "info"
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.message}"


@dataclass
class CheckpointLoadResult:
    """Result of loading a checkpoint file."""
    segments: List[WordSegment]
    metadata: Dict[str, Any]
    warnings: List[CheckpointWarning]
    speakers_found: List[str]
    total_words: int
    duration_seconds: float


def load_diarized_checkpoint(json_path: Path) -> CheckpointLoadResult:
    """
    Load and validate a diarized word segments JSON file.
    
    Args:
        json_path: Path to the JSON checkpoint file
        
    Returns:
        CheckpointLoadResult with segments, metadata, and validation info
        
    Raises:
        CheckpointValidationError: If the file cannot be loaded or has fatal issues
    """
    path = Path(json_path)
    
    if not path.exists():
        raise CheckpointValidationError(f"Checkpoint file not found: {path}")
    
    if not path.suffix.lower() == '.json':
        raise CheckpointValidationError(f"Expected JSON file, got: {path.suffix}")
    
    # Load JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CheckpointValidationError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise CheckpointValidationError(f"Failed to read file: {e}")
    
    # Validate structure
    warnings = []
    
    # Extract metadata
    metadata = data.get('metadata', {})
    
    # Get words array
    if 'words' not in data:
        raise CheckpointValidationError("Missing 'words' array in checkpoint file")
    
    words_data = data['words']
    if not isinstance(words_data, list):
        raise CheckpointValidationError("'words' must be an array")
    
    if len(words_data) == 0:
        raise CheckpointValidationError("'words' array is empty")
    
    # Validate and convert word segments
    segments = []
    speakers_found = set()
    unknown_speakers = set()
    last_end_time = 0.0
    overlap_count = 0
    gap_count = 0
    
    for i, word_data in enumerate(words_data):
        # Validate required fields
        missing_fields = []
        for field in ['text', 'start', 'end']:
            if field not in word_data:
                missing_fields.append(field)
        
        if missing_fields:
            raise CheckpointValidationError(
                f"Word at index {i} missing required fields: {', '.join(missing_fields)}"
            )
        
        # Extract values
        text = word_data['text']
        start = float(word_data['start'])
        end = float(word_data['end'])
        speaker = word_data.get('speaker', None)
        
        # Track speaker
        if speaker:
            speakers_found.add(speaker)
            if speaker.lower() in ('unknown', 'unknown_speaker', ''):
                unknown_speakers.add(speaker)
        else:
            unknown_speakers.add('(no speaker)')
        
        # Check for timing issues
        if start > end:
            warnings.append(CheckpointWarning(
                f"Word '{text}' at index {i} has start ({start}) > end ({end})",
                "warning"
            ))
        
        if start < last_end_time:
            overlap_count += 1
        elif start > last_end_time + 1.0:  # Gap > 1 second
            gap_count += 1
        
        last_end_time = end
        
        # Create WordSegment
        segment = WordSegment(
            text=text,
            start=start,
            end=end,
            speaker=speaker
        )
        segments.append(segment)
    
    # Add summary warnings
    if overlap_count > 0:
        warnings.append(CheckpointWarning(
            f"Found {overlap_count} overlapping word segments",
            "info"
        ))
    
    if gap_count > 0:
        warnings.append(CheckpointWarning(
            f"Found {gap_count} gaps > 1 second between words",
            "info"
        ))
    
    if unknown_speakers:
        warnings.append(CheckpointWarning(
            f"Found words with unknown/missing speakers: {', '.join(sorted(unknown_speakers))}",
            "warning"
        ))
    
    # Calculate duration
    duration = segments[-1].end if segments else 0.0
    
    # Enrich metadata
    metadata['loaded_at'] = datetime.now().isoformat()
    metadata['source_file'] = str(path.absolute())
    
    return CheckpointLoadResult(
        segments=segments,
        metadata=metadata,
        warnings=warnings,
        speakers_found=sorted(speakers_found),
        total_words=len(segments),
        duration_seconds=duration
    )


def validate_checkpoint_for_reentry(
    result: CheckpointLoadResult,
    target_stage: str = "turn_building"
) -> Tuple[bool, List[str]]:
    """
    Validate that a checkpoint is suitable for re-entry at a given stage.
    
    Args:
        result: The loaded checkpoint result
        target_stage: The stage to re-enter at
        
    Returns:
        Tuple of (is_valid, list of error/warning messages)
    """
    messages = []
    is_valid = True
    
    # For turn_building, we need speaker assignments
    if target_stage == "turn_building":
        # Check if any segments have speakers
        segments_with_speakers = sum(1 for s in result.segments if s.speaker)
        
        if segments_with_speakers == 0:
            messages.append("ERROR: No segments have speaker assignments. Cannot proceed with turn building.")
            is_valid = False
        elif segments_with_speakers < len(result.segments):
            pct = (segments_with_speakers / len(result.segments)) * 100
            messages.append(f"WARNING: Only {pct:.1f}% of segments have speaker assignments.")
    
    # Add warnings from loading
    for warning in result.warnings:
        messages.append(str(warning))
    
    return is_valid, messages


def print_checkpoint_summary(result: CheckpointLoadResult) -> None:
    """Print a human-readable summary of the loaded checkpoint."""
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)
    
    print(f"\n📁 Source: {result.metadata.get('source_file', 'Unknown')}")
    print(f"📊 Total words: {result.total_words:,}")
    print(f"⏱️  Duration: {result.duration_seconds:.1f} seconds ({result.duration_seconds/60:.1f} minutes)")
    print(f"🎤 Speakers found: {', '.join(result.speakers_found) if result.speakers_found else 'None'}")
    
    # Show metadata if available
    if result.metadata.get('format_version'):
        print(f"📋 Format version: {result.metadata['format_version']}")
    if result.metadata.get('mode'):
        print(f"🔧 Mode: {result.metadata['mode']}")
    if result.metadata.get('source_audio'):
        print(f"🔊 Original audio: {result.metadata['source_audio']}")
    
    # Show sample text from each speaker
    print("\n" + "-" * 40)
    print("SAMPLE TEXT BY SPEAKER")
    print("-" * 40)
    
    speaker_samples = {}
    for segment in result.segments:
        speaker = segment.speaker or "(no speaker)"
        if speaker not in speaker_samples:
            speaker_samples[speaker] = []
        if len(speaker_samples[speaker]) < 20:  # Collect first 20 words
            speaker_samples[speaker].append(segment.text)
    
    for speaker, words in sorted(speaker_samples.items()):
        sample_text = ' '.join(words[:20])
        if len(words) >= 20:
            sample_text += "..."
        print(f"\n  {speaker}:")
        print(f"    \"{sample_text}\"")
    
    # Show warnings
    if result.warnings:
        print("\n" + "-" * 40)
        print("WARNINGS")
        print("-" * 40)
        for warning in result.warnings:
            print(f"  {warning}")
    
    print("\n" + "=" * 60)


def get_mode_from_checkpoint(result: CheckpointLoadResult) -> Optional[str]:
    """
    Attempt to determine the pipeline mode from checkpoint metadata.
    
    Returns:
        Mode string if determinable, None otherwise
    """
    # Check explicit metadata
    if result.metadata.get('mode'):
        return result.metadata['mode']
    
    # Try to infer from speaker patterns
    speakers = result.speakers_found
    
    # Diarization typically produces SPEAKER_00, SPEAKER_01, etc.
    diarization_pattern = all(
        s.startswith('SPEAKER_') or s.lower() in ('unknown', 'unknown_speaker')
        for s in speakers if s
    )
    
    if diarization_pattern:
        return "combined_audio"
    
    # Named speakers suggest split_audio mode
    named_speakers = ['interviewer', 'participant', 'speaker']
    has_named = any(
        any(name in s.lower() for name in named_speakers)
        for s in speakers if s
    )
    
    if has_named:
        return "split_audio"
    
    return None
