#!/usr/bin/env python3
"""
Speaker mapping utilities for pipeline processing.

This module provides functions for detecting, mapping, and applying
speaker labels to word segments.
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict

from local_transcribe.framework.plugin_interfaces import WordSegment


def detect_speakers_in_segments(segments: List[WordSegment]) -> List[str]:
    """
    Extract unique speaker labels from segments.
    
    Args:
        segments: List of WordSegment objects
        
    Returns:
        Sorted list of unique speaker labels
    """
    speakers = set()
    for segment in segments:
        if segment.speaker:
            speakers.add(segment.speaker)
    return sorted(speakers)


def get_speaker_word_counts(segments: List[WordSegment]) -> Dict[str, int]:
    """
    Count words per speaker.
    
    Args:
        segments: List of WordSegment objects
        
    Returns:
        Dictionary mapping speaker labels to word counts
    """
    counts = defaultdict(int)
    for segment in segments:
        speaker = segment.speaker or "(no speaker)"
        counts[speaker] += 1
    return dict(counts)


def get_speaker_samples(
    segments: List[WordSegment],
    words_per_sample: int = 30,
    samples_per_speaker: int = 3
) -> Dict[str, List[str]]:
    """
    Get sample text excerpts for each speaker.
    
    Args:
        segments: List of WordSegment objects
        words_per_sample: Number of words per sample excerpt
        samples_per_speaker: Maximum number of samples per speaker
        
    Returns:
        Dictionary mapping speaker labels to list of sample text strings
    """
    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for segment in segments:
        speaker = segment.speaker or "(no speaker)"
        speaker_segments[speaker].append(segment)
    
    samples = {}
    for speaker, segs in speaker_segments.items():
        speaker_samples = []
        
        # Get samples from beginning, middle, and end
        total = len(segs)
        if total == 0:
            continue
        
        # Calculate sample positions
        positions = [0]  # Start
        if total > words_per_sample * 2:
            positions.append(total // 2)  # Middle
        if total > words_per_sample * 3:
            positions.append(total - words_per_sample)  # End
        
        for pos in positions[:samples_per_speaker]:
            end_pos = min(pos + words_per_sample, total)
            sample_words = [s.text for s in segs[pos:end_pos]]
            sample_text = ' '.join(sample_words)
            speaker_samples.append(sample_text)
        
        samples[speaker] = speaker_samples
    
    return samples


def suggest_speaker_name(speaker_id: str, position: int, mode: str) -> str:
    """
    Suggest a default speaker name based on context.
    
    Args:
        speaker_id: The original speaker ID (e.g., "SPEAKER_00")
        position: The position/index of this speaker (0-based)
        mode: The pipeline mode ("combined_audio" or "split_audio")
        
    Returns:
        Suggested name string
    """
    # For two-speaker scenarios, suggest Interviewer/Participant
    if position == 0:
        return "Interviewer"
    elif position == 1:
        return "Participant"
    else:
        return f"Speaker {position + 1}"


def create_speaker_mapping_interactive(
    segments: List[WordSegment],
    mode: str,
    show_samples: bool = True
) -> Dict[str, str]:
    """
    Interactive prompt to map speaker IDs to names.
    
    Shows sample text from each speaker to help identification,
    then prompts the user to enter names for each speaker.
    
    Args:
        segments: List of WordSegment objects with speaker assignments
        mode: Pipeline mode ("combined_audio" or "split_audio")
        show_samples: Whether to show sample text excerpts
        
    Returns:
        Dictionary mapping original speaker IDs to new names
    """
    speakers = detect_speakers_in_segments(segments)
    
    if not speakers:
        print("No speakers detected in segments.")
        return {}
    
    word_counts = get_speaker_word_counts(segments)
    
    print("\n" + "=" * 60)
    print("SPEAKER NAME ASSIGNMENT")
    print("=" * 60)
    print(f"\nDetected {len(speakers)} speakers in the transcript.")
    print("You can assign human-readable names to each speaker ID.")
    print("Press Enter to accept the suggested default, or type a new name.")
    print("Type 'skip' to keep the original speaker ID.")
    
    if show_samples:
        samples = get_speaker_samples(segments)
        print("\n" + "-" * 40)
        print("SPEAKER SAMPLES")
        print("-" * 40)
        
        for speaker in speakers:
            count = word_counts.get(speaker, 0)
            print(f"\n🎤 {speaker} ({count:,} words):")
            if speaker in samples:
                for i, sample in enumerate(samples[speaker], 1):
                    # Truncate long samples
                    if len(sample) > 100:
                        sample = sample[:100] + "..."
                    print(f"   Sample {i}: \"{sample}\"")
    
    # Collect mappings
    mapping = {}
    print("\n" + "-" * 40)
    print("ENTER SPEAKER NAMES")
    print("-" * 40)
    
    for i, speaker in enumerate(speakers):
        suggestion = suggest_speaker_name(speaker, i, mode)
        
        while True:
            prompt = f"\nName for '{speaker}' [Default: {suggestion}]: "
            user_input = input(prompt).strip()
            
            if not user_input:
                # Accept default
                mapping[speaker] = suggestion
                print(f"  ✓ {speaker} → {suggestion}")
                break
            elif user_input.lower() == 'skip':
                # Keep original
                mapping[speaker] = speaker
                print(f"  ✓ {speaker} → {speaker} (kept original)")
                break
            else:
                # Use user's input
                mapping[speaker] = user_input
                print(f"  ✓ {speaker} → {user_input}")
                break
    
    print("\n" + "-" * 40)
    print("MAPPING SUMMARY")
    print("-" * 40)
    for original, new_name in mapping.items():
        arrow = "→" if original != new_name else "="
        print(f"  {original} {arrow} {new_name}")
    
    return mapping


def create_speaker_mapping_from_args(
    segments: List[WordSegment],
    speaker_map_arg: Optional[str]
) -> Dict[str, str]:
    """
    Create speaker mapping from command-line argument.
    
    Args:
        segments: List of WordSegment objects
        speaker_map_arg: Comma-separated mapping string (e.g., "SPEAKER_00=Interviewer,SPEAKER_01=Participant")
        
    Returns:
        Dictionary mapping original speaker IDs to new names
    """
    if not speaker_map_arg:
        return {}
    
    mapping = {}
    pairs = speaker_map_arg.split(',')
    
    for pair in pairs:
        if '=' not in pair:
            print(f"Warning: Invalid speaker mapping '{pair}', expected format 'ID=Name'")
            continue
        
        parts = pair.split('=', 1)
        if len(parts) == 2:
            speaker_id = parts[0].strip()
            name = parts[1].strip()
            if speaker_id and name:
                mapping[speaker_id] = name
    
    return mapping


def apply_speaker_mapping(
    segments: List[WordSegment],
    mapping: Dict[str, str]
) -> List[WordSegment]:
    """
    Apply speaker name mapping to segments.
    
    Creates new WordSegment objects with updated speaker names.
    
    Args:
        segments: Original list of WordSegment objects
        mapping: Dictionary mapping original speaker IDs to new names
        
    Returns:
        New list of WordSegment objects with updated speakers
    """
    if not mapping:
        return segments
    
    updated = []
    for segment in segments:
        new_speaker = mapping.get(segment.speaker, segment.speaker)
        updated.append(WordSegment(
            text=segment.text,
            start=segment.start,
            end=segment.end,
            speaker=new_speaker
        ))
    
    return updated


def validate_speaker_mapping(
    segments: List[WordSegment],
    mapping: Dict[str, str]
) -> List[str]:
    """
    Validate a speaker mapping against the segments.
    
    Args:
        segments: List of WordSegment objects
        mapping: Proposed speaker mapping
        
    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []
    
    actual_speakers = set(detect_speakers_in_segments(segments))
    mapped_speakers = set(mapping.keys())
    
    # Check for unmapped speakers
    unmapped = actual_speakers - mapped_speakers
    if unmapped:
        warnings.append(f"Speakers not in mapping will keep original names: {', '.join(sorted(unmapped))}")
    
    # Check for extra mappings
    extra = mapped_speakers - actual_speakers
    if extra:
        warnings.append(f"Mapping contains speakers not in segments: {', '.join(sorted(extra))}")
    
    # Check for duplicate target names
    target_names = list(mapping.values())
    duplicates = [name for name in set(target_names) if target_names.count(name) > 1]
    if duplicates:
        warnings.append(f"Multiple speakers mapped to same name: {', '.join(duplicates)}")
    
    return warnings
