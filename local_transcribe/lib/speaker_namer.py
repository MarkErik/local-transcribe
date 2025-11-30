#!/usr/bin/env python3
# lib/speaker_namer.py - Interactive speaker naming for combined audio mode

from typing import List, Any
from local_transcribe.framework.plugin_interfaces import Turn

def assign_speaker_names(transcript: Any, is_interactive: bool, mode: str) -> Any:
    """
    Assign custom names to speakers in combined audio mode if interactive.
    
    Args:
        transcript: TranscriptFlow or List of Turn objects with speaker, start, end, text attributes
        is_interactive: Whether running in interactive mode
        mode: 'combined_audio' or 'split_audio'
    
    Returns:
        Updated transcript with speaker names replaced
    """
    if not is_interactive or mode != "combined_audio":
        return transcript
    
    # Handle TranscriptFlow
    is_transcript_flow = hasattr(transcript, 'turns') and hasattr(transcript, 'metadata')
    
    if is_transcript_flow:
        turns = transcript.turns
        # HierarchicalTurn uses primary_speaker
        speaker_attr = 'primary_speaker'
    else:
        turns = transcript
        speaker_attr = 'speaker'
    
    # Collect unique speaker labels
    speaker_labels = set()
    for turn in turns:
        speaker = getattr(turn, speaker_attr, None)
        if speaker:
            speaker_labels.add(speaker)
    
    speaker_labels = sorted(speaker_labels)
    
    if not speaker_labels:
        return transcript
    
    # Show preview of first few turns
    print("\n=== Speaker Naming ===")
    print("Detected speakers in the audio. Here's a preview of the conversation:")
    
    preview_turns = turns[:10]  # First 10 turns
    for i, turn in enumerate(preview_turns, 1):
        speaker = getattr(turn, speaker_attr, 'Unknown')
        text = turn.text[:50] if turn.text else ''
        suffix = '...' if turn.text and len(turn.text) > 50 else ''
        print(f"{i}. {speaker}: {text}{suffix}")
    
    print(f"\nFound {len(speaker_labels)} speakers: {', '.join(speaker_labels)}")
    print("Please assign names to each speaker:")
    
    # Prompt for names
    name_mapping = {}
    for label in speaker_labels:
        while True:
            name = input(f"Enter name for {label}: ").strip()
            if name:
                name_mapping[label] = name
                break
            else:
                print("Name cannot be empty. Please try again.")
    
    # Apply name mapping
    if is_transcript_flow:
        # Update HierarchicalTurn objects in place
        for turn in transcript.turns:
            if turn.primary_speaker in name_mapping:
                turn.primary_speaker = name_mapping[turn.primary_speaker]
            # Also update interjections
            for ij in turn.interjections:
                if ij.speaker in name_mapping:
                    ij.speaker = name_mapping[ij.speaker]
        
        print(f"\nSpeaker names assigned: {', '.join(f'{k} → {v}' for k, v in name_mapping.items())}")
        return transcript
    else:
        # Replace speaker labels in Turn objects
        updated_turns = []
        for turn in turns:
            updated_turn = Turn(
                speaker=name_mapping.get(turn.speaker, turn.speaker),
                start=turn.start,
                end=turn.end,
                text=turn.text
            )
            updated_turns.append(updated_turn)
        
        print(f"\nSpeaker names assigned: {', '.join(f'{k} → {v}' for k, v in name_mapping.items())}")
        return updated_turns