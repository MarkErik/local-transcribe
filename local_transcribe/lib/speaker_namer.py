#!/usr/bin/env python3
# lib/speaker_namer.py - Interactive speaker naming for combined audio mode

from typing import List
from local_transcribe.framework.plugin_interfaces import Turn

def assign_speaker_names(turns: List[Turn], is_interactive: bool, mode: str) -> List[Turn]:
    """
    Assign custom names to speakers in combined audio mode if interactive.
    
    Args:
        turns: List of Turn objects with speaker, start, end, text attributes
        is_interactive: Whether running in interactive mode
        mode: 'combined_audio' or 'split_audio'
    
    Returns:
        Updated turns with speaker names replaced
    """
    if not is_interactive or mode != "combined_audio":
        return turns
    
    # Collect unique speaker labels
    speaker_labels = set()
    for turn in turns:
        speaker_labels.add(turn.speaker)
    
    speaker_labels = sorted(speaker_labels)
    
    if not speaker_labels:
        return turns
    
    # Show preview of first few turns
    print("\n=== Speaker Naming ===")
    print("Detected speakers in the audio. Here's a preview of the conversation:")
    
    preview_turns = turns[:10]  # First 10 turns
    for i, turn in enumerate(preview_turns, 1):
        print(f"{i}. {turn.speaker}: {turn.text[:50]}{'...' if len(turn.text) > 50 else ''}")
    
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
    
    # Replace speaker labels in turns
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