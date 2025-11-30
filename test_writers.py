#!/usr/bin/env python3
"""
Test script for the new hierarchical output writers.

This script loads a TranscriptFlow JSON file and generates all the new
output formats to verify they work correctly.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from local_transcribe.processing.turn_building.turn_building_data_structures import (
    TranscriptFlow,
    HierarchicalTurn,
    InterjectionSegment
)
from local_transcribe.framework.plugin_interfaces import WordSegment


def load_transcript_flow_from_json(json_path: Path) -> TranscriptFlow:
    """Load a TranscriptFlow from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct turns
    turns = []
    for turn_data in data.get('turns', []):
        # Reconstruct interjections
        interjections = []
        for ij_data in turn_data.get('interjections', []):
            ij = InterjectionSegment(
                speaker=ij_data.get('speaker', 'Unknown'),
                start=ij_data.get('start', 0),
                end=ij_data.get('end', 0),
                text=ij_data.get('text', ''),
                words=[],  # We don't have word data in JSON
                confidence=ij_data.get('confidence', 0),
                interjection_type=ij_data.get('interjection_type', 'unclear'),
                interrupt_level=ij_data.get('interrupt_level', 'none'),
                classification_method=ij_data.get('classification_method', 'rule'),
                likely_diarization_error=ij_data.get('likely_diarization_error', False)
            )
            interjections.append(ij)
        
        # Create turn using object.__new__ to bypass __post_init__
        # This preserves the pre-calculated metrics from JSON
        turn = object.__new__(HierarchicalTurn)
        turn.turn_id = turn_data.get('turn_id', 0)
        turn.primary_speaker = turn_data.get('primary_speaker', 'Unknown')
        turn.start = turn_data.get('start', 0)
        turn.end = turn_data.get('end', 0)
        turn.text = turn_data.get('text', '')
        turn.words = []  # We don't have word data in JSON
        turn.interjections = interjections
        turn.flow_continuity = turn_data.get('flow_continuity', 1.0)
        turn.turn_type = turn_data.get('turn_type', 'monologue')
        turn.word_count = turn_data.get('word_count', 0)
        turn.duration = turn_data.get('duration', 0)
        turn.speaking_rate = turn_data.get('speaking_rate', 0)
        
        turns.append(turn)
    
    # Create TranscriptFlow
    transcript_flow = TranscriptFlow(
        turns=turns,
        metadata=data.get('metadata', {}),
        conversation_metrics=data.get('conversation_metrics', {}),
        speaker_statistics=data.get('speaker_statistics', {})
    )
    
    return transcript_flow


def main():
    # Path to sample JSON
    sample_json = project_root / "samples" / "transcript_flow_output.json"
    
    if not sample_json.exists():
        print(f"Error: Sample file not found: {sample_json}")
        sys.exit(1)
    
    print(f"Loading transcript from: {sample_json}")
    transcript_flow = load_transcript_flow_from_json(sample_json)
    
    print(f"Loaded transcript with {len(transcript_flow.turns)} turns")
    print(f"Total interjections: {transcript_flow.total_interjections}")
    print(f"Speakers: {transcript_flow.speakers}")
    print()
    
    # Output directory
    output_dir = project_root / "samples" / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Import writers after loading data (to ensure proper registration)
    from local_transcribe.providers.file_writers.annotated_markdown_writer import write_annotated_markdown
    from local_transcribe.providers.file_writers.dialogue_script_writer import write_dialogue_script
    from local_transcribe.providers.file_writers.html_timeline_writer import write_html_timeline
    
    # Test Annotated Markdown
    print("Writing annotated markdown...")
    md_path = output_dir / "transcript.annotated.md"
    write_annotated_markdown(transcript_flow, md_path)
    print(f"  ✓ Written to: {md_path}")
    
    # Test Dialogue Script
    print("Writing dialogue script...")
    script_path = output_dir / "transcript.script.txt"
    write_dialogue_script(transcript_flow, script_path)
    print(f"  ✓ Written to: {script_path}")
    
    # Test HTML Timeline
    print("Writing HTML timeline...")
    html_path = output_dir / "transcript_timeline.html"
    write_html_timeline(transcript_flow, html_path)
    print(f"  ✓ Written to: {html_path}")
    
    print()
    print("=" * 60)
    print("All writers completed successfully!")
    print(f"Output files are in: {output_dir}")
    print("=" * 60)
    
    # Show a preview of each file
    print()
    print("PREVIEW: Annotated Markdown (first 50 lines)")
    print("-" * 60)
    with open(md_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:
                print("... (truncated)")
                break
            print(line.rstrip())
    
    print()
    print("PREVIEW: Dialogue Script (first 50 lines)")
    print("-" * 60)
    with open(script_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 50:
                print("... (truncated)")
                break
            print(line.rstrip())
    
    print()
    print(f"HTML Timeline written to: {html_path}")
    print("Open in a browser to view the interactive timeline.")


if __name__ == "__main__":
    main()
