#!/usr/bin/env python3
"""
Test script for the split audio turn builder using real transcript data.

This tests the full turn building pipeline with actual transcript data
from the transcript-sample directory.
"""

import json
import time
from pathlib import Path
from typing import List

# Add the project to path
import sys
sys.path.insert(0, '/Users/ai/ai-Dev/local-transcribe')

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.processing.turn_building.split_audio_llm_turn_builder import (
    SplitAudioTurnBuilder,
    build_turns_split_audio
)

# LLM endpoint for testing
LLM_URL = "http://100.84.208.72:8080"

# Transcript sample directory
SAMPLE_DIR = Path("/Users/ai/ai-Dev/local-transcribe/transcript-sample")


def load_word_segments(file_path: Path) -> List[WordSegment]:
    """Load word segments from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    for word_data in data.get('words', []):
        words.append(WordSegment(
            text=word_data['text'],
            start=word_data['start'],
            end=word_data['end'],
            speaker=word_data.get('speaker', 'Unknown')
        ))
    
    return words


def test_load_sample_data():
    """Test that we can load the sample transcript data."""
    print("\n" + "="*60)
    print("TEST 1: Load Sample Data")
    print("="*60)
    
    interviewer_file = SAMPLE_DIR / "interviewer_word_segments_deidentified.json"
    participant_file = SAMPLE_DIR / "participant_word_segments_deidentified.json"
    
    print(f"  Interviewer file: {interviewer_file.exists()}")
    print(f"  Participant file: {participant_file.exists()}")
    
    interviewer_words = load_word_segments(interviewer_file)
    participant_words = load_word_segments(participant_file)
    
    print(f"  Interviewer words: {len(interviewer_words)}")
    print(f"  Participant words: {len(participant_words)}")
    
    # Show sample of first few words
    print(f"\n  Sample interviewer words (first 5):")
    for w in interviewer_words[:5]:
        print(f"    [{w.start:.2f}-{w.end:.2f}] {w.speaker}: '{w.text}'")
    
    print(f"\n  Sample participant words (first 5):")
    for w in participant_words[:5]:
        print(f"    [{w.start:.2f}-{w.end:.2f}] {w.speaker}: '{w.text}'")
    
    return interviewer_words, participant_words


def test_merge_word_streams(interviewer_words: List[WordSegment], participant_words: List[WordSegment]):
    """Test merging word streams from both speakers."""
    print("\n" + "="*60)
    print("TEST 2: Merge Word Streams")
    print("="*60)
    
    from local_transcribe.processing.turn_building.base import merge_word_streams
    
    all_words = interviewer_words + participant_words
    print(f"  Combined words before merge: {len(all_words)}")
    
    merged = merge_word_streams(all_words)
    print(f"  Merged words: {len(merged)}")
    
    # Show the first 20 merged words to see timeline interleaving
    print(f"\n  First 20 merged words (timeline order):")
    for i, w in enumerate(merged[:20]):
        print(f"    {i+1:3d}. [{w.start:.2f}-{w.end:.2f}] {w.speaker}: '{w.text}'")
    
    return merged


def test_smart_grouping(merged_words: List[WordSegment]):
    """Test smart grouping with interjection detection."""
    print("\n" + "="*60)
    print("TEST 3: Smart Grouping with Interjection Detection")
    print("="*60)
    
    from local_transcribe.processing.turn_building.base import smart_group_with_interjection_detection
    from local_transcribe.processing.turn_building.data_structures import TurnBuilderConfig
    
    config = TurnBuilderConfig()
    
    # Test with a subset of words first (first 500)
    test_words = merged_words[:500]
    print(f"  Testing with first {len(test_words)} words")
    
    start_time = time.time()
    primary_segments, pending_interjections = smart_group_with_interjection_detection(
        test_words, config
    )
    elapsed = time.time() - start_time
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Primary segments: {len(primary_segments)}")
    print(f"  Pending interjections: {len(pending_interjections)}")
    
    # Show some primary segments
    print(f"\n  First 5 primary segments:")
    for i, seg in enumerate(primary_segments[:5]):
        preview = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        print(f"    {i+1}. [{seg.start:.2f}-{seg.end:.2f}] {seg.speaker} ({seg.word_count} words): '{preview}'")
    
    # Show pending interjections
    if pending_interjections:
        print(f"\n  First 5 pending interjections:")
        for i, ij in enumerate(pending_interjections[:5]):
            print(f"    {i+1}. [{ij.start:.2f}-{ij.end:.2f}] {ij.speaker} during {ij.detected_during_turn_of}'s turn: '{ij.text}'")
    
    return primary_segments, pending_interjections


def test_single_llm_verification():
    """Test a single LLM verification call with detailed timing."""
    print("\n" + "="*60)
    print("TEST 4: Single LLM Verification Call")
    print("="*60)
    
    from local_transcribe.processing.turn_building.base import PendingInterjection
    from local_transcribe.processing.turn_building.data_structures import RawSegment, TurnBuilderConfig
    
    builder = SplitAudioTurnBuilder(llm_url=LLM_URL)
    
    # Create a mock pending interjection
    mock_word = WordSegment(text="yeah", start=10.0, end=10.3, speaker="Interviewer")
    pending = PendingInterjection(
        speaker="Interviewer",
        words=[mock_word],
        start=10.0,
        end=10.3,
        detected_during_turn_of="Participant"
    )
    
    # Create mock context segments
    context_before = RawSegment(
        speaker="Participant",
        start=5.0,
        end=9.8,
        text="So I was working at the hospital during that time and it was really challenging",
        words=[]
    )
    
    context_after = RawSegment(
        speaker="Participant",
        start=10.5,
        end=15.0,
        text="and we had to work overtime just to keep up with all the patients",
        words=[]
    )
    
    print(f"  Testing single LLM verification...")
    print(f"  Pending interjection: '{pending.text}' by {pending.speaker}")
    print(f"  Context before: '{context_before.text[:50]}...'")
    print(f"  Context after: '{context_after.text[:50]}...'")
    
    start_time = time.time()
    result = builder._verify_with_llm(pending, context_before, context_after)
    elapsed = time.time() - start_time
    
    print(f"\n  Time: {elapsed:.2f}s")
    print(f"  Result: {json.dumps(result, indent=2) if result else 'None (failed)'}")
    
    return result


def test_full_turn_building_small():
    """Test full turn building with a small subset of data."""
    print("\n" + "="*60)
    print("TEST 5: Full Turn Building (Small Subset)")
    print("="*60)
    
    # Load both transcripts
    interviewer_words = load_word_segments(SAMPLE_DIR / "interviewer_word_segments_deidentified.json")
    participant_words = load_word_segments(SAMPLE_DIR / "participant_word_segments_deidentified.json")
    
    # Use a small subset for testing
    # Get words from approximately the first minute (up to timestamp 60)
    all_words = interviewer_words + participant_words
    test_words = [w for w in all_words if w.end <= 60.0]
    
    print(f"  Testing with {len(test_words)} words (first ~60 seconds)")
    
    builder = SplitAudioTurnBuilder(llm_url=LLM_URL)
    
    start_time = time.time()
    transcript_flow = builder.build_turns(
        test_words,
        llm_url=LLM_URL,
        llm_timeout=120
    )
    elapsed = time.time() - start_time
    
    print(f"\n  Time: {elapsed:.2f}s")
    print(f"  Total turns: {transcript_flow.total_turns}")
    print(f"  Total interjections: {transcript_flow.total_interjections}")
    print(f"  LLM stats: {builder.llm_stats}")
    
    # Show first few turns
    print(f"\n  First 3 turns:")
    for i, turn in enumerate(transcript_flow.turns[:3]):
        preview = turn.text[:60] + "..." if len(turn.text) > 60 else turn.text
        print(f"    Turn {turn.turn_id}: [{turn.start:.2f}-{turn.end:.2f}] {turn.primary_speaker} ({turn.word_count} words)")
        print(f"      Text: '{preview}'")
        if turn.interjections:
            for ij in turn.interjections:
                print(f"      Interjection: [{ij.start:.2f}] {ij.speaker}: '{ij.text}' ({ij.interjection_type})")
    
    return transcript_flow


def test_needs_llm_verification():
    """Test the _needs_llm_verification method."""
    print("\n" + "="*60)
    print("TEST 6: Needs LLM Verification Logic")
    print("="*60)
    
    from local_transcribe.processing.turn_building.base import PendingInterjection, detect_interjection_type
    from local_transcribe.processing.turn_building.data_structures import TurnBuilderConfig
    
    builder = SplitAudioTurnBuilder(llm_url=LLM_URL)
    config = TurnBuilderConfig()
    
    test_cases = [
        # (text, expected_needs_llm)
        ("yeah", False),  # 1 word, pattern match
        ("uh-huh", False),  # 1 word, pattern match
        ("okay", False),  # 1 word, pattern match
        ("right", False),  # 1 word, pattern match
        ("that's interesting", True),  # 2 words, no clear pattern
        ("so what happened next", True),  # 4 words
        ("really", False),  # 1 word, reaction pattern
        ("oh wow that's great", True),  # 4 words
        ("mhm", False),  # 1 word, pattern match
    ]
    
    for text, expected in test_cases:
        mock_word = WordSegment(text=text, start=0, end=0.5, speaker="Test")
        words = text.split()
        mock_words = [WordSegment(text=w, start=i*0.1, end=(i+1)*0.1, speaker="Test") for i, w in enumerate(words)]
        
        pending = PendingInterjection(
            speaker="Interviewer",
            words=mock_words,
            start=0,
            end=0.5,
            detected_during_turn_of="Participant"
        )
        
        needs_llm = builder._needs_llm_verification(pending)
        pattern_type = detect_interjection_type(text, config)
        
        status = "✓" if needs_llm == expected else "✗"
        print(f"  {status} '{text}' ({len(words)} words): needs_llm={needs_llm}, pattern={pattern_type}")


if __name__ == "__main__":
    print(f"Testing Split Audio Turn Builder with LLM: {LLM_URL}")
    print("="*60)
    
    # Run tests in order
    interviewer_words, participant_words = test_load_sample_data()
    
    merged_words = test_merge_word_streams(interviewer_words, participant_words)
    
    primary_segments, pending_interjections = test_smart_grouping(merged_words)
    
    test_needs_llm_verification()
    
    test_single_llm_verification()
    
    test_full_turn_building_small()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
