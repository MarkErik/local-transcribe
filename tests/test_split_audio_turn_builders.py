#!/usr/bin/env python3
"""
Test script for split audio turn builders.

This script tests both the rule-based and LLM-enhanced turn builders
with sample word segment data.
"""

import json
from pathlib import Path
from datetime import datetime

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.providers.turn_builders.split_audio_turn_builder import SplitAudioTurnBuilderProvider
from local_transcribe.providers.turn_builders.split_audio_llm_turn_builder import SplitAudioLLMTurnBuilderProvider


def create_test_words() -> list:
    """Create realistic test data simulating an interview."""
    words = [
        # Interviewer asks a question (0-5 seconds)
        WordSegment(text="so", start=0.0, end=0.2, speaker="Interviewer"),
        WordSegment(text="tell", start=0.3, end=0.5, speaker="Interviewer"),
        WordSegment(text="me", start=0.5, end=0.6, speaker="Interviewer"),
        WordSegment(text="about", start=0.7, end=0.9, speaker="Interviewer"),
        WordSegment(text="your", start=1.0, end=1.2, speaker="Interviewer"),
        WordSegment(text="experience", start=1.3, end=1.8, speaker="Interviewer"),
        WordSegment(text="with", start=1.9, end=2.1, speaker="Interviewer"),
        WordSegment(text="online", start=2.2, end=2.5, speaker="Interviewer"),
        WordSegment(text="communities", start=2.6, end=3.2, speaker="Interviewer"),
        
        # Participant starts responding (5-15 seconds)
        WordSegment(text="yeah", start=5.0, end=5.2, speaker="Participant"),
        WordSegment(text="so", start=5.3, end=5.4, speaker="Participant"),
        WordSegment(text="I've", start=5.5, end=5.7, speaker="Participant"),
        WordSegment(text="been", start=5.8, end=5.9, speaker="Participant"),
        WordSegment(text="involved", start=6.0, end=6.4, speaker="Participant"),
        WordSegment(text="in", start=6.5, end=6.6, speaker="Participant"),
        WordSegment(text="online", start=6.7, end=7.0, speaker="Participant"),
        WordSegment(text="spaces", start=7.1, end=7.4, speaker="Participant"),
        WordSegment(text="for", start=7.5, end=7.6, speaker="Participant"),
        WordSegment(text="about", start=7.7, end=7.9, speaker="Participant"),
        WordSegment(text="ten", start=8.0, end=8.2, speaker="Participant"),
        WordSegment(text="years", start=8.3, end=8.6, speaker="Participant"),
        WordSegment(text="now", start=8.7, end=8.9, speaker="Participant"),
        
        # Interviewer acknowledgment (interjection) - 9 seconds
        WordSegment(text="mm-hmm", start=9.0, end=9.2, speaker="Interviewer"),
        
        # Participant continues (9.5-18 seconds)
        WordSegment(text="and", start=9.5, end=9.6, speaker="Participant"),
        WordSegment(text="it's", start=9.7, end=9.9, speaker="Participant"),
        WordSegment(text="really", start=10.0, end=10.3, speaker="Participant"),
        WordSegment(text="shaped", start=10.4, end=10.7, speaker="Participant"),
        WordSegment(text="how", start=10.8, end=10.9, speaker="Participant"),
        WordSegment(text="I", start=11.0, end=11.1, speaker="Participant"),
        WordSegment(text="think", start=11.2, end=11.4, speaker="Participant"),
        WordSegment(text="about", start=11.5, end=11.7, speaker="Participant"),
        WordSegment(text="community", start=11.8, end=12.3, speaker="Participant"),
        WordSegment(text="building", start=12.4, end=12.8, speaker="Participant"),
        
        # Interviewer reaction (interjection) - 13 seconds
        WordSegment(text="interesting", start=13.0, end=13.5, speaker="Interviewer"),
        
        # Participant continues (14-22 seconds)
        WordSegment(text="I", start=14.0, end=14.1, speaker="Participant"),
        WordSegment(text="think", start=14.2, end=14.4, speaker="Participant"),
        WordSegment(text="people", start=14.5, end=14.8, speaker="Participant"),
        WordSegment(text="underestimate", start=14.9, end=15.5, speaker="Participant"),
        WordSegment(text="how", start=15.6, end=15.7, speaker="Participant"),
        WordSegment(text="meaningful", start=15.8, end=16.3, speaker="Participant"),
        WordSegment(text="these", start=16.4, end=16.6, speaker="Participant"),
        WordSegment(text="connections", start=16.7, end=17.2, speaker="Participant"),
        WordSegment(text="can", start=17.3, end=17.4, speaker="Participant"),
        WordSegment(text="be", start=17.5, end=17.6, speaker="Participant"),
        
        # Interviewer agreement (interjection) - 18 seconds
        WordSegment(text="yeah", start=18.0, end=18.1, speaker="Interviewer"),
        
        # Participant finishes thought (19-25 seconds)
        WordSegment(text="especially", start=19.0, end=19.5, speaker="Participant"),
        WordSegment(text="during", start=19.6, end=19.9, speaker="Participant"),
        WordSegment(text="the", start=20.0, end=20.1, speaker="Participant"),
        WordSegment(text="pandemic", start=20.2, end=20.7, speaker="Participant"),
        WordSegment(text="when", start=20.8, end=21.0, speaker="Participant"),
        WordSegment(text="physical", start=21.1, end=21.5, speaker="Participant"),
        WordSegment(text="interactions", start=21.6, end=22.2, speaker="Participant"),
        WordSegment(text="were", start=22.3, end=22.5, speaker="Participant"),
        WordSegment(text="limited", start=22.6, end=23.0, speaker="Participant"),
        
        # Interviewer follow-up question (26-32 seconds) - substantive turn
        WordSegment(text="and", start=26.0, end=26.1, speaker="Interviewer"),
        WordSegment(text="what", start=26.2, end=26.4, speaker="Interviewer"),
        WordSegment(text="platforms", start=26.5, end=27.0, speaker="Interviewer"),
        WordSegment(text="were", start=27.1, end=27.3, speaker="Interviewer"),
        WordSegment(text="you", start=27.4, end=27.5, speaker="Interviewer"),
        WordSegment(text="most", start=27.6, end=27.8, speaker="Interviewer"),
        WordSegment(text="active", start=27.9, end=28.3, speaker="Interviewer"),
        WordSegment(text="on", start=28.4, end=28.5, speaker="Interviewer"),
        
        # Participant responds (33-40 seconds)
        WordSegment(text="primarily", start=33.0, end=33.5, speaker="Participant"),
        WordSegment(text="Discord", start=33.6, end=34.0, speaker="Participant"),
        WordSegment(text="and", start=34.1, end=34.2, speaker="Participant"),
        WordSegment(text="Reddit", start=34.3, end=34.7, speaker="Participant"),
    ]
    
    return words


def test_rule_based_builder():
    """Test the rule-based turn builder."""
    print("\n" + "=" * 60)
    print("Testing Rule-Based Turn Builder")
    print("=" * 60)
    
    words = create_test_words()
    builder = SplitAudioTurnBuilderProvider()
    
    turns = builder.build_turns(words)
    
    print(f"\nResults: {len(turns)} turns")
    print("-" * 40)
    
    for i, turn in enumerate(turns, 1):
        print(f"\nTurn {i}:")
        print(f"  Speaker: {turn.speaker}")
        print(f"  Time: {turn.start:.1f}s - {turn.end:.1f}s")
        print(f"  Text: {turn.text[:80]}{'...' if len(turn.text) > 80 else ''}")
    
    return turns


def test_llm_builder():
    """Test the LLM-enhanced turn builder."""
    print("\n" + "=" * 60)
    print("Testing LLM-Enhanced Turn Builder")
    print("=" * 60)
    
    words = create_test_words()
    builder = SplitAudioLLMTurnBuilderProvider()
    
    # Use the specified LLM endpoint
    turns = builder.build_turns(words, llm_url="http://100.84.208.72:8080")
    
    print(f"\nResults: {len(turns)} turns")
    print("-" * 40)
    
    for i, turn in enumerate(turns, 1):
        print(f"\nTurn {i}:")
        print(f"  Speaker: {turn.speaker}")
        print(f"  Time: {turn.start:.1f}s - {turn.end:.1f}s")
        print(f"  Text: {turn.text[:80]}{'...' if len(turn.text) > 80 else ''}")
    
    return turns


def test_with_interjections():
    """Test with interjections included in output."""
    print("\n" + "=" * 60)
    print("Testing with Interjections in Output")
    print("=" * 60)
    
    words = create_test_words()
    builder = SplitAudioTurnBuilderProvider()
    
    turns = builder.build_turns(words, include_interjections_in_output=True)
    
    print(f"\nResults: {len(turns)} turns (including interjections)")
    print("-" * 40)
    
    for i, turn in enumerate(turns, 1):
        # Detect likely interjections by short duration
        duration = turn.end - turn.start
        is_short = duration < 1.0 and len(turn.text.split()) <= 3
        marker = " [INTERJECTION]" if is_short else ""
        
        print(f"\nTurn {i}{marker}:")
        print(f"  Speaker: {turn.speaker}")
        print(f"  Time: {turn.start:.1f}s - {turn.end:.1f}s ({duration:.1f}s)")
        print(f"  Text: {turn.text}")
    
    return turns


def compare_builders():
    """Compare output from both builders."""
    print("\n" + "=" * 60)
    print("Comparing Rule-Based vs LLM-Enhanced Builders")
    print("=" * 60)
    
    words = create_test_words()
    
    rule_builder = SplitAudioTurnBuilderProvider()
    llm_builder = SplitAudioLLMTurnBuilderProvider()
    
    rule_turns = rule_builder.build_turns(words)
    llm_turns = llm_builder.build_turns(words, llm_url="http://100.84.208.72:8080")
    
    print(f"\nRule-based: {len(rule_turns)} turns")
    print(f"LLM-enhanced: {len(llm_turns)} turns")
    
    if len(rule_turns) == len(llm_turns):
        print("\nBoth builders produced the same number of turns")
        
        # Compare content
        differences = 0
        for i, (r, l) in enumerate(zip(rule_turns, llm_turns)):
            if r.text != l.text or r.speaker != l.speaker:
                differences += 1
                print(f"\nDifference in turn {i+1}:")
                print(f"  Rule: [{r.speaker}] {r.text[:60]}...")
                print(f"  LLM:  [{l.speaker}] {l.text[:60]}...")
        
        if differences == 0:
            print("Turn content is identical")
    else:
        print("\nBuilders produced different number of turns - detailed comparison:")
        print("\nRule-based turns:")
        for i, t in enumerate(rule_turns, 1):
            print(f"  {i}. [{t.speaker}] {t.text[:60]}...")
        
        print("\nLLM-enhanced turns:")
        for i, t in enumerate(llm_turns, 1):
            print(f"  {i}. [{t.speaker}] {t.text[:60]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Split Audio Turn Builder Tests")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Run tests
    test_rule_based_builder()
    test_with_interjections()
    
    # Test LLM builder (requires network)
    print("\nNote: LLM tests require connection to http://100.84.208.72:8080")
    try:
        test_llm_builder()
        compare_builders()
    except Exception as e:
        print(f"\nLLM test skipped due to error: {e}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
