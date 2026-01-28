#!/usr/bin/env python3
"""Test script for the new deserialization and checkpoint loading functions."""

import tempfile
import json
import os
import sys

from local_transcribe.processing.turn_building.turn_building_data_structures import (
    TranscriptFlow, HierarchicalTurn, InterjectionSegment
)
from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.framework.checkpoint_loader import (
    detect_checkpoint_type,
    load_transcript_flow_checkpoint,
    CheckpointValidationError
)


def test_interjection_segment_roundtrip():
    """Test InterjectionSegment serialization round-trip."""
    print("Testing InterjectionSegment round-trip...")
    
    ij = InterjectionSegment(
        speaker='Participant',
        start=10.5,
        end=11.0,
        text='Yeah',
        words=[WordSegment('Yeah', 10.5, 11.0, 'Participant')],
        likely_diarization_error=False
    )
    
    ij_dict = ij.to_dict()
    # Add words back (not in to_dict output)
    ij_dict['words'] = [{'text': 'Yeah', 'start': 10.5, 'end': 11.0, 'speaker': 'Participant'}]
    
    ij2 = InterjectionSegment.from_dict(ij_dict)
    
    assert ij2.speaker == ij.speaker
    assert ij2.start == ij.start
    assert ij2.end == ij.end
    assert ij2.text == ij.text
    print("  ✓ InterjectionSegment round-trip passed")


def test_hierarchical_turn_roundtrip():
    """Test HierarchicalTurn serialization round-trip."""
    print("Testing HierarchicalTurn round-trip...")
    
    ij = InterjectionSegment(
        speaker='Participant',
        start=2.5,
        end=3.0,
        text='Uh-huh',
        words=[WordSegment('Uh-huh', 2.5, 3.0, 'Participant')],
    )
    
    turn = HierarchicalTurn(
        turn_id=1,
        primary_speaker='Interviewer',
        start=0.0,
        end=5.0,
        text='Hello how are you',
        words=[
            WordSegment('Hello', 0.0, 1.0, 'Interviewer'),
            WordSegment('how', 1.0, 2.0, 'Interviewer'),
            WordSegment('are', 2.0, 3.0, 'Interviewer'),
            WordSegment('you', 3.0, 4.0, 'Interviewer'),
        ],
        interjections=[ij],
        source_block_ids=[1, 2, 3]
    )
    
    turn_dict = turn.to_dict()
    
    # Verify source_block_ids is serialized
    assert 'source_block_ids' in turn_dict
    assert turn_dict['source_block_ids'] == [1, 2, 3]
    
    # Add words back for deserialization
    turn_dict['words'] = [
        {'text': 'Hello', 'start': 0.0, 'end': 1.0, 'speaker': 'Interviewer'},
        {'text': 'how', 'start': 1.0, 'end': 2.0, 'speaker': 'Interviewer'},
        {'text': 'are', 'start': 2.0, 'end': 3.0, 'speaker': 'Interviewer'},
        {'text': 'you', 'start': 3.0, 'end': 4.0, 'speaker': 'Interviewer'},
    ]
    turn_dict['interjections'][0]['words'] = [
        {'text': 'Uh-huh', 'start': 2.5, 'end': 3.0, 'speaker': 'Participant'}
    ]
    
    turn2 = HierarchicalTurn.from_dict(turn_dict)
    
    assert turn2.turn_id == turn.turn_id
    assert turn2.primary_speaker == turn.primary_speaker
    assert turn2.start == turn.start
    assert turn2.end == turn.end
    assert turn2.text == turn.text
    assert turn2.source_block_ids == [1, 2, 3]
    assert len(turn2.interjections) == 1
    print("  ✓ HierarchicalTurn round-trip passed")


def test_transcript_flow_roundtrip():
    """Test TranscriptFlow serialization round-trip."""
    print("Testing TranscriptFlow round-trip...")
    
    turn = HierarchicalTurn(
        turn_id=0,
        primary_speaker='Interviewer',
        start=0.0,
        end=5.0,
        text='Hello how are you',
        words=[WordSegment('Hello', 0.0, 1.0, 'Interviewer')],
        interjections=[],
        source_block_ids=[1]
    )
    
    tf = TranscriptFlow(
        turns=[turn],
        metadata={'mode': 'vad_split_audio'},
        conversation_metrics={'total_duration': 60.0},
        speaker_statistics={'Interviewer': {'word_count': 4}}
    )
    
    tf_dict = tf.to_dict()
    
    # Add words back for deserialization
    tf_dict['turns'][0]['words'] = [
        {'text': 'Hello', 'start': 0.0, 'end': 1.0, 'speaker': 'Interviewer'}
    ]
    
    tf2 = TranscriptFlow.from_dict(tf_dict)
    
    assert len(tf2.turns) == len(tf.turns)
    assert tf2.turns[0].primary_speaker == tf.turns[0].primary_speaker
    assert tf2.turns[0].source_block_ids == [1]
    assert tf2.metadata['mode'] == 'vad_split_audio'
    print("  ✓ TranscriptFlow round-trip passed")


def test_detect_checkpoint_type():
    """Test checkpoint type detection."""
    print("Testing checkpoint type detection...")
    
    # Test TranscriptFlow format
    tf_data = {'turns': [], 'metadata': {}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(tf_data, f)
        tf_path = f.name
    
    try:
        assert detect_checkpoint_type(tf_path) == 'transcript_flow'
        print("  ✓ Detected transcript_flow format")
    finally:
        os.unlink(tf_path)
    
    # Test word_segments format
    ws_data = {'words': [], 'metadata': {}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ws_data, f)
        ws_path = f.name
    
    try:
        assert detect_checkpoint_type(ws_path) == 'word_segments'
        print("  ✓ Detected word_segments format")
    finally:
        os.unlink(ws_path)


def test_load_transcript_flow_checkpoint():
    """Test loading a TranscriptFlow checkpoint."""
    print("Testing TranscriptFlow checkpoint loading...")
    
    test_data = {
        'metadata': {'mode': 'vad_split_audio'},
        'conversation_metrics': {'total_duration': 60.0},
        'speaker_statistics': {},
        'turns': [
            {
                'turn_id': 0,
                'primary_speaker': 'Interviewer',
                'start': 0.0,
                'end': 5.0,
                'text': 'Hello how are you',
                'word_count': 4,
                'duration': 5.0,
                'speaking_rate': 48.0,
                'interjections': [],
                'words': [
                    {'text': 'Hello', 'start': 0.0, 'end': 1.0, 'speaker': 'Interviewer'},
                ],
                'source_block_ids': [1, 2]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        result = load_transcript_flow_checkpoint(temp_path)
        
        assert result.total_turns == 1
        assert 'Interviewer' in result.speakers_found
        assert result.transcript.turns[0].source_block_ids == [1, 2]
        print(f"  ✓ Loaded checkpoint with {result.total_turns} turns")
        print(f"  ✓ Source block IDs preserved: {result.transcript.turns[0].source_block_ids}")
    finally:
        os.unlink(temp_path)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Web Interface Pre-Implementation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_interjection_segment_roundtrip()
        test_hierarchical_turn_roundtrip()
        test_transcript_flow_roundtrip()
        test_detect_checkpoint_type()
        test_load_transcript_flow_checkpoint()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60 + "\n")
        return 0
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
