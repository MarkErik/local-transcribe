#!/usr/bin/env python3
"""
Tests for the edit applicator service.

This module tests the EditApplicator class which applies stored edits
to TranscriptFlow objects.
"""

import sys
import os

# Add parent directory to path for local_transcribe imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_api.services.edit_applicator import EditApplicator, apply_edits_to_transcript
from web_api.database import Edit


def create_test_transcript():
    """Create a sample transcript for testing."""
    return {
        'turns': [
            {
                'turn_id': 1,
                'primary_speaker': 'Interviewer',
                'speaker': 'Interviewer',
                'text': 'Hello how are you',
                'start_time': 0.0,
                'end_time': 2.0,
                'words': [
                    {'word': 'Hello', 'start_time': 0.0, 'end_time': 0.5, 'confidence': 0.95},
                    {'word': 'how', 'start_time': 0.5, 'end_time': 1.0, 'confidence': 0.90},
                    {'word': 'are', 'start_time': 1.0, 'end_time': 1.5, 'confidence': 0.92},
                    {'word': 'you', 'start_time': 1.5, 'end_time': 2.0, 'confidence': 0.98},
                ],
                'interjections': []
            },
            {
                'turn_id': 2,
                'primary_speaker': 'Participant',
                'speaker': 'Participant',
                'text': 'I am fine thanks',
                'start_time': 2.0,
                'end_time': 4.0,
                'words': [
                    {'word': 'I', 'start_time': 2.0, 'end_time': 2.25, 'confidence': 0.99},
                    {'word': 'am', 'start_time': 2.25, 'end_time': 2.5, 'confidence': 0.97},
                    {'word': 'fine', 'start_time': 2.5, 'end_time': 3.0, 'confidence': 0.85},
                    {'word': 'thanks', 'start_time': 3.0, 'end_time': 4.0, 'confidence': 0.93},
                ],
                'interjections': []
            },
            {
                'turn_id': 3,
                'primary_speaker': 'Interviewer',
                'speaker': 'Interviewer',
                'text': 'Thats great to hear',
                'start_time': 4.0,
                'end_time': 6.0,
                'words': [
                    {'word': 'Thats', 'start_time': 4.0, 'end_time': 4.5, 'confidence': 0.80},
                    {'word': 'great', 'start_time': 4.5, 'end_time': 5.0, 'confidence': 0.95},
                    {'word': 'to', 'start_time': 5.0, 'end_time': 5.3, 'confidence': 0.99},
                    {'word': 'hear', 'start_time': 5.3, 'end_time': 6.0, 'confidence': 0.94},
                ],
                'interjections': []
            },
        ]
    }


def test_word_change():
    """Test changing a word's text."""
    print("Testing word_change...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='word_change',
        turn_id=1, start_index=0, new_value='Hi'
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    assert result['turns'][0]['words'][0]['word'] == 'Hi', "word should be changed to 'Hi'"
    print("  ✓ word_change passed")


def test_word_insert():
    """Test inserting a new word."""
    print("Testing word_insert...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='word_insert',
        turn_id=1, start_index=1, new_value='there'
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    words = result['turns'][0]['words']
    assert len(words) == 5, f"should have 5 words, got {len(words)}"
    assert words[1]['word'] == 'there', f"inserted word should be 'there', got '{words[1]['word']}'"
    print("  ✓ word_insert passed")


def test_word_delete():
    """Test deleting a word."""
    print("Testing word_delete...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='word_delete',
        turn_id=1, start_index=1, end_index=1
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    words = result['turns'][0]['words']
    assert len(words) == 3, f"should have 3 words after delete, got {len(words)}"
    assert words[1]['word'] == 'are', f"second word should be 'are', got '{words[1]['word']}'"
    print("  ✓ word_delete passed")


def test_speaker_change():
    """Test changing the speaker of a turn."""
    print("Testing speaker_change...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='speaker_change',
        turn_id=2, new_value='Interviewer'
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    assert result['turns'][1]['speaker'] == 'Interviewer', "speaker should be changed"
    print("  ✓ speaker_change passed")


def test_merge_words():
    """Test merging consecutive words."""
    print("Testing merge_words...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='merge_words',
        turn_id=3, start_index=0, end_index=1  # Merge "Thats" and "great"
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    words = result['turns'][2]['words']
    assert len(words) == 3, f"should have 3 words after merge, got {len(words)}"
    assert words[0]['word'] == 'Thatsgreat', f"merged word should be 'Thatsgreat', got '{words[0]['word']}'"
    print("  ✓ merge_words passed")


def test_split_word():
    """Test splitting a word into multiple words."""
    print("Testing split_word...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='split_word',
        turn_id=3, start_index=0, new_value="That's"  # Split "Thats" into "That's" (one word)
    )
    
    # For proper split, need multiple words
    transcript['turns'][2]['words'][0]['word'] = 'cannot'
    edit.new_value = 'can not'
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    words = result['turns'][2]['words']
    assert len(words) == 5, f"should have 5 words after split, got {len(words)}"
    assert words[0]['word'] == 'can', f"first split word should be 'can', got '{words[0]['word']}'"
    assert words[1]['word'] == 'not', f"second split word should be 'not', got '{words[1]['word']}'"
    print("  ✓ split_word passed")


def test_turn_merge():
    """Test merging two turns."""
    print("Testing turn_merge...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='turn_merge',
        turn_id=1, target_turn_id=2  # Merge turn 1 with turn 2
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    assert len(result['turns']) == 2, f"should have 2 turns after merge, got {len(result['turns'])}"
    assert len(result['turns'][0]['words']) == 8, f"merged turn should have 8 words, got {len(result['turns'][0]['words'])}"
    print("  ✓ turn_merge passed")


def test_turn_split():
    """Test splitting a turn."""
    print("Testing turn_split...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='turn_split',
        turn_id=1, start_index=2  # Split at word index 2 ("are")
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    assert len(result['turns']) == 4, f"should have 4 turns after split, got {len(result['turns'])}"
    assert len(result['turns'][0]['words']) == 2, f"first part should have 2 words, got {len(result['turns'][0]['words'])}"
    assert len(result['turns'][1]['words']) == 2, f"second part should have 2 words, got {len(result['turns'][1]['words'])}"
    print("  ✓ turn_split passed")


def test_insert_annotation():
    """Test inserting an annotation marker."""
    print("Testing insert_annotation...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='insert_annotation',
        turn_id=2, start_index=2, annotation_type='laughter'
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    words = result['turns'][1]['words']
    assert len(words) == 5, f"should have 5 words after annotation, got {len(words)}"
    assert words[2]['word'] == '[laughter]', f"annotation should be '[laughter]', got '{words[2]['word']}'"
    assert words[2].get('is_annotation') == True, "annotation should be marked"
    print("  ✓ insert_annotation passed")


def test_multiple_edits():
    """Test applying multiple edits in sequence."""
    print("Testing multiple edits...")
    
    transcript = create_test_transcript()
    edits = [
        Edit(id=1, job_id='test', stage_name='test', edit_type='word_change',
             turn_id=1, start_index=0, new_value='Hi'),
        Edit(id=2, job_id='test', stage_name='test', edit_type='word_insert',
             turn_id=1, start_index=1, new_value='there'),
        Edit(id=3, job_id='test', stage_name='test', edit_type='speaker_change',
             turn_id=2, new_value='Unknown'),
    ]
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits(edits)
    
    assert result['turns'][0]['words'][0]['word'] == 'Hi'
    assert result['turns'][0]['words'][1]['word'] == 'there'
    assert result['turns'][1]['speaker'] == 'Unknown'
    print("  ✓ multiple edits passed")


def test_convenience_function():
    """Test the apply_edits_to_transcript convenience function."""
    print("Testing convenience function...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='word_change',
        turn_id=1, start_index=0, new_value='Hi'
    )
    
    result = apply_edits_to_transcript(transcript, [edit])
    assert result['turns'][0]['words'][0]['word'] == 'Hi'
    print("  ✓ convenience function passed")


def test_nonexistent_turn():
    """Test editing a turn that doesn't exist."""
    print("Testing nonexistent turn handling...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='word_change',
        turn_id=999, start_index=0, new_value='Hi'  # Turn 999 doesn't exist
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    # Should not raise an error, just skip the edit
    assert len(result['turns']) == 3, "transcript should be unchanged"
    print("  ✓ nonexistent turn handled gracefully")


def test_unknown_edit_type():
    """Test handling unknown edit types."""
    print("Testing unknown edit type handling...")
    
    transcript = create_test_transcript()
    edit = Edit(
        id=1, job_id='test', stage_name='test', edit_type='unknown_type',
        turn_id=1
    )
    
    applicator = EditApplicator(transcript)
    result = applicator.apply_edits([edit])
    
    # Should not raise an error, just skip the edit
    assert len(result['turns']) == 3, "transcript should be unchanged"
    print("  ✓ unknown edit type handled gracefully")


def main():
    print("\n" + "=" * 60)
    print("Edit Applicator Tests")
    print("=" * 60 + "\n")
    
    test_word_change()
    test_word_insert()
    test_word_delete()
    test_speaker_change()
    test_merge_words()
    test_split_word()
    test_turn_merge()
    test_turn_split()
    test_insert_annotation()
    test_multiple_edits()
    test_convenience_function()
    test_nonexistent_turn()
    test_unknown_edit_type()
    
    print("\n" + "=" * 60)
    print("All edit applicator tests passed! ✓")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
