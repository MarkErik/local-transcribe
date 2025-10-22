#!/usr/bin/env python3
"""
Test script for the diarization fix and ASR text saving functionality.
This script tests both the new write_asr_words function and the diarization API compatibility.
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from txt_writer import write_asr_words
from diarize import diarize_mixed


def test_write_asr_words():
    """Test the new write_asr_words function."""
    print("Testing write_asr_words function...")
    
    # Create test words data
    test_words = [
        {"text": "Hello", "start": 0.0, "end": 0.5, "speaker": None},
        {"text": "world", "start": 0.6, "end": 1.0, "speaker": None},
        {"text": "this", "start": 1.1, "end": 1.4, "speaker": None},
        {"text": "is", "start": 1.5, "end": 1.7, "speaker": None},
        {"text": "a", "start": 1.8, "end": 1.9, "speaker": None},
        {"text": "test", "start": 2.0, "end": 2.5, "speaker": None},
    ]
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name
    
    try:
        # Write the words to file
        write_asr_words(test_words, temp_path)
        
        # Read and verify content
        with open(temp_path, 'r') as f:
            content = f.read().strip()
        
        expected = "Hello world this is a test"
        assert content == expected, f"Expected '{expected}', got '{content}'"
        
        print("✅ write_asr_words test passed")
        
    finally:
        # Clean up
        os.unlink(temp_path)


def test_diarization_new_api():
    """Test diarization with the new API (tracks.items())."""
    print("Testing diarization with new API...")
    
    # Create mock diarization result with new API
    mock_diar = Mock()
    mock_segment1 = Mock()
    mock_segment1.start = 0.0
    mock_segment1.end = 5.0
    mock_segment2 = Mock()
    mock_segment2.start = 5.1
    mock_segment2.end = 10.0
    
    # Mock the tracks.items() method (new API)
    mock_diar.tracks.items.return_value = [
        ("Speaker_A", mock_segment1),
        ("Speaker_B", mock_segment2)
    ]
    
    # Mock the itertracks method to raise AttributeError (old API not available)
    mock_diar.itertracks = Mock(side_effect=AttributeError("'DiarizeOutput' object has no attribute 'itertracks'"))
    
    # Create test words
    test_words = [
        {"text": "First", "start": 1.0, "end": 2.0, "speaker": None},
        {"text": "sentence", "start": 2.1, "end": 3.0, "speaker": None},
        {"text": "Second", "start": 6.0, "end": 7.0, "speaker": None},
        {"text": "sentence", "start": 7.1, "end": 8.0, "speaker": None},
    ]
    
    # Create a temporary audio file for the test
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # Mock the entire diarization pipeline
        with patch('diarize.Pipeline') as mock_pipeline_class, \
             patch('diarize._load_waveform_mono_32f') as mock_load_audio, \
             patch('diarize._maybe_resample') as mock_resample, \
             patch('diarize.build_turns') as mock_build_turns, \
             patch('diarize.merge_turn_streams') as mock_merge:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.return_value = mock_diar
            
            mock_load_audio.return_value = (Mock(), 16000)
            mock_resample.return_value = (Mock(), 16000)
            
            # Mock the turn building and merging
            mock_build_turns.return_value = []
            mock_merge.return_value = []
            
            # Test the diarization function
            try:
                result = diarize_mixed(temp_audio_path, test_words)
                print("✅ New API diarization test passed")
            except Exception as e:
                print(f"❌ New API test failed: {e}")
                raise
    finally:
        # Clean up temporary audio file
        os.unlink(temp_audio_path)


def test_diarization_old_api():
    """Test diarization with the old API (itertracks)."""
    print("Testing diarization with old API...")
    
    # Create mock diarization result with old API
    mock_diar = Mock()
    mock_segment1 = Mock()
    mock_segment1.start = 0.0
    mock_segment1.end = 5.0
    mock_segment2 = Mock()
    mock_segment2.start = 5.1
    mock_segment2.end = 10.0
    
    # Mock the itertracks method (old API)
    mock_diar.itertracks.return_value = [
        (mock_segment1, "track1", "Speaker_A"),
        (mock_segment2, "track2", "Speaker_B")
    ]
    
    # Mock the tracks attribute to not have items() (new API not available)
    mock_diar.tracks = Mock()
    mock_diar.tracks.items.side_effect = AttributeError("'DiarizeOutput' object has no attribute 'tracks'")
    
    # Create test words
    test_words = [
        {"text": "First", "start": 1.0, "end": 2.0, "speaker": None},
        {"text": "sentence", "start": 2.1, "end": 3.0, "speaker": None},
        {"text": "Second", "start": 6.0, "end": 7.0, "speaker": None},
        {"text": "sentence", "start": 7.1, "end": 8.0, "speaker": None},
    ]
    
    # Create a temporary audio file for the test
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    try:
        # Mock the entire diarization pipeline
        with patch('diarize.Pipeline') as mock_pipeline_class, \
             patch('diarize._load_waveform_mono_32f') as mock_load_audio, \
             patch('diarize._maybe_resample') as mock_resample, \
             patch('diarize.build_turns') as mock_build_turns, \
             patch('diarize.merge_turn_streams') as mock_merge:
            
            # Setup mocks
            mock_pipeline = Mock()
            mock_pipeline_class.from_pretrained.return_value = mock_pipeline
            mock_pipeline.return_value = mock_diar
            
            mock_load_audio.return_value = (Mock(), 16000)
            mock_resample.return_value = (Mock(), 16000)
            
            # Mock the turn building and merging
            mock_build_turns.return_value = []
            mock_merge.return_value = []
            
            # Test the diarization function
            try:
                result = diarize_mixed(temp_audio_path, test_words)
                print("✅ Old API diarization test passed")
            except Exception as e:
                print(f"❌ Old API test failed: {e}")
                raise
    finally:
        # Clean up temporary audio file
        os.unlink(temp_audio_path)


def test_combined_mode_integration():
    """Test the integration of ASR text saving in combined mode."""
    print("Testing combined mode integration...")
    
    # Create test words
    test_words = [
        {"text": "This", "start": 0.0, "end": 0.5, "speaker": None},
        {"text": "is", "start": 0.6, "end": 1.0, "speaker": None},
        {"text": "a", "start": 1.1, "end": 1.4, "speaker": None},
        {"text": "test", "start": 1.5, "end": 2.0, "speaker": None},
    ]
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        merged_dir = Path(temp_dir) / "merged"
        merged_dir.mkdir()
        
        asr_file = merged_dir / "asr.txt"
        
        # Write ASR results
        write_asr_words(test_words, asr_file)
        
        # Verify the file was created and has correct content
        assert asr_file.exists(), "ASR file was not created"
        
        with open(asr_file, 'r') as f:
            content = f.read().strip()
        
        expected = "This is a test"
        assert content == expected, f"Expected '{expected}', got '{content}'"
        
        print("✅ Combined mode integration test passed")


def main():
    """Run all tests."""
    print("\n🧪 Running Diarization Fix Tests\n")
    
    try:
        test_write_asr_words()
        print()
        
        test_diarization_new_api()
        print()
        
        test_diarization_old_api()
        print()
        
        test_combined_mode_integration()
        print()
        
        print("✅ All diarization tests passed!")
        print("\nThe diarization fix and ASR text saving features are working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())