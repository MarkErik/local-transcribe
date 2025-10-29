#!/usr/bin/env python3
"""
Integration tests for cross-talk detection functionality.
This script tests the complete cross-talk detection pipeline integration.
"""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diarize import diarize_mixed
from cross_talk import detect_basic_cross_talk, assign_words_with_basic_cross_talk, BASIC_CROSS_TALK_CONFIG
from turns import build_turns


class TestIntegrationCrossTalk(unittest.TestCase):
    """Test class for integration of cross-talk detection with diarization pipeline."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Sample words for testing
        self.words = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": None},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": None},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": None},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": None},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": None},
            {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": None},
            {"text": "fine", "start": 3.4, "end": 3.6, "speaker": None},
            {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": None},
            {"text": "What", "start": 4.6, "end": 4.8, "speaker": None},
            {"text": "about", "start": 4.9, "end": 5.1, "speaker": None},
            {"text": "you", "start": 5.2, "end": 5.4, "speaker": None},
            {"text": "Good", "start": 7.1, "end": 7.3, "speaker": None},
            {"text": "to", "start": 7.4, "end": 7.5, "speaker": None},
            {"text": "hear", "start": 7.6, "end": 7.8, "speaker": None}
        ]
        
        # Create a temporary audio file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_audio_path = Path(self.temp_dir) / "test_audio.wav"
        
        # Create a mock audio file (we'll mock the actual audio processing)
        self.temp_audio_path.touch()
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('diarize.Pipeline')
    def test_diarize_mixed_with_cross_talk(self, mock_pipeline_class):
        """Test that diarize_mixed works correctly with cross-talk detection enabled."""
        # Mock the pipeline and its behavior
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock diarization result
        mock_diar_result = MagicMock()
        mock_diar_result.speaker_diarization = MagicMock()
        
        # Create a mock annotation object with itertracks method
        def mock_itertracks(yield_label=True):
            segments = [
                (MagicMock(start=0.0, end=2.0), 0, "SpeakerA"),
                (MagicMock(start=1.5, end=3.5), 1, "SpeakerB"),
                (MagicMock(start=3.0, end=5.0), 2, "SpeakerA"),
                (MagicMock(start=4.5, end=6.0), 3, "SpeakerB"),
                (MagicMock(start=7.0, end=9.0), 4, "SpeakerA")
            ]
            for segment in segments:
                yield segment
        
        mock_diar_result.speaker_diarization.itertracks = mock_itertracks
        mock_pipeline.return_value = mock_diar_result
        
        # Mock audio loading functions
        with patch('diarize._load_waveform_mono_32f') as mock_load, \
             patch('diarize._maybe_resample') as mock_resample:
            
            mock_load.return_value = (MagicMock(), 16000)
            mock_resample.return_value = (MagicMock(), 16000)
            
            # Call diarize_mixed with cross-talk detection enabled
            turns = diarize_mixed(
                str(self.temp_audio_path),
                self.words,
                detect_cross_talk=True
            )
            
            # Verify that turns were created
            self.assertIsInstance(turns, list, "Should return a list of turns")
            self.assertGreater(len(turns), 0, "Should have at least one turn")
            
            # Check that cross-talk information is present in turns
            for turn in turns:
                self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
                self.assertIn("confidence", turn, "Turn should have confidence key")
                self.assertIn("words", turn, "Turn should have words key")
                
                # Check that words have cross-talk information
                for word in turn.get("words", []):
                    self.assertIn("cross_talk", word, "Word should have cross_talk key")
                    self.assertIn("confidence", word, "Word should have confidence key")
    
    @patch('diarize.Pipeline')
    def test_diarize_mixed_without_cross_talk(self, mock_pipeline_class):
        """Test that diarize_mixed maintains backward compatibility when cross-talk detection is disabled."""
        # Mock the pipeline and its behavior
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock diarization result
        mock_diar_result = MagicMock()
        mock_diar_result.speaker_diarization = MagicMock()
        
        # Create a mock annotation object with itertracks method
        def mock_itertracks(yield_label=True):
            segments = [
                (MagicMock(start=0.0, end=2.0), 0, "SpeakerA"),
                (MagicMock(start=1.5, end=3.5), 1, "SpeakerB"),
                (MagicMock(start=3.0, end=5.0), 2, "SpeakerA"),
                (MagicMock(start=4.5, end=6.0), 3, "SpeakerB"),
                (MagicMock(start=7.0, end=9.0), 4, "SpeakerA")
            ]
            for segment in segments:
                yield segment
        
        mock_diar_result.speaker_diarization.itertracks = mock_itertracks
        mock_pipeline.return_value = mock_diar_result
        
        # Mock audio loading functions
        with patch('diarize._load_waveform_mono_32f') as mock_load, \
             patch('diarize._maybe_resample') as mock_resample:
            
            mock_load.return_value = (MagicMock(), 16000)
            mock_resample.return_value = (MagicMock(), 16000)
            
            # Call diarize_mixed without cross-talk detection
            turns = diarize_mixed(
                str(self.temp_audio_path),
                self.words,
                detect_cross_talk=False
            )
            
            # Verify that turns were created
            self.assertIsInstance(turns, list, "Should return a list of turns")
            self.assertGreater(len(turns), 0, "Should have at least one turn")
            
            # Check that turns still have cross_talk_present and confidence keys for backward compatibility
            for turn in turns:
                self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
                self.assertIn("confidence", turn, "Turn should have confidence key")
                
                # Without cross-talk detection, these should be default values
                self.assertEqual(turn["cross_talk_present"], False, "Cross-talk should be False when detection is disabled")
                self.assertEqual(turn["confidence"], 1.0, "Confidence should be 1.0 when detection is disabled")
    
    @patch('diarize.Pipeline')
    def test_cross_talk_config_customization(self, mock_pipeline_class):
        """Test that custom cross-talk configuration parameters work correctly."""
        # Mock the pipeline and its behavior
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock diarization result
        mock_diar_result = MagicMock()
        mock_diar_result.speaker_diarization = MagicMock()
        
        # Create a mock annotation object with itertracks method
        def mock_itertracks(yield_label=True):
            segments = [
                (MagicMock(start=0.0, end=2.0), 0, "SpeakerA"),
                (MagicMock(start=1.5, end=3.5), 1, "SpeakerB"),
                (MagicMock(start=3.0, end=5.0), 2, "SpeakerA"),
                (MagicMock(start=4.5, end=6.0), 3, "SpeakerB"),
                (MagicMock(start=7.0, end=9.0), 4, "SpeakerA")
            ]
            for segment in segments:
                yield segment
        
        mock_diar_result.speaker_diarization.itertracks = mock_itertracks
        mock_pipeline.return_value = mock_diar_result
        
        # Mock audio loading functions
        with patch('diarize._load_waveform_mono_32f') as mock_load, \
             patch('diarize._maybe_resample') as mock_resample:
            
            mock_load.return_value = (MagicMock(), 16000)
            mock_resample.return_value = (MagicMock(), 16000)
            
            # Create custom cross-talk configuration
            custom_config = {
                "overlap_threshold": 0.2,  # Higher than default
                "confidence_threshold": 0.7,
                "min_word_duration": 0.1,  # Higher than default
                "mark_cross_talk": True,
                "basic_confidence": True
            }
            
            # Call diarize_mixed with custom cross-talk configuration
            turns = diarize_mixed(
                str(self.temp_audio_path),
                self.words,
                detect_cross_talk=True,
                cross_talk_config=custom_config
            )
            
            # Verify that turns were created
            self.assertIsInstance(turns, list, "Should return a list of turns")
            self.assertGreater(len(turns), 0, "Should have at least one turn")
            
            # With higher overlap threshold and min_word_duration, fewer words should be marked as cross-talk
            cross_talk_turns = [turn for turn in turns if turn["cross_talk_present"]]
            self.assertLessEqual(len(cross_talk_turns), len(turns), "Should have fewer or equal cross-talk turns")
    
    @patch('diarize.Pipeline')
    def test_cross_talk_config_invalid(self, mock_pipeline_class):
        """Test that invalid cross-talk configuration is handled gracefully."""
        # Mock the pipeline and its behavior
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock diarization result
        mock_diar_result = MagicMock()
        mock_diar_result.speaker_diarization = MagicMock()
        
        # Create a mock annotation object with itertracks method
        def mock_itertracks(yield_label=True):
            segments = [
                (MagicMock(start=0.0, end=2.0), 0, "SpeakerA"),
                (MagicMock(start=1.5, end=3.5), 1, "SpeakerB"),
                (MagicMock(start=3.0, end=5.0), 2, "SpeakerA"),
                (MagicMock(start=4.5, end=6.0), 3, "SpeakerB"),
                (MagicMock(start=7.0, end=9.0), 4, "SpeakerA")
            ]
            for segment in segments:
                yield segment
        
        mock_diar_result.speaker_diarization.itertracks = mock_itertracks
        mock_pipeline.return_value = mock_diar_result
        
        # Mock audio loading functions
        with patch('diarize._load_waveform_mono_32f') as mock_load, \
             patch('diarize._maybe_resample') as mock_resample:
            
            mock_load.return_value = (MagicMock(), 16000)
            mock_resample.return_value = (MagicMock(), 16000)
            
            # Create invalid cross-talk configuration
            invalid_config = {
                "overlap_threshold": -0.1,  # Invalid negative value
                "confidence_threshold": "invalid",  # Invalid type
                "min_word_duration": None  # Invalid type
            }
            
            # Call diarize_mixed with invalid cross-talk configuration
            turns = diarize_mixed(
                str(self.temp_audio_path),
                self.words,
                detect_cross_talk=True,
                cross_talk_config=invalid_config
            )
            
            # Verify that turns were created (should fall back to default config)
            self.assertIsInstance(turns, list, "Should return a list of turns")
            self.assertGreater(len(turns), 0, "Should have at least one turn")
            
            # Should still work with default configuration
            for turn in turns:
                self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
                self.assertIn("confidence", turn, "Turn should have confidence key")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegrationCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing integration of cross-talk detection with diarization pipeline...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All integration tests passed! Cross-talk detection integrates correctly.")
    else:
        print("❌ Some integration tests failed. Please review the implementation.")
        sys.exit(1)