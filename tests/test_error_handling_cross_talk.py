#!/usr/bin/env python3
"""
Error handling tests for cross-talk functionality.
This script tests error scenarios and graceful fallback behavior in cross-talk detection.
"""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diarize import diarize_mixed, DiarizationError
from cross_talk import detect_basic_cross_talk, assign_words_with_basic_cross_talk, BASIC_CROSS_TALK_CONFIG
from turns import build_turns
from txt_writer import write_timestamped_txt, write_plain_txt
from csv_writer import write_conversation_csv


class TestErrorHandlingCrossTalk(unittest.TestCase):
    """Test class for error handling in cross-talk functionality."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock audio file
        self.mock_audio_path = Path(self.temp_dir) / "test_audio.wav"
        self.mock_audio_path.touch()
        
        # Sample words for testing
        self.sample_words = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": None},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": None},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": None},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": None},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": None}
        ]
        
        # Sample diarization segments
        self.sample_segments = [
            {"start": 0.0, "end": 2.0, "label": "SpeakerA"},
            {"start": 1.5, "end": 3.5, "label": "SpeakerB"}
        ]
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cross_talk_error_handling_invalid_segments(self):
        """Test that errors in cross-talk detection are handled gracefully with invalid segments."""
        # Test with None segments
        result = detect_basic_cross_talk(None)
        self.assertIsInstance(result, list, "Should return empty list for None segments")
        self.assertEqual(len(result), 0, "Should return empty list for None segments")
        
        # Test with empty segments
        result = detect_basic_cross_talk([])
        self.assertIsInstance(result, list, "Should return empty list for empty segments")
        self.assertEqual(len(result), 0, "Should return empty list for empty segments")
        
        # Test with segments missing required keys
        invalid_segments = [
            {"start": 0.0, "end": 2.0},  # Missing 'label' key
            {"label": "SpeakerB", "end": 3.5}  # Missing 'start' key
        ]
        
        # This should not crash, but handle gracefully
        result = detect_basic_cross_talk(invalid_segments)
        self.assertIsInstance(result, list, "Should return list even for invalid segments")
    
    def test_cross_talk_error_handling_invalid_words(self):
        """Test that errors in word assignment are handled gracefully with invalid words."""
        # Create valid cross-talk segments
        cross_talk_segments = detect_basic_cross_talk(self.sample_segments)
        
        # Test with None words
        result = assign_words_with_basic_cross_talk(None, self.sample_segments, cross_talk_segments)
        self.assertIsInstance(result, list, "Should return empty list for None words")
        self.assertEqual(len(result), 0, "Should return empty list for None words")
        
        # Test with empty words
        result = assign_words_with_basic_cross_talk([], self.sample_segments, cross_talk_segments)
        self.assertIsInstance(result, list, "Should return empty list for empty words")
        self.assertEqual(len(result), 0, "Should return empty list for empty words")
        
        # Test with words missing required keys
        invalid_words = [
            {"text": "Hello", "start": 0.2},  # Missing 'end' key
            {"end": 0.9, "speaker": "SpeakerA"}  # Missing 'text' and 'start' keys
        ]
        
        # This should not crash, but handle gracefully
        result = assign_words_with_basic_cross_talk(invalid_words, self.sample_segments, cross_talk_segments)
        self.assertIsInstance(result, list, "Should return list even for invalid words")
        
        # All words should have default values
        for word in result:
            self.assertIn("cross_talk", word, "Word should have cross_talk key")
            self.assertIn("confidence", word, "Word should have confidence key")
            self.assertFalse(word["cross_talk"], "Invalid words should not be marked as cross-talk")
            self.assertEqual(word["confidence"], 1.0, "Invalid words should have default confidence")
    
    def test_cross_talk_error_handling_invalid_config(self):
        """Test that invalid cross-talk configuration is handled correctly."""
        # Test with None config
        result = assign_words_with_basic_cross_talk(self.sample_words, self.sample_segments, [])
        self.assertIsInstance(result, list, "Should return list for None config")
        
        # All words should have default values from BASIC_CROSS_TALK_CONFIG
        for word in result:
            self.assertIn("cross_talk", word, "Word should have cross_talk key")
            self.assertIn("confidence", word, "Word should have confidence key")
        
        # Test with invalid config values (this would be handled by the function)
        original_config = BASIC_CROSS_TALK_CONFIG.copy()
        
        # Temporarily modify the global config to test error handling
        try:
            # Test with negative min_word_duration
            BASIC_CROSS_TALK_CONFIG["min_word_duration"] = -0.1
            
            result = assign_words_with_basic_cross_talk(self.sample_words, self.sample_segments, [])
            
            # Should still work, using the invalid value
            self.assertIsInstance(result, list, "Should handle negative min_word_duration")
            
        finally:
            # Restore original config
            BASIC_CROSS_TALK_CONFIG.clear()
            BASIC_CROSS_TALK_CONFIG.update(original_config)
    
    @patch('diarize.Pipeline')
    def test_diarize_mixed_cross_talk_detection_failure(self, mock_pipeline_class):
        """Test that diarize_mixed handles cross-talk detection failure gracefully."""
        # Mock the pipeline to work normally
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock diarization result
        mock_diar_result = MagicMock()
        mock_diar_result.speaker_diarization = MagicMock()
        
        def mock_itertracks(yield_label=True):
            segments = [
                (MagicMock(start=0.0, end=2.0), 0, "SpeakerA"),
                (MagicMock(start=1.5, end=3.5), 1, "SpeakerB")
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
            
            # Test with cross-talk detection enabled but it fails
            # We'll patch the cross-talk functions to raise exceptions
            
            with patch('diarize.detect_basic_cross_talk') as mock_detect, \
                 patch('diarize.assign_words_with_basic_cross_talk') as mock_assign:
                
                # Make cross-talk detection fail
                mock_detect.side_effect = Exception("Cross-talk detection failed")
                
                # Call diarize_mixed with cross-talk detection enabled
                turns = diarize_mixed(
                    str(self.mock_audio_path),
                    self.sample_words,
                    detect_cross_talk=True
                )
                
                # Verify that turns were still created (fallback to standard diarization)
                self.assertIsInstance(turns, list, "Should return a list of turns despite cross-talk failure")
                self.assertGreater(len(turns), 0, "Should have at least one turn despite cross-talk failure")
                
                # Verify that turns have default cross-talk values
                for turn in turns:
                    self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
                    self.assertIn("confidence", turn, "Turn should have confidence key")
                    self.assertFalse(turn["cross_talk_present"], "Cross-talk should be False when detection fails")
                    self.assertEqual(turn["confidence"], 1.0, "Confidence should be 1.0 when detection fails")
    
    def test_txt_writer_error_handling_cross_talk(self):
        """Test that txt_writer handles cross-talk errors gracefully."""
        # Create turns with problematic cross-talk data
        problematic_turns = [
            {
                "speaker": "SpeakerA",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "cross_talk_present": True,
                "confidence": 1.5,  # Invalid confidence (> 1.0)
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", 
                     "cross_talk": False, "confidence": 1.0},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", 
                     "cross_talk": True, "confidence": -0.5}  # Invalid confidence (< 0.0)
                ]
            }
        ]
        
        # Test timestamped text writer
        output_path = self.temp_dir / "timestamped_error_test.txt"
        
        # This should not crash, but handle gracefully
        try:
            write_timestamped_txt(problematic_turns, output_path, mark_cross_talk=True)
            self.assertTrue(output_path.exists(), "File should be created despite problematic data")
            
            # Read and verify content
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("Hello world", content, "Text should be present despite problematic data")
            
        except Exception as e:
            self.fail(f"txt_writer should handle problematic cross-talk data gracefully, but raised: {e}")
        
        # Test plain text writer
        output_path = self.temp_dir / "plain_error_test.txt"
        
        try:
            write_plain_txt(problematic_turns, output_path, mark_cross_talk=True)
            self.assertTrue(output_path.exists(), "File should be created despite problematic data")
            
            # Read and verify content
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("Hello world", content, "Text should be present despite problematic data")
            
        except Exception as e:
            self.fail(f"txt_writer should handle problematic cross-talk data gracefully, but raised: {e}")
    
    def test_csv_writer_error_handling_cross_talk(self):
        """Test that csv_writer handles cross-talk errors gracefully."""
        # Create turns with problematic cross-talk data
        problematic_turns = [
            {
                "speaker": "Interviewer",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "cross_talk_present": True,
                "confidence": "invalid",  # Invalid confidence type
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "Interviewer", 
                     "cross_talk": False, "confidence": 1.0},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "Interviewer", 
                     "cross_talk": True, "confidence": None}  # Invalid confidence type
                ]
            }
        ]
        
        # Test CSV writer with cross-talk columns
        output_path = self.temp_dir / "csv_error_test.csv"
        
        # This should not crash, but handle gracefully
        try:
            write_conversation_csv(problematic_turns, output_path, include_cross_talk=True)
            self.assertTrue(output_path.exists(), "CSV file should be created despite problematic data")
            
            # Read and verify content
            import csv
            with open(output_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            
            self.assertEqual(len(rows[0]), 4, "Header should have 4 columns")
            self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'])
            
            # Verify that problematic data was handled with defaults
            self.assertEqual(rows[1][2], "False", "Cross-talk should default to False")
            self.assertEqual(rows[1][3], "1.000", "Confidence should default to 1.000")
            
        except Exception as e:
            self.fail(f"csv_writer should handle problematic cross-talk data gracefully, but raised: {e}")
    
    def test_turns_error_handling_cross_talk(self):
        """Test that turns building handles cross-talk errors gracefully."""
        # Create words with problematic cross-talk data
        problematic_words = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", 
             "cross_talk": "invalid", "confidence": 1.0},  # Invalid cross_talk type
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", 
             "cross_talk": True, "confidence": "invalid"}  # Invalid confidence type
        ]
        
        # This should not crash, but handle gracefully
        try:
            turns = build_turns(problematic_words, speaker_label="SpeakerA")
            
            self.assertIsInstance(turns, list, "Should return list despite problematic data")
            self.assertGreater(len(turns), 0, "Should have at least one turn despite problematic data")
            
            # Verify that turns have default values
            for turn in turns:
                self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
                self.assertIn("confidence", turn, "Turn should have confidence key")
                
                # Values should be reasonable defaults
                self.assertIsInstance(turn["cross_talk_present"], bool, 
                                    "cross_talk_present should be boolean")
                self.assertIsInstance(turn["confidence"], (int, float), 
                                    "confidence should be numeric")
                self.assertGreaterEqual(turn["confidence"], 0.0, "confidence should be >= 0")
                self.assertLessEqual(turn["confidence"], 1.0, "confidence should be <= 1")
            
        except Exception as e:
            self.fail(f"build_turns should handle problematic cross-talk data gracefully, but raised: {e}")
    
    def test_cross_talk_memory_error_handling(self):
        """Test that cross-talk functions handle memory errors gracefully."""
        # Create a large amount of test data to potentially cause memory issues
        large_segments = []
        for i in range(10000):
            large_segments.append({
                "start": i * 0.1,
                "end": (i + 1) * 0.1,
                "label": f"Speaker{i % 2}"
            })
        
        # This should either work or fail gracefully, not crash
        try:
            result = detect_basic_cross_talk(large_segments)
            self.assertIsInstance(result, list, "Should return list for large data")
            
        except MemoryError:
            # MemoryError is acceptable for very large data
            self.skipTest("MemoryError encountered with large test data - this is acceptable")
            
        except Exception as e:
            self.fail(f"detect_basic_cross_talk should handle large data gracefully or raise MemoryError, but raised: {e}")
    
    def test_cross_talk_recursive_error_handling(self):
        """Test that cross-talk functions handle recursive/stack overflow errors gracefully."""
        # Create segments that could potentially cause recursive issues
        # (This is more of a theoretical test, as the current implementation isn't recursive)
        problematic_segments = [
            {"start": 0.0, "end": 10.0, "label": "SpeakerA"},
            {"start": 0.1, "end": 9.9, "label": "SpeakerB"},
            {"start": 0.2, "end": 9.8, "label": "SpeakerA"},
            {"start": 0.3, "end": 9.7, "label": "SpeakerB"}
        ]
        
        # This should work without issues
        try:
            result = detect_basic_cross_talk(problematic_segments)
            self.assertIsInstance(result, list, "Should return list for potentially problematic data")
            
        except RecursionError:
            # RecursionError would indicate a design issue
            self.fail("detect_basic_cross_talk should not cause RecursionError")
            
        except Exception as e:
            self.fail(f"detect_basic_cross_talk should handle potentially problematic data gracefully, but raised: {e}")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestErrorHandlingCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing error handling in cross-talk functionality...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All error handling tests passed! Cross-talk error scenarios are handled gracefully.")
    else:
        print("❌ Some error handling tests failed. Please review the implementation.")
        sys.exit(1)