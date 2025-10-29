#!/usr/bin/env python3
"""
Test script for CLI cross-talk functionality.
This script tests the CLI integration with cross-talk detection options.
"""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import main module functions
import main


class TestCliCrossTalk(unittest.TestCase):
    """Test class for CLI cross-talk functionality."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock audio file
        self.mock_audio_path = Path(self.temp_dir) / "test_audio.wav"
        self.mock_audio_path.touch()
        
        # Create a mock output directory
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_cross_talk_detection(self):
        """Test that CLI correctly enables cross-talk detection with --detect-cross-talk."""
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Parse arguments using the main module's parse_args function
            args = main.parse_args()
            
            # Verify that cross-talk detection is enabled
            self.assertTrue(args.detect_cross_talk, "Cross-talk detection should be enabled")
            self.assertFalse(args.mark_cross_talk, "Cross-talk marking should be disabled by default")
            self.assertFalse(args.basic_confidence, "Basic confidence should be disabled by default")
            
            # Verify that combined mode is set
            self.assertIsNotNone(args.combined, "Combined mode should be set")
            self.assertIsNone(args.interviewer, "Interviewer mode should not be set")
            self.assertIsNone(args.participant, "Participant mode should not be set")
    
    def test_cli_cross_talk_marking(self):
        """Test that CLI correctly enables cross-talk marking with --mark-cross-talk."""
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--mark-cross-talk"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Parse arguments using the main module's parse_args function
            args = main.parse_args()
            
            # Verify that both cross-talk detection and marking are enabled
            self.assertTrue(args.detect_cross_talk, "Cross-talk detection should be enabled")
            self.assertTrue(args.mark_cross_talk, "Cross-talk marking should be enabled")
            self.assertFalse(args.basic_confidence, "Basic confidence should be disabled by default")
    
    def test_cli_overlap_threshold(self):
        """Test that CLI correctly handles custom overlap threshold with --overlap-threshold."""
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--overlap-threshold", "0.2"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Parse arguments using the main module's parse_args function
            args = main.parse_args()
            
            # Verify that overlap threshold is set correctly
            self.assertTrue(args.detect_cross_talk, "Cross-talk detection should be enabled")
            self.assertEqual(args.overlap_threshold, 0.2, "Overlap threshold should be 0.2")
    
    def test_cli_basic_confidence(self):
        """Test that CLI correctly enables basic confidence with --basic-confidence."""
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--basic-confidence"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Parse arguments using the main module's parse_args function
            args = main.parse_args()
            
            # Verify that basic confidence is enabled
            self.assertTrue(args.detect_cross_talk, "Cross-talk detection should be enabled")
            self.assertTrue(args.basic_confidence, "Basic confidence should be enabled")
            self.assertFalse(args.mark_cross_talk, "Cross-talk marking should be disabled by default")
    
    def test_cli_all_cross_talk_options(self):
        """Test that CLI correctly handles all cross-talk options together."""
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--mark-cross-talk",
            "--overlap-threshold", "0.15",
            "--basic-confidence"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Parse arguments using the main module's parse_args function
            args = main.parse_args()
            
            # Verify that all cross-talk options are enabled
            self.assertTrue(args.detect_cross_talk, "Cross-talk detection should be enabled")
            self.assertTrue(args.mark_cross_talk, "Cross-talk marking should be enabled")
            self.assertTrue(args.basic_confidence, "Basic confidence should be enabled")
            self.assertEqual(args.overlap_threshold, 0.15, "Overlap threshold should be 0.15")
    
    def test_cli_backward_compatibility(self):
        """Test that CLI maintains backward compatibility when no cross-talk options are used."""
        # Mock sys.argv to simulate CLI arguments without cross-talk options
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir)
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Parse arguments using the main module's parse_args function
            args = main.parse_args()
            
            # Verify that cross-talk options are disabled by default
            self.assertFalse(args.detect_cross_talk, "Cross-talk detection should be disabled by default")
            self.assertFalse(args.mark_cross_talk, "Cross-talk marking should be disabled by default")
            self.assertFalse(args.basic_confidence, "Basic confidence should be disabled by default")
            self.assertEqual(args.overlap_threshold, 0.1, "Overlap threshold should be default 0.1")
    
    def test_cli_cross_talk_validation_errors(self):
        """Test that CLI correctly validates cross-talk option combinations."""
        
        # Test 1: Cross-talk marking without detection
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--mark-cross-talk"  # Without --detect-cross-talk
        ]
        
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                main.parse_args()
            
            # Should exit with error code
            self.assertEqual(cm.exception.code, 1, "Should exit with error code 1")
        
        # Test 2: Basic confidence without detection
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--basic-confidence"  # Without --detect-cross-talk
        ]
        
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                main.parse_args()
            
            # Should exit with error code
            self.assertEqual(cm.exception.code, 1, "Should exit with error code 1")
        
        # Test 3: Negative overlap threshold
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--overlap-threshold", "-0.1"  # Negative value
        ]
        
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                main.parse_args()
            
            # Should exit with error code
            self.assertEqual(cm.exception.code, 1, "Should exit with error code 1")
        
        # Test 4: Cross-talk detection in dual-track mode
        test_args = [
            "main.py",
            "--interviewer", str(self.mock_audio_path),
            "--participant", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk"  # Cross-talk detection in dual-track mode
        ]
        
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                main.parse_args()
            
            # Should exit with error code
            self.assertEqual(cm.exception.code, 1, "Should exit with error code 1")
    
    @patch('main.diarize_mixed')
    @patch('main.transcribe_with_alignment')
    @patch('main.standardize_and_get_path')
    @patch('main.ensure_session_dirs')
    @patch('main.get_progress_tracker')
    def test_cli_main_function_with_cross_talk(self, mock_tracker, mock_session_dirs, 
                                               mock_standardize, mock_transcribe, mock_diarize):
        """Test the main function with cross-talk options enabled."""
        # Setup mocks
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance
        
        mock_session_dirs.return_value = {
            "root": self.output_dir,
            "merged": self.output_dir,
            "speaker_interviewer": self.output_dir,
            "speaker_participant": self.output_dir
        }
        
        mock_standardize.return_value = str(self.mock_audio_path)
        mock_transcribe.return_value = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": None},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": None}
        ]
        
        mock_diarize.return_value = [
            {
                "speaker": "SpeakerA",
                "start": 0.0,
                "end": 1.0,
                "text": "Hello world",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}
                ]
            }
        ]
        
        # Mock sys.argv to simulate CLI arguments with cross-talk options
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--mark-cross-talk",
            "--basic-confidence"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Run the main function
            result = main.main()
            
            # Verify that the function completed successfully
            self.assertEqual(result, 0, "Main function should return 0 for success")
            
            # Verify that diarize_mixed was called with cross-talk detection enabled
            mock_diarize.assert_called_once()
            call_args = mock_diarize.call_args
            
            # Check that cross-talk detection was enabled
            self.assertTrue(call_args[1]["detect_cross_talk"], 
                          "Cross-talk detection should be enabled in diarize_mixed call")
            
            # Check that cross-talk config was passed
            self.assertIn("cross_talk_config", call_args[1], 
                         "Cross-talk config should be passed to diarize_mixed")
            
            cross_talk_config = call_args[1]["cross_talk_config"]
            self.assertTrue(cross_talk_config["mark_cross_talk"], 
                          "Mark cross-talk should be enabled in config")
            self.assertTrue(cross_talk_config["basic_confidence"], 
                          "Basic confidence should be enabled in config")
            self.assertEqual(cross_talk_config["overlap_threshold"], 0.1, 
                           "Overlap threshold should be default 0.1")
    
    @patch('main.diarize_mixed')
    @patch('main.transcribe_with_alignment')
    @patch('main.standardize_and_get_path')
    @patch('main.ensure_session_dirs')
    @patch('main.get_progress_tracker')
    def test_cli_main_function_cross_talk_fallback(self, mock_tracker, mock_session_dirs, 
                                                   mock_standardize, mock_transcribe, mock_diarize):
        """Test that CLI falls back gracefully when cross-talk detection fails."""
        # Setup mocks
        mock_tracker_instance = MagicMock()
        mock_tracker.return_value = mock_tracker_instance
        
        mock_session_dirs.return_value = {
            "root": self.output_dir,
            "merged": self.output_dir,
            "speaker_interviewer": self.output_dir,
            "speaker_participant": self.output_dir
        }
        
        mock_standardize.return_value = str(self.mock_audio_path)
        mock_transcribe.return_value = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": None},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": None}
        ]
        
        # Make diarize_mixed fail on first call (with cross-talk) but succeed on second (without)
        mock_diarize.side_effect = [
            Exception("Cross-talk detection failed"),  # First call fails
            [  # Second call succeeds
                {
                    "speaker": "SpeakerA",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello world",
                    "cross_talk_present": False,
                    "confidence": 1.0,
                    "words": [
                        {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
                        {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}
                    ]
                }
            ]
        ]
        
        # Mock sys.argv to simulate CLI arguments with cross-talk options
        test_args = [
            "main.py",
            "--combined", str(self.mock_audio_path),
            "--outdir", str(self.output_dir),
            "--detect-cross-talk",
            "--mark-cross-talk"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Run the main function
            result = main.main()
            
            # Verify that the function completed successfully (fallback worked)
            self.assertEqual(result, 0, "Main function should return 0 for success after fallback")
            
            # Verify that diarize_mixed was called twice (once with cross-talk, once without)
            self.assertEqual(mock_diarize.call_count, 2, 
                           "diarize_mixed should be called twice (with and without cross-talk)")
            
            # Check first call (with cross-talk)
            first_call_args = mock_diarize.call_args_list[0]
            self.assertTrue(first_call_args[1]["detect_cross_talk"], 
                          "First call should have cross-talk detection enabled")
            
            # Check second call (fallback without cross-talk)
            second_call_args = mock_diarize.call_args_list[1]
            self.assertFalse(second_call_args[1].get("detect_cross_talk", False), 
                           "Second call should not have cross-talk detection enabled")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCliCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing CLI cross-talk functionality...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All CLI cross-talk tests passed! CLI integration with cross-talk options works correctly.")
    else:
        print("❌ Some CLI cross-talk tests failed. Please review the implementation.")
        sys.exit(1)