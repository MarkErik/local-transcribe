#!/usr/bin/env python3
"""
End-to-end tests for the complete cross-talk pipeline.
This script tests the entire pipeline from audio input to output files with cross-talk detection.
"""

import sys
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import necessary modules
import main
from diarize import diarize_mixed
from cross_talk import detect_basic_cross_talk, assign_words_with_basic_cross_talk
from turns import build_turns
from txt_writer import write_timestamped_txt, write_plain_txt
from csv_writer import write_conversation_csv


class TestEndToEndCrossTalk(unittest.TestCase):
    """Test class for end-to-end cross-talk pipeline."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock audio file
        self.mock_audio_path = Path(self.temp_dir) / "test_audio.wav"
        self.mock_audio_path.touch()
        
        # Create output directory
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Sample words that would come from ASR
        self.sample_words = [
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
        
        # Expected diarization segments (simulating pyannote output)
        self.diar_segments = [
            {"start": 0.0, "end": 2.0, "label": "SpeakerA"},
            {"start": 1.5, "end": 3.5, "label": "SpeakerB"},
            {"start": 3.0, "end": 5.0, "label": "SpeakerA"},
            {"start": 4.5, "end": 6.0, "label": "SpeakerB"},
            {"start": 7.0, "end": 9.0, "label": "SpeakerA"}
        ]
        
        # Expected cross-talk segments
        self.expected_cross_talk_segments = [
            {"start": 1.5, "end": 2.0, "speakers": ["SpeakerA", "SpeakerB"], "duration": 0.5},
            {"start": 3.0, "end": 3.5, "speakers": ["SpeakerA", "SpeakerB"], "duration": 0.5},
            {"start": 4.5, "end": 5.0, "speakers": ["SpeakerA", "SpeakerB"], "duration": 0.5}
        ]
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('diarize.Pipeline')
    def test_end_to_end_cross_talk_pipeline(self, mock_pipeline_class):
        """Test the complete pipeline from audio input to output files with cross-talk detection."""
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
            
            # Step 1: Run diarization with cross-talk detection
            turns = diarize_mixed(
                str(self.mock_audio_path),
                self.sample_words,
                detect_cross_talk=True
            )
            
            # Verify turns were created
            self.assertIsInstance(turns, list, "Should return a list of turns")
            self.assertGreater(len(turns), 0, "Should have at least one turn")
            
            # Step 2: Verify cross-talk information is present
            for turn in turns:
                self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
                self.assertIn("confidence", turn, "Turn should have confidence key")
                self.assertIn("words", turn, "Turn should have words key")
                
                # Verify words have cross-talk information
                for word in turn.get("words", []):
                    self.assertIn("cross_talk", word, "Word should have cross_talk key")
                    self.assertIn("confidence", word, "Word should have confidence key")
            
            # Step 3: Generate output files
            timestamped_txt_path = self.output_dir / "transcript.timestamped.txt"
            plain_txt_path = self.output_dir / "transcript.txt"
            csv_path = self.output_dir / "transcript.csv"
            
            # Write timestamped text with cross-talk marking
            write_timestamped_txt(turns, timestamped_txt_path, mark_cross_talk=True)
            
            # Write plain text with cross-talk marking
            write_plain_txt(turns, plain_txt_path, mark_cross_talk=True)
            
            # Write CSV with cross-talk columns
            write_conversation_csv(turns, csv_path, include_cross_talk=True)
            
            # Step 4: Verify output files exist and have correct content
            self.assertTrue(timestamped_txt_path.exists(), "Timestamped text file should exist")
            self.assertTrue(plain_txt_path.exists(), "Plain text file should exist")
            self.assertTrue(csv_path.exists(), "CSV file should exist")
            
            # Verify timestamped text content
            timestamped_content = timestamped_txt_path.read_text(encoding="utf-8")
            self.assertIn("*are*", timestamped_content, "Cross-talk word 'are' should be marked with asterisks")
            self.assertIn("*you*", timestamped_content, "Cross-talk word 'you' should be marked with asterisks")
            self.assertIn("*I'm*", timestamped_content, "Cross-talk word 'I'm' should be marked with asterisks")
            self.assertIn("*fine*", timestamped_content, "Cross-talk word 'fine' should be marked with asterisks")
            
            # Verify plain text content
            plain_content = plain_txt_path.read_text(encoding="utf-8")
            self.assertIn("*are*", plain_content, "Cross-talk word 'are' should be marked with asterisks")
            self.assertIn("*you*", plain_content, "Cross-talk word 'you' should be marked with asterisks")
            self.assertIn("*I'm*", plain_content, "Cross-talk word 'I'm' should be marked with asterisks")
            self.assertIn("*fine*", plain_content, "Cross-talk word 'fine' should be marked with asterisks")
            
            # Verify CSV content
            import csv
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            
            self.assertEqual(len(rows[0]), 4, "CSV header should have 4 columns")
            self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'])
            
            # Verify that some rows have cross-talk information
            cross_talk_rows = [row for row in rows[1:] if row[2] == "True"]
            self.assertGreater(len(cross_talk_rows), 0, "Should have at least one row with cross-talk")
    
    @patch('main.diarize_mixed')
    @patch('main.transcribe_with_alignment')
    @patch('main.standardize_and_get_path')
    @patch('main.ensure_session_dirs')
    @patch('main.get_progress_tracker')
    def test_end_to_end_cli_integration(self, mock_tracker, mock_session_dirs, 
                                        mock_standardize, mock_transcribe, mock_diarize):
        """Test end-to-end integration through CLI with cross-talk options."""
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
        mock_transcribe.return_value = self.sample_words
        
        # Create expected turns with cross-talk information
        expected_turns = [
            {
                "speaker": "SpeakerA",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}
                ]
            },
            {
                "speaker": "SpeakerB",
                "start": 1.5,
                "end": 3.5,
                "text": "How are you",
                "cross_talk_present": True,
                "confidence": 0.5,
                "words": [
                    {"text": "How", "start": 1.6, "end": 1.8, "speaker": "SpeakerB", "cross_talk": False, "confidence": 1.0},
                    {"text": "are", "start": 1.9, "end": 2.1, "speaker": "SpeakerB", "cross_talk": True, "confidence": 0.5},
                    {"text": "you", "start": 2.2, "end": 2.4, "speaker": "SpeakerB", "cross_talk": True, "confidence": 0.3}
                ]
            }
        ]
        
        mock_diarize.return_value = expected_turns
        
        # Mock sys.argv to simulate CLI arguments
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
            
            # Verify that output files were created
            timestamped_txt_path = self.output_dir / "transcript.timestamped.txt"
            plain_txt_path = self.output_dir / "transcript.txt"
            csv_path = self.output_dir / "transcript.csv"
            md_path = self.output_dir / "transcript.md"
            
            self.assertTrue(timestamped_txt_path.exists(), "Timestamped text file should exist")
            self.assertTrue(plain_txt_path.exists(), "Plain text file should exist")
            self.assertTrue(csv_path.exists(), "CSV file should exist")
            self.assertTrue(md_path.exists(), "Markdown file should exist")
            
            # Verify content of output files
            timestamped_content = timestamped_txt_path.read_text(encoding="utf-8")
            self.assertIn("*are*", timestamped_content, "Cross-talk word 'are' should be marked with asterisks")
            self.assertIn("*you*", timestamped_content, "Cross-talk word 'you' should be marked with asterisks")
            
            # Verify CSV has cross-talk columns
            import csv
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
            
            self.assertEqual(len(rows[0]), 4, "CSV header should have 4 columns with cross-talk info")
            self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'])
    
    def test_cross_talk_performance(self):
        """Test that cross-talk detection doesn't significantly impact performance."""
        # Create a larger set of test data for performance testing
        large_words = []
        large_diar_segments = []
        
        # Generate 1000 words and segments
        for i in range(1000):
            start_time = i * 0.5
            end_time = start_time + 0.4
            large_words.append({
                "text": f"word{i}",
                "start": start_time,
                "end": end_time,
                "speaker": None
            })
            
            # Create overlapping segments for cross-talk testing
            speaker = "SpeakerA" if i % 2 == 0 else "SpeakerB"
            large_diar_segments.append({
                "start": start_time - 0.1,
                "end": end_time + 0.1,
                "label": speaker
            })
        
        # Test cross-talk detection performance
        start_time = time.time()
        
        # Detect cross-talk segments
        cross_talk_segments = detect_basic_cross_talk(large_diar_segments)
        
        # Assign words with cross-talk information
        enhanced_words = assign_words_with_basic_cross_talk(large_words, large_diar_segments, cross_talk_segments)
        
        # Build turns
        turns = build_turns(enhanced_words, speaker_label="SpeakerA")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify that processing completed successfully
        self.assertIsInstance(cross_talk_segments, list, "Should return a list of cross-talk segments")
        self.assertIsInstance(enhanced_words, list, "Should return a list of enhanced words")
        self.assertIsInstance(turns, list, "Should return a list of turns")
        
        # Verify that processing time is reasonable (should be much less than 1 second for this test data)
        self.assertLess(processing_time, 1.0, 
                       f"Cross-talk processing should be fast, but took {processing_time:.3f} seconds")
        
        # Verify that results are correct
        self.assertEqual(len(enhanced_words), len(large_words), 
                        "Should have same number of enhanced words as input words")
        
        # Verify that some words were marked as cross-talk (given the overlapping segments)
        cross_talk_words = [w for w in enhanced_words if w.get("cross_talk", False)]
        self.assertGreater(len(cross_talk_words), 0, "Should have some words marked as cross-talk")
        
        print(f"Cross-talk performance test: {len(large_words)} words processed in {processing_time:.3f} seconds")
    
    def test_cross_talk_pipeline_consistency(self):
        """Test that the cross-talk pipeline produces consistent results across multiple runs."""
        # Test data with known cross-talk segments
        test_words = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": None},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": None},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": None},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": None},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": None}
        ]
        
        test_segments = [
            {"start": 0.0, "end": 2.0, "label": "SpeakerA"},
            {"start": 1.5, "end": 3.0, "label": "SpeakerB"}
        ]
        
        # Run the cross-talk detection pipeline multiple times
        results = []
        for _ in range(5):  # Run 5 times to check consistency
            # Detect cross-talk segments
            cross_talk_segments = detect_basic_cross_talk(test_segments)
            
            # Assign words with cross-talk information
            enhanced_words = assign_words_with_basic_cross_talk(test_words, test_segments, cross_talk_segments)
            
            # Build turns
            turns = build_turns(enhanced_words, speaker_label="SpeakerA")
            
            results.append({
                "cross_talk_segments": cross_talk_segments,
                "enhanced_words": enhanced_words,
                "turns": turns
            })
        
        # Verify that all runs produce the same results
        first_result = results[0]
        
        for i, result in enumerate(results[1:], start=2):
            # Check cross-talk segments
            self.assertEqual(len(result["cross_talk_segments"]), len(first_result["cross_talk_segments"]),
                           f"Run {i} should have same number of cross-talk segments as run 1")
            
            # Check enhanced words
            self.assertEqual(len(result["enhanced_words"]), len(first_result["enhanced_words"]),
                           f"Run {i} should have same number of enhanced words as run 1")
            
            # Check turns
            self.assertEqual(len(result["turns"]), len(first_result["turns"]),
                           f"Run {i} should have same number of turns as run 1")
            
            # Check that cross-talk flags are consistent
            for j, (word1, word2) in enumerate(zip(result["enhanced_words"], first_result["enhanced_words"])):
                self.assertEqual(word1.get("cross_talk", False), word2.get("cross_talk", False),
                               f"Word {j} cross-talk flag should be consistent across runs")
                self.assertAlmostEqual(word1.get("confidence", 1.0), word2.get("confidence", 1.0), places=3,
                                     f"Word {j} confidence should be consistent across runs")
        
        print("Cross-talk pipeline consistency test: All 5 runs produced identical results")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing end-to-end cross-talk pipeline...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All end-to-end cross-talk tests passed! The complete pipeline works correctly.")
    else:
        print("❌ Some end-to-end cross-talk tests failed. Please review the implementation.")
        sys.exit(1)