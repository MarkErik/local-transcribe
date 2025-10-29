#!/usr/bin/env python3
"""
Test script for txt_writer cross-talk functionality.
This script tests the text writer output formatting with cross-talk features.
"""

import sys
from pathlib import Path
import tempfile
import unittest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from txt_writer import write_timestamped_txt, write_plain_txt, write_asr_words, write_txt


class TestTxtWriterCrossTalk(unittest.TestCase):
    """Test class for txt_writer cross-talk functionality."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample turns with cross-talk information
        self.turns_with_cross_talk = [
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
            },
            {
                "speaker": "SpeakerA",
                "start": 3.0,
                "end": 5.0,
                "text": "I'm fine thanks",
                "cross_talk_present": True,
                "confidence": 0.2,
                "words": [
                    {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.2},
                    {"text": "fine", "start": 3.4, "end": 3.6, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.0},
                    {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}
                ]
            },
            {
                "speaker": "SpeakerB",
                "start": 7.0,
                "end": 9.0,
                "text": "Good to hear",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [
                    {"text": "Good", "start": 7.1, "end": 7.3, "speaker": "SpeakerB", "cross_talk": False, "confidence": 1.0},
                    {"text": "to", "start": 7.4, "end": 7.5, "speaker": "SpeakerB", "cross_talk": False, "confidence": 1.0},
                    {"text": "hear", "start": 7.6, "end": 7.8, "speaker": "SpeakerB", "cross_talk": False, "confidence": 1.0}
                ]
            }
        ]
        
        # Sample turns without cross-talk information (backward compatibility)
        self.turns_without_cross_talk = [
            {
                "speaker": "SpeakerA",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA"},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA"}
                ]
            },
            {
                "speaker": "SpeakerB",
                "start": 1.5,
                "end": 3.5,
                "text": "How are you",
                "words": [
                    {"text": "How", "start": 1.6, "end": 1.8, "speaker": "SpeakerB"},
                    {"text": "are", "start": 1.9, "end": 2.1, "speaker": "SpeakerB"},
                    {"text": "you", "start": 2.2, "end": 2.4, "speaker": "SpeakerB"}
                ]
            }
        ]
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_txt_writer_with_cross_talk_timestamped(self):
        """Test that text writer correctly marks cross-talk words with asterisks in timestamped format."""
        output_path = Path(self.temp_dir) / "timestamped_with_cross_talk.txt"
        
        # Write timestamped text with cross-talk marking
        write_timestamped_txt(self.turns_with_cross_talk, output_path, mark_cross_talk=True)
        
        # Read the output file
        content = output_path.read_text(encoding="utf-8")
        
        # Verify that cross-talk words are marked with asterisks
        self.assertIn("*are*", content, "Cross-talk word 'are' should be marked with asterisks")
        self.assertIn("*you*", content, "Cross-talk word 'you' should be marked with asterisks")
        self.assertIn("*I'm*", content, "Cross-talk word 'I'm' should be marked with asterisks")
        self.assertIn("*fine*", content, "Cross-talk word 'fine' should be marked with asterisks")
        
        # Verify that non-cross-talk words are not marked
        self.assertIn("Hello", content, "Non-cross-talk word 'Hello' should not be marked")
        self.assertIn("world", content, "Non-cross-talk word 'world' should not be marked")
        self.assertIn("How", content, "Non-cross-talk word 'How' should not be marked")
        self.assertIn("Good", content, "Non-cross-talk word 'Good' should not be marked")
        self.assertIn("to", content, "Non-cross-talk word 'to' should not be marked")
        self.assertIn("hear", content, "Non-cross-talk word 'hear' should not be marked")
        
        # Verify that asterisks are only around cross-talk words
        lines = content.strip().split('\n')
        for line in lines:
            if "SpeakerA: Hello world" in line:
                self.assertNotIn("*", line, "Line without cross-talk should not have asterisks")
            elif "SpeakerB: How *are* *you*" in line:
                self.assertEqual(line.count("*"), 4, "Line should have exactly 4 asterisks (2 words)")
            elif "SpeakerA: *I'm* *fine* thanks" in line:
                self.assertEqual(line.count("*"), 4, "Line should have exactly 4 asterisks (2 words)")
            elif "SpeakerB: Good to hear" in line:
                self.assertNotIn("*", line, "Line without cross-talk should not have asterisks")
    
    def test_txt_writer_without_cross_talk_timestamped(self):
        """Test that text writer maintains backward compatibility when cross-talk marking is disabled."""
        output_path = Path(self.temp_dir) / "timestamped_without_cross_talk.txt"
        
        # Write timestamped text without cross-talk marking
        write_timestamped_txt(self.turns_with_cross_talk, output_path, mark_cross_talk=False)
        
        # Read the output file
        content = output_path.read_text(encoding="utf-8")
        
        # Verify that no words are marked with asterisks
        self.assertNotIn("*", content, "No words should be marked with asterisks when cross-talk marking is disabled")
        
        # Verify that all text is present
        self.assertIn("Hello world", content, "Text should be present")
        self.assertIn("How are you", content, "Text should be present")
        self.assertIn("I'm fine thanks", content, "Text should be present")
        self.assertIn("Good to hear", content, "Text should be present")
    
    def test_txt_writer_with_cross_talk_plain(self):
        """Test that text writer correctly marks cross-talk words with asterisks in plain format."""
        output_path = Path(self.temp_dir) / "plain_with_cross_talk.txt"
        
        # Write plain text with cross-talk marking
        write_plain_txt(self.turns_with_cross_talk, output_path, mark_cross_talk=True)
        
        # Read the output file
        content = output_path.read_text(encoding="utf-8")
        
        # Verify that cross-talk words are marked with asterisks
        self.assertIn("*are*", content, "Cross-talk word 'are' should be marked with asterisks")
        self.assertIn("*you*", content, "Cross-talk word 'you' should be marked with asterisks")
        self.assertIn("*I'm*", content, "Cross-talk word 'I'm' should be marked with asterisks")
        self.assertIn("*fine*", content, "Cross-talk word 'fine' should be marked with asterisks")
        
        # Verify that non-cross-talk words are not marked
        self.assertIn("Hello", content, "Non-cross-talk word 'Hello' should not be marked")
        self.assertIn("world", content, "Non-cross-talk word 'world' should not be marked")
        self.assertIn("How", content, "Non-cross-talk word 'How' should not be marked")
        self.assertIn("Good", content, "Non-cross-talk word 'Good' should not be marked")
        self.assertIn("to", content, "Non-cross-talk word 'to' should not be marked")
        self.assertIn("hear", content, "Non-cross-talk word 'hear' should not be marked")
    
    def test_txt_writer_without_cross_talk_plain(self):
        """Test that text writer maintains backward compatibility when cross-talk marking is disabled in plain format."""
        output_path = Path(self.temp_dir) / "plain_without_cross_talk.txt"
        
        # Write plain text without cross-talk marking
        write_plain_txt(self.turns_with_cross_talk, output_path, mark_cross_talk=False)
        
        # Read the output file
        content = output_path.read_text(encoding="utf-8")
        
        # Verify that no words are marked with asterisks
        self.assertNotIn("*", content, "No words should be marked with asterisks when cross-talk marking is disabled")
        
        # Verify that all text is present
        self.assertIn("Hello world", content, "Text should be present")
        self.assertIn("How are you", content, "Text should be present")
        self.assertIn("I'm fine thanks", content, "Text should be present")
        self.assertIn("Good to hear", content, "Text should be present")
    
    def test_asr_words_with_cross_talk(self):
        """Test that ASR words writer correctly marks cross-talk words with asterisks."""
        output_path = Path(self.temp_dir) / "asr_words_with_cross_talk.txt"
        
        # Extract words from turns
        words = []
        for turn in self.turns_with_cross_talk:
            words.extend(turn["words"])
        
        # Write ASR words with cross-talk marking
        write_asr_words(words, output_path, mark_cross_talk=True)
        
        # Read the output file
        content = output_path.read_text(encoding="utf-8")
        
        # Verify that cross-talk words are marked with asterisks
        self.assertIn("*are*", content, "Cross-talk word 'are' should be marked with asterisks")
        self.assertIn("*you*", content, "Cross-talk word 'you' should be marked with asterisks")
        self.assertIn("*I'm*", content, "Cross-talk word 'I'm' should be marked with asterisks")
        self.assertIn("*fine*", content, "Cross-talk word 'fine' should be marked with asterisks")
        
        # Verify that non-cross-talk words are not marked
        self.assertIn("Hello", content, "Non-cross-talk word 'Hello' should not be marked")
        self.assertIn("world", content, "Non-cross-talk word 'world' should not be marked")
        self.assertIn("How", content, "Non-cross-talk word 'How' should not be marked")
        self.assertIn("Good", content, "Non-cross-talk word 'Good' should not be marked")
        self.assertIn("to", content, "Non-cross-talk word 'to' should not be marked")
        self.assertIn("hear", content, "Non-cross-talk word 'hear' should not be marked")
    
    def test_asr_words_without_cross_talk(self):
        """Test that ASR words writer maintains backward compatibility when cross-talk marking is disabled."""
        output_path = Path(self.temp_dir) / "asr_words_without_cross_talk.txt"
        
        # Extract words from turns
        words = []
        for turn in self.turns_with_cross_talk:
            words.extend(turn["words"])
        
        # Write ASR words without cross-talk marking
        write_asr_words(words, output_path, mark_cross_talk=False)
        
        # Read the output file
        content = output_path.read_text(encoding="utf-8")
        
        # Verify that no words are marked with asterisks
        self.assertNotIn("*", content, "No words should be marked with asterisks when cross-talk marking is disabled")
        
        # Verify that all text is present
        expected_words = ["Hello", "world", "How", "are", "you", "I'm", "fine", "thanks", "Good", "to", "hear"]
        for word in expected_words:
            self.assertIn(word, content, f"Word '{word}' should be present in output")
    
    def test_txt_writer_backward_compatibility(self):
        """Test that text writer works with turns that don't have cross-talk information."""
        output_path_timestamped = Path(self.temp_dir) / "timestamped_backward_compat.txt"
        output_path_plain = Path(self.temp_dir) / "plain_backward_compat.txt"
        
        # Write timestamped text with turns without cross-talk info
        write_timestamped_txt(self.turns_without_cross_talk, output_path_timestamped, mark_cross_talk=True)
        
        # Write plain text with turns without cross-talk info
        write_plain_txt(self.turns_without_cross_talk, output_path_plain, mark_cross_talk=True)
        
        # Read the output files
        timestamped_content = output_path_timestamped.read_text(encoding="utf-8")
        plain_content = output_path_plain.read_text(encoding="utf-8")
        
        # Verify that no asterisks are present (since there's no cross-talk info)
        self.assertNotIn("*", timestamped_content, "No asterisks should be present when turns lack cross-talk info")
        self.assertNotIn("*", plain_content, "No asterisks should be present when turns lack cross-talk info")
        
        # Verify that all text is present
        self.assertIn("Hello world", timestamped_content, "Text should be present in timestamped output")
        self.assertIn("How are you", timestamped_content, "Text should be present in timestamped output")
        self.assertIn("Hello world", plain_content, "Text should be present in plain output")
        self.assertIn("How are you", plain_content, "Text should be present in plain output")
    
    def test_write_txt_with_cross_talk(self):
        """Test the main write_txt function with cross-talk marking."""
        output_path_timestamped = Path(self.temp_dir) / "txt_main_timestamped.txt"
        output_path_plain = Path(self.temp_dir) / "txt_main_plain.txt"
        
        # Test timestamped output with cross-talk marking
        write_txt(self.turns_with_cross_talk, str(output_path_timestamped), mark_cross_talk=True, timestamped=True)
        
        # Test plain output with cross-talk marking
        write_txt(self.turns_with_cross_talk, str(output_path_plain), mark_cross_talk=True, timestamped=False)
        
        # Read the output files
        timestamped_content = output_path_timestamped.read_text(encoding="utf-8")
        plain_content = output_path_plain.read_text(encoding="utf-8")
        
        # Verify that cross-talk words are marked in both outputs
        for content in [timestamped_content, plain_content]:
            self.assertIn("*are*", content, "Cross-talk word 'are' should be marked with asterisks")
            self.assertIn("*you*", content, "Cross-talk word 'you' should be marked with asterisks")
            self.assertIn("*I'm*", content, "Cross-talk word 'I'm' should be marked with asterisks")
            self.assertIn("*fine*", content, "Cross-talk word 'fine' should be marked with asterisks")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTxtWriterCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing txt_writer cross-talk functionality...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All txt_writer cross-talk tests passed! Text output formatting works correctly.")
    else:
        print("❌ Some txt_writer cross-talk tests failed. Please review the implementation.")
        sys.exit(1)