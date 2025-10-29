#!/usr/bin/env python3
"""
Test script for basic cross-talk detection functionality.
This script tests the cross-talk detection and word marking features.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_talk import detect_basic_cross_talk, assign_words_with_basic_cross_talk, BASIC_CROSS_TALK_CONFIG
import unittest


class TestBasicCrossTalk(unittest.TestCase):
    """Test class for basic cross-talk detection functionality."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Sample diarization segments for testing
        self.diar_segments = [
            {"start": 0.0, "end": 2.0, "label": "SpeakerA"},
            {"start": 1.5, "end": 3.5, "label": "SpeakerB"},
            {"start": 3.0, "end": 5.0, "label": "SpeakerA"},
            {"start": 4.5, "end": 6.0, "label": "SpeakerB"},
            {"start": 7.0, "end": 9.0, "label": "SpeakerA"}
        ]
        
        # Sample words for testing
        self.words = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA"},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA"},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": "SpeakerB"},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": "SpeakerB"},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": "SpeakerB"},
            {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": "SpeakerA"},
            {"text": "fine", "start": 3.4, "end": 3.6, "speaker": "SpeakerA"},
            {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": "SpeakerA"},
            {"text": "What", "start": 4.6, "end": 4.8, "speaker": "SpeakerB"},
            {"text": "about", "start": 4.9, "end": 5.1, "speaker": "SpeakerB"},
            {"text": "you", "start": 5.2, "end": 5.4, "speaker": "SpeakerB"},
            {"text": "Good", "start": 7.1, "end": 7.3, "speaker": "SpeakerA"},
            {"text": "to", "start": 7.4, "end": 7.5, "speaker": "SpeakerA"},
            {"text": "hear", "start": 7.6, "end": 7.8, "speaker": "SpeakerA"}
        ]
        
        # Expected cross-talk segments from the diar_segments
        self.expected_cross_talk = [
            {"start": 1.5, "end": 2.0, "speakers": ["SpeakerA", "SpeakerB"], "duration": 0.5},
            {"start": 3.0, "end": 3.5, "speakers": ["SpeakerA", "SpeakerB"], "duration": 0.5},
            {"start": 4.5, "end": 5.0, "speakers": ["SpeakerA", "SpeakerB"], "duration": 0.5}
        ]
    
    def test_detect_basic_cross_talk_simple_overlap(self):
        """Test basic overlap detection with two speakers."""
        # Test with default threshold
        result = detect_basic_cross_talk(self.diar_segments)
        
        # Verify we detected the expected cross-talk segments
        self.assertEqual(len(result), 3, "Should detect 3 cross-talk segments")
        
        # Check first cross-talk segment
        self.assertEqual(result[0]["start"], 1.5, "First cross-talk should start at 1.5")
        self.assertEqual(result[0]["end"], 2.0, "First cross-talk should end at 2.0")
        self.assertEqual(result[0]["duration"], 0.5, "First cross-talk should have duration of 0.5")
        self.assertIn("SpeakerA", result[0]["speakers"], "First cross-talk should include SpeakerA")
        self.assertIn("SpeakerB", result[0]["speakers"], "First cross-talk should include SpeakerB")
        
        # Check second cross-talk segment
        self.assertEqual(result[1]["start"], 3.0, "Second cross-talk should start at 3.0")
        self.assertEqual(result[1]["end"], 3.5, "Second cross-talk should end at 3.5")
        
        # Check third cross-talk segment
        self.assertEqual(result[2]["start"], 4.5, "Third cross-talk should start at 4.5")
        self.assertEqual(result[2]["end"], 5.0, "Third cross-talk should end at 5.0")
    
    def test_detect_basic_cross_talk_no_overlap(self):
        """Test scenario with no overlapping segments."""
        # Create segments with no overlaps
        no_overlap_segments = [
            {"start": 0.0, "end": 1.0, "label": "SpeakerA"},
            {"start": 2.0, "end": 3.0, "label": "SpeakerB"},
            {"start": 4.0, "end": 5.0, "label": "SpeakerA"}
        ]
        
        result = detect_basic_cross_talk(no_overlap_segments)
        
        # Verify no cross-talk segments were detected
        self.assertEqual(len(result), 0, "Should detect no cross-talk segments when there are no overlaps")
    
    def test_detect_basic_cross_talk_threshold_filtering(self):
        """Test that very short overlaps are filtered out."""
        # Create segments with a very short overlap (0.05 seconds)
        short_overlap_segments = [
            {"start": 0.0, "end": 1.0, "label": "SpeakerA"},
            {"start": 0.95, "end": 2.0, "label": "SpeakerB"}  # Only 0.05 seconds overlap
        ]
        
        # Test with default threshold (0.1)
        result = detect_basic_cross_talk(short_overlap_segments)
        self.assertEqual(len(result), 0, "Should filter out overlaps shorter than threshold")
        
        # Test with lower threshold (0.05)
        result = detect_basic_cross_talk(short_overlap_segments, overlap_threshold=0.05)
        self.assertEqual(len(result), 1, "Should detect overlap when threshold is lowered")
        self.assertAlmostEqual(result[0]["duration"], 0.05, places=2, 
                              msg="Detected overlap should have correct duration")
    
    def test_assign_words_with_basic_cross_talk_marking(self):
        """Test that words are correctly marked with cross-talk flags."""
        # Get cross-talk segments
        cross_talk_segments = detect_basic_cross_talk(self.diar_segments)
        
        # Assign words with cross-talk marking
        result = assign_words_with_basic_cross_talk(self.words, self.diar_segments, cross_talk_segments)
        
        # Verify the result has the same number of words
        self.assertEqual(len(result), len(self.words), "Should return the same number of words")
        
        # Check that words during cross-talk are marked
        # Word "are" (1.9-2.1) should be marked as cross-talk
        are_word = next(w for w in result if w["text"] == "are")
        self.assertTrue(are_word["cross_talk"], "Word 'are' should be marked as cross-talk")
        
        # Word "I'm" (3.1-3.3) should be marked as cross-talk
        im_word = next(w for w in result if w["text"] == "I'm")
        self.assertTrue(im_word["cross_talk"], "Word 'I'm' should be marked as cross-talk")
        
        # Word "What" (4.6-4.8) should be marked as cross-talk
        what_word = next(w for w in result if w["text"] == "What")
        self.assertTrue(what_word["cross_talk"], "Word 'What' should be marked as cross-talk")
        
        # Word "Hello" (0.2-0.5) should NOT be marked as cross-talk
        hello_word = next(w for w in result if w["text"] == "Hello")
        self.assertFalse(hello_word["cross_talk"], "Word 'Hello' should not be marked as cross-talk")
        
        # Word "Good" (7.1-7.3) should NOT be marked as cross-talk
        good_word = next(w for w in result if w["text"] == "Good")
        self.assertFalse(good_word["cross_talk"], "Word 'Good' should not be marked as cross-talk")
    
    def test_assign_words_with_basic_cross_talk_confidence(self):
        """Test confidence score calculation."""
        # Get cross-talk segments
        cross_talk_segments = detect_basic_cross_talk(self.diar_segments)
        
        # Assign words with cross-talk marking and confidence
        result = assign_words_with_basic_cross_talk(self.words, self.diar_segments, cross_talk_segments)
        
        # Check confidence for words with cross-talk
        # Word "are" (1.9-2.1) has duration 0.2, overlap is 0.1 (50% overlap)
        are_word = next(w for w in result if w["text"] == "are")
        self.assertAlmostEqual(are_word["confidence"], 0.5, places=2, 
                              msg="Confidence should be 0.5 for 50% overlap")
        
        # Word "I'm" (3.1-3.3) has duration 0.2, overlap is 0.2 (100% overlap)
        im_word = next(w for w in result if w["text"] == "I'm")
        self.assertAlmostEqual(im_word["confidence"], 0.0, places=2, 
                              msg="Confidence should be 0.0 for 100% overlap")
        
        # Word "What" (4.6-4.8) has duration 0.2, overlap is 0.2 (100% overlap)
        what_word = next(w for w in result if w["text"] == "What")
        self.assertAlmostEqual(what_word["confidence"], 0.0, places=2, 
                              msg="Confidence should be 0.0 for 100% overlap")
        
        # Check confidence for words without cross-talk
        hello_word = next(w for w in result if w["text"] == "Hello")
        self.assertEqual(hello_word["confidence"], 1.0, "Confidence should be 1.0 for no cross-talk")
        
        good_word = next(w for w in result if w["text"] == "Good")
        self.assertEqual(good_word["confidence"], 1.0, "Confidence should be 1.0 for no cross-talk")
    
    def test_assign_words_with_basic_cross_talk_edge_cases(self):
        """Test edge cases like very short words."""
        # Create a word that's too short (below min_word_duration)
        short_word = {"text": "a", "start": 1.5, "end": 1.52, "speaker": "SpeakerB"}  # 0.02 seconds
        words_with_short = self.words + [short_word]
        
        # Get cross-talk segments
        cross_talk_segments = detect_basic_cross_talk(self.diar_segments)
        
        # Assign words with cross-talk marking
        result = assign_words_with_basic_cross_talk(words_with_short, self.diar_segments, cross_talk_segments)
        
        # Find the short word in the result
        short_word_result = next(w for w in result if w["text"] == "a")
        
        # Verify the short word is not marked as cross-talk and has default confidence
        self.assertFalse(short_word_result["cross_talk"], "Very short word should not be marked as cross-talk")
        self.assertEqual(short_word_result["confidence"], 1.0, "Very short word should have default confidence")
        
        # Test word that exactly matches the minimum duration
        min_duration_word = {"text": "min", "start": 1.5, "end": 1.55, "speaker": "SpeakerB"}  # 0.05 seconds
        words_with_min = self.words + [min_duration_word]
        
        result = assign_words_with_basic_cross_talk(words_with_min, self.diar_segments, cross_talk_segments)
        min_word_result = next(w for w in result if w["text"] == "min")
        
        # This word should be processed since it meets the minimum duration
        self.assertTrue(min_word_result["cross_talk"], "Word at minimum duration should be marked as cross-talk")
        self.assertLess(min_word_result["confidence"], 1.0, "Word at minimum duration should have reduced confidence")
    
    def test_basic_cross_talk_config(self):
        """Test that configuration parameters are applied correctly."""
        # Test default configuration
        self.assertEqual(BASIC_CROSS_TALK_CONFIG["overlap_threshold"], 0.1, "Default overlap threshold should be 0.1")
        self.assertEqual(BASIC_CROSS_TALK_CONFIG["confidence_threshold"], 0.6, "Default confidence threshold should be 0.6")
        self.assertEqual(BASIC_CROSS_TALK_CONFIG["min_word_duration"], 0.05, "Default min word duration should be 0.05")
        self.assertTrue(BASIC_CROSS_TALK_CONFIG["mark_cross_talk"], "Default mark_cross_talk should be True")
        self.assertTrue(BASIC_CROSS_TALK_CONFIG["basic_confidence"], "Default basic_confidence should be True")
        
        # Test that the configuration is used in the functions
        # Create a word with duration exactly at the minimum threshold
        min_word = {"text": "test", "start": 1.5, "end": 1.55, "speaker": "SpeakerB"}  # 0.05 seconds
        words = [min_word]
        
        # Get cross-talk segments
        cross_talk_segments = detect_basic_cross_talk(self.diar_segments)
        
        # This word should be processed since it meets the minimum duration
        result = assign_words_with_basic_cross_talk(words, self.diar_segments, cross_talk_segments)
        self.assertEqual(len(result), 1, "Should return one word")
        
        # The word should be marked as cross-talk since it's in the overlap region
        self.assertTrue(result[0]["cross_talk"], "Word at minimum duration should be marked as cross-talk")
        
        # Test with a word shorter than the minimum
        too_short_word = {"text": "a", "start": 1.5, "end": 1.54, "speaker": "SpeakerB"}  # 0.04 seconds
        words = [too_short_word]
        
        result = assign_words_with_basic_cross_talk(words, self.diar_segments, cross_talk_segments)
        # The word should not be marked as cross-talk since it's below the minimum duration
        self.assertFalse(result[0]["cross_talk"], "Word below minimum duration should not be marked as cross-talk")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBasicCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing basic cross-talk detection functionality...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed! Basic cross-talk detection is working correctly.")
    else:
        print("❌ Some tests failed. Please review the implementation.")
        sys.exit(1)