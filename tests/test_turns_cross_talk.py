#!/usr/bin/env python3
"""
Test script for turns cross-talk functionality.
This script tests the turn building with cross-talk information.
"""

import sys
from pathlib import Path
import unittest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from turns import build_turns, _create_enhanced_turn


class TestTurnsCrossTalk(unittest.TestCase):
    """Test class for turns cross-talk functionality."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Sample words with cross-talk information
        self.words_with_cross_talk = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.5},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.3},
            {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.2},
            {"text": "fine", "start": 3.4, "end": 3.6, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.0},
            {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "Good", "start": 7.1, "end": 7.3, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "to", "start": 7.4, "end": 7.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "hear", "start": 7.6, "end": 7.8, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}
        ]
        
        # Sample words without cross-talk information (backward compatibility)
        self.words_without_cross_talk = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA"},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA"},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": "SpeakerA"},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": "SpeakerA"},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": "SpeakerA"},
            {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": "SpeakerA"},
            {"text": "fine", "start": 3.4, "end": 3.6, "speaker": "SpeakerA"},
            {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": "SpeakerA"},
            {"text": "Good", "start": 7.1, "end": 7.3, "speaker": "SpeakerA"},
            {"text": "to", "start": 7.4, "end": 7.5, "speaker": "SpeakerA"},
            {"text": "hear", "start": 7.6, "end": 7.8, "speaker": "SpeakerA"}
        ]
        
        # Sample words with mixed cross-talk content
        self.words_mixed_content = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.8},
            {"text": "How", "start": 1.6, "end": 1.8, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "are", "start": 1.9, "end": 2.1, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.5},
            {"text": "you", "start": 2.2, "end": 2.4, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.2},
            {"text": "fine", "start": 3.4, "end": 3.6, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.1}
        ]
    
    def test_turn_building_with_cross_talk(self):
        """Test that turns are correctly built with cross-talk flags and confidence scores."""
        turns = build_turns(self.words_with_cross_talk, speaker_label="SpeakerA")
        
        # Verify that turns were created
        self.assertIsInstance(turns, list, "Should return a list of turns")
        self.assertGreater(len(turns), 0, "Should have at least one turn")
        
        # Verify that each turn has cross-talk information
        for turn in turns:
            self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
            self.assertIn("confidence", turn, "Turn should have confidence key")
            self.assertIn("words", turn, "Turn should have words key")
            
            # Verify that confidence is within valid range [0, 1]
            self.assertGreaterEqual(turn["confidence"], 0.0, "Confidence should be >= 0")
            self.assertLessEqual(turn["confidence"], 1.0, "Confidence should be <= 1")
            
            # Verify that words are preserved
            self.assertIsInstance(turn["words"], list, "Words should be a list")
            self.assertEqual(len(turn["words"]), len(turn["text"].split()), 
                           "Number of words should match text split")
        
        # Check specific turns for cross-talk presence
        # First turn: "Hello world" - no cross-talk
        first_turn = turns[0]
        self.assertEqual(first_turn["text"], "Hello world", "First turn should be 'Hello world'")
        self.assertFalse(first_turn["cross_talk_present"], "First turn should not have cross-talk")
        self.assertEqual(first_turn["confidence"], 1.0, "First turn should have confidence 1.0")
        
        # Second turn: "How are you" - has cross-talk
        second_turn = turns[1]
        self.assertEqual(second_turn["text"], "How are you", "Second turn should be 'How are you'")
        self.assertTrue(second_turn["cross_talk_present"], "Second turn should have cross-talk")
        self.assertLess(second_turn["confidence"], 1.0, "Second turn should have confidence < 1.0")
        
        # Third turn: "I'm fine thanks" - has cross-talk
        third_turn = turns[2]
        self.assertEqual(third_turn["text"], "I'm fine thanks", "Third turn should be 'I'm fine thanks'")
        self.assertTrue(third_turn["cross_talk_present"], "Third turn should have cross-talk")
        self.assertLess(third_turn["confidence"], 1.0, "Third turn should have confidence < 1.0")
        
        # Fourth turn: "Good to hear" - no cross-talk
        fourth_turn = turns[3]
        self.assertEqual(fourth_turn["text"], "Good to hear", "Fourth turn should be 'Good to hear'")
        self.assertFalse(fourth_turn["cross_talk_present"], "Fourth turn should not have cross-talk")
        self.assertEqual(fourth_turn["confidence"], 1.0, "Fourth turn should have confidence 1.0")
    
    def test_turn_building_without_cross_talk(self):
        """Test that turn building maintains backward compatibility when words don't have cross-talk information."""
        turns = build_turns(self.words_without_cross_talk, speaker_label="SpeakerA")
        
        # Verify that turns were created
        self.assertIsInstance(turns, list, "Should return a list of turns")
        self.assertGreater(len(turns), 0, "Should have at least one turn")
        
        # Verify that each turn has cross-talk information (backward compatibility)
        for turn in turns:
            self.assertIn("cross_talk_present", turn, "Turn should have cross_talk_present key")
            self.assertIn("confidence", turn, "Turn should have confidence key")
            self.assertIn("words", turn, "Turn should have words key")
            
            # Without cross-talk information, these should be default values
            self.assertFalse(turn["cross_talk_present"], "Cross-talk should be False when words lack cross-talk info")
            self.assertEqual(turn["confidence"], 1.0, "Confidence should be 1.0 when words lack cross-talk info")
    
    def test_turn_building_mixed_content(self):
        """Test that turns with both cross-talk and non-cross-talk words are handled correctly."""
        turns = build_turns(self.words_mixed_content, speaker_label="SpeakerA")
        
        # Verify that turns were created
        self.assertIsInstance(turns, list, "Should return a list of turns")
        self.assertGreater(len(turns), 0, "Should have at least one turn")
        
        # Check that turns with mixed content have appropriate cross-talk flags
        for turn in turns:
            # If any word in the turn has cross-talk, the whole turn should be marked
            has_cross_talk_words = any(word.get("cross_talk", False) for word in turn["words"])
            self.assertEqual(turn["cross_talk_present"], has_cross_talk_words,
                           f"Turn cross-talk flag should match presence of cross-talk words: {turn['text']}")
            
            # If there are cross-talk words, confidence should be less than 1.0
            if has_cross_talk_words:
                self.assertLess(turn["confidence"], 1.0, 
                              f"Turn with cross-talk should have confidence < 1.0: {turn['text']}")
            else:
                self.assertEqual(turn["confidence"], 1.0, 
                               f"Turn without cross-talk should have confidence 1.0: {turn['text']}")
    
    def test_create_enhanced_turn_with_cross_talk(self):
        """Test the _create_enhanced_turn function with cross-talk information."""
        text_buf = ["Hello", "world"]
        word_buf = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.8}
        ]
        
        turn = _create_enhanced_turn("SpeakerA", 0.2, 0.9, text_buf, word_buf)
        
        # Verify turn structure
        self.assertEqual(turn["speaker"], "SpeakerA", "Speaker should be SpeakerA")
        self.assertEqual(turn["start"], 0.2, "Start time should be 0.2")
        self.assertEqual(turn["end"], 0.9, "End time should be 0.9")
        self.assertEqual(turn["text"], "Hello world", "Text should be 'Hello world'")
        
        # Verify cross-talk information
        self.assertTrue(turn["cross_talk_present"], "Turn should have cross-talk present")
        self.assertEqual(turn["confidence"], 0.9, "Confidence should be average of word confidences")
        self.assertEqual(turn["words"], word_buf, "Words should be preserved")
    
    def test_create_enhanced_turn_without_cross_talk(self):
        """Test the _create_enhanced_turn function without cross-talk information."""
        text_buf = ["Hello", "world"]
        word_buf = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA"},
            {"text": "world", "start": 0.6, "end": 0.9, "speaker": "SpeakerA"}
        ]
        
        turn = _create_enhanced_turn("SpeakerA", 0.2, 0.9, text_buf, word_buf)
        
        # Verify turn structure
        self.assertEqual(turn["speaker"], "SpeakerA", "Speaker should be SpeakerA")
        self.assertEqual(turn["start"], 0.2, "Start time should be 0.2")
        self.assertEqual(turn["end"], 0.9, "End time should be 0.9")
        self.assertEqual(turn["text"], "Hello world", "Text should be 'Hello world'")
        
        # Verify default cross-talk information (backward compatibility)
        self.assertFalse(turn["cross_talk_present"], "Cross-talk should be False by default")
        self.assertEqual(turn["confidence"], 1.0, "Confidence should be 1.0 by default")
        self.assertEqual(turn["words"], word_buf, "Words should be preserved")
    
    def test_create_enhanced_turn_confidence_calculation(self):
        """Test confidence calculation in _create_enhanced_turn function."""
        # Test with various confidence values
        test_cases = [
            # (word_confidences, expected_avg)
            ([1.0, 1.0], 1.0),
            ([0.5, 0.5], 0.5),
            ([0.0, 1.0], 0.5),
            ([0.2, 0.4, 0.6], 0.4),
            ([1.5, 0.5], 1.0),  # Should be clamped to 1.0
            ([-0.5, 0.5], 0.0),  # Should be clamped to 0.0
        ]
        
        for word_confidences, expected_avg in test_cases:
            text_buf = [f"word{i}" for i in range(len(word_confidences))]
            word_buf = [
                {"text": f"word{i}", "start": i*0.1, "end": (i+1)*0.1, "speaker": "SpeakerA", 
                 "cross_talk": True, "confidence": conf}
                for i, conf in enumerate(word_confidences)
            ]
            
            turn = _create_enhanced_turn("SpeakerA", 0.0, len(word_confidences)*0.1, text_buf, word_buf)
            
            self.assertEqual(turn["confidence"], expected_avg, 
                           f"Confidence should be {expected_avg} for word confidences {word_confidences}")
            self.assertGreaterEqual(turn["confidence"], 0.0, "Confidence should be >= 0")
            self.assertLessEqual(turn["confidence"], 1.0, "Confidence should be <= 1")
    
    def test_turn_building_empty_words(self):
        """Test turn building with empty words list."""
        turns = build_turns([], speaker_label="SpeakerA")
        
        # Verify that empty list returns empty turns
        self.assertIsInstance(turns, list, "Should return a list")
        self.assertEqual(len(turns), 0, "Should return empty list for empty words")
    
    def test_turn_building_single_word(self):
        """Test turn building with single word."""
        single_word = [{"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}]
        
        turns = build_turns(single_word, speaker_label="SpeakerA")
        
        # Verify that single word creates one turn
        self.assertIsInstance(turns, list, "Should return a list")
        self.assertEqual(len(turns), 1, "Should have one turn for single word")
        
        turn = turns[0]
        self.assertEqual(turn["text"], "Hello", "Turn text should be 'Hello'")
        self.assertFalse(turn["cross_talk_present"], "Turn should not have cross-talk")
        self.assertEqual(turn["confidence"], 1.0, "Turn confidence should be 1.0")
        self.assertEqual(len(turn["words"]), 1, "Turn should have one word")
    
    def test_turn_building_large_gaps(self):
        """Test turn building with large gaps between words."""
        words_with_gaps = [
            {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0},
            {"text": "world", "start": 2.0, "end": 2.3, "speaker": "SpeakerA", "cross_talk": True, "confidence": 0.8},  # Large gap
            {"text": "How", "start": 2.4, "end": 2.6, "speaker": "SpeakerA", "cross_talk": False, "confidence": 1.0}
        ]
        
        turns = build_turns(words_with_gaps, speaker_label="SpeakerA", max_gap_s=0.8)
        
        # Verify that large gaps create separate turns
        self.assertEqual(len(turns), 2, "Should have 2 turns due to large gap")
        
        # First turn: "Hello"
        self.assertEqual(turns[0]["text"], "Hello", "First turn should be 'Hello'")
        self.assertFalse(turns[0]["cross_talk_present"], "First turn should not have cross-talk")
        
        # Second turn: "world How"
        self.assertEqual(turns[1]["text"], "world How", "Second turn should be 'world How'")
        self.assertTrue(turns[1]["cross_talk_present"], "Second turn should have cross-talk")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTurnsCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing turns cross-talk functionality...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All turns cross-talk tests passed! Turn building with cross-talk information works correctly.")
    else:
        print("❌ Some turns cross-talk tests failed. Please review the implementation.")
        sys.exit(1)