#!/usr/bin/env python3
"""
Test script for csv_writer cross-talk functionality.
This script tests the CSV writer output formatting with cross-talk features.
"""

import sys
from pathlib import Path
import tempfile
import csv
import unittest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from csv_writer import write_conversation_csv, write_csv


class TestCsvWriterCrossTalk(unittest.TestCase):
    """Test class for csv_writer cross-talk functionality."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample turns with cross-talk information
        self.turns_with_cross_talk = [
            {
                "speaker": "Interviewer",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "Interviewer", "cross_talk": False, "confidence": 1.0},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "Interviewer", "cross_talk": False, "confidence": 1.0}
                ]
            },
            {
                "speaker": "Participant",
                "start": 1.5,
                "end": 3.5,
                "text": "How are you",
                "cross_talk_present": True,
                "confidence": 0.5,
                "words": [
                    {"text": "How", "start": 1.6, "end": 1.8, "speaker": "Participant", "cross_talk": False, "confidence": 1.0},
                    {"text": "are", "start": 1.9, "end": 2.1, "speaker": "Participant", "cross_talk": True, "confidence": 0.5},
                    {"text": "you", "start": 2.2, "end": 2.4, "speaker": "Participant", "cross_talk": True, "confidence": 0.3}
                ]
            },
            {
                "speaker": "Interviewer",
                "start": 3.0,
                "end": 5.0,
                "text": "I'm fine thanks",
                "cross_talk_present": True,
                "confidence": 0.2,
                "words": [
                    {"text": "I'm", "start": 3.1, "end": 3.3, "speaker": "Interviewer", "cross_talk": True, "confidence": 0.2},
                    {"text": "fine", "start": 3.4, "end": 3.6, "speaker": "Interviewer", "cross_talk": True, "confidence": 0.0},
                    {"text": "thanks", "start": 3.7, "end": 4.0, "speaker": "Interviewer", "cross_talk": False, "confidence": 1.0}
                ]
            },
            {
                "speaker": "Participant",
                "start": 7.0,
                "end": 9.0,
                "text": "Good to hear",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [
                    {"text": "Good", "start": 7.1, "end": 7.3, "speaker": "Participant", "cross_talk": False, "confidence": 1.0},
                    {"text": "to", "start": 7.4, "end": 7.5, "speaker": "Participant", "cross_talk": False, "confidence": 1.0},
                    {"text": "hear", "start": 7.6, "end": 7.8, "speaker": "Participant", "cross_talk": False, "confidence": 1.0}
                ]
            }
        ]
        
        # Sample turns without cross-talk information (backward compatibility)
        self.turns_without_cross_talk = [
            {
                "speaker": "Interviewer",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "words": [
                    {"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "Interviewer"},
                    {"text": "world", "start": 0.6, "end": 0.9, "speaker": "Interviewer"}
                ]
            },
            {
                "speaker": "Participant",
                "start": 1.5,
                "end": 3.5,
                "text": "How are you",
                "words": [
                    {"text": "How", "start": 1.6, "end": 1.8, "speaker": "Participant"},
                    {"text": "are", "start": 1.9, "end": 2.1, "speaker": "Participant"},
                    {"text": "you", "start": 2.2, "end": 2.4, "speaker": "Participant"}
                ]
            }
        ]
        
        # Sample turns with mixed speaker names (not just Interviewer/Participant)
        self.turns_mixed_speakers = [
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
    
    def tearDown(self):
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_writer_with_cross_talk(self):
        """Test that CSV writer includes cross-talk columns when enabled."""
        output_path = Path(self.temp_dir) / "with_cross_talk.csv"
        
        # Write CSV with cross-talk columns
        write_conversation_csv(self.turns_with_cross_talk, output_path, include_cross_talk=True)
        
        # Read the CSV file
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Verify header includes cross-talk columns
        self.assertEqual(len(rows[0]), 4, "Header should have 4 columns when cross-talk is enabled")
        self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'], 
                         "Header should include cross-talk columns")
        
        # Verify data rows have cross-talk information
        self.assertEqual(len(rows), 5, "Should have 4 data rows plus header")
        
        # Check first row (Interviewer: Hello world)
        self.assertEqual(rows[1][0], "Hello world", "First row should have interviewer text")
        self.assertEqual(rows[1][1], "", "First row should have empty participant column")
        self.assertEqual(rows[1][2], "False", "First row should have False cross_talk")
        self.assertEqual(rows[1][3], "1.000", "First row should have 1.000 confidence")
        
        # Check second row (Participant: How are you)
        self.assertEqual(rows[2][0], "", "Second row should have empty interviewer column")
        self.assertEqual(rows[2][1], "How are you", "Second row should have participant text")
        self.assertEqual(rows[2][2], "True", "Second row should have True cross_talk")
        self.assertEqual(rows[2][3], "0.500", "Second row should have 0.500 confidence")
        
        # Check third row (Interviewer: I'm fine thanks)
        self.assertEqual(rows[3][0], "I'm fine thanks", "Third row should have interviewer text")
        self.assertEqual(rows[3][1], "", "Third row should have empty participant column")
        self.assertEqual(rows[3][2], "True", "Third row should have True cross_talk")
        self.assertEqual(rows[3][3], "0.200", "Third row should have 0.200 confidence")
        
        # Check fourth row (Participant: Good to hear)
        self.assertEqual(rows[4][0], "", "Fourth row should have empty interviewer column")
        self.assertEqual(rows[4][1], "Good to hear", "Fourth row should have participant text")
        self.assertEqual(rows[4][2], "False", "Fourth row should have False cross_talk")
        self.assertEqual(rows[4][3], "1.000", "Fourth row should have 1.000 confidence")
    
    def test_csv_writer_without_cross_talk(self):
        """Test that CSV writer maintains backward compatibility when cross-talk columns are disabled."""
        output_path = Path(self.temp_dir) / "without_cross_talk.csv"
        
        # Write CSV without cross-talk columns
        write_conversation_csv(self.turns_with_cross_talk, output_path, include_cross_talk=False)
        
        # Read the CSV file
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Verify header does not include cross-talk columns
        self.assertEqual(len(rows[0]), 2, "Header should have 2 columns when cross-talk is disabled")
        self.assertEqual(rows[0], ['Interviewer', 'Participant'], 
                         "Header should not include cross-talk columns")
        
        # Verify data rows don't have cross-talk information
        self.assertEqual(len(rows), 5, "Should have 4 data rows plus header")
        
        # Check first row (Interviewer: Hello world)
        self.assertEqual(rows[1][0], "Hello world", "First row should have interviewer text")
        self.assertEqual(rows[1][1], "", "First row should have empty participant column")
        
        # Check second row (Participant: How are you)
        self.assertEqual(rows[2][0], "", "Second row should have empty interviewer column")
        self.assertEqual(rows[2][1], "How are you", "Second row should have participant text")
        
        # Check third row (Interviewer: I'm fine thanks)
        self.assertEqual(rows[3][0], "I'm fine thanks", "Third row should have interviewer text")
        self.assertEqual(rows[3][1], "", "Third row should have empty participant column")
        
        # Check fourth row (Participant: Good to hear)
        self.assertEqual(rows[4][0], "", "Fourth row should have empty interviewer column")
        self.assertEqual(rows[4][1], "Good to hear", "Fourth row should have participant text")
    
    def test_csv_writer_backward_compatibility(self):
        """Test that CSV writer works with turns that don't have cross-talk information."""
        output_path_with_cross_talk = Path(self.temp_dir) / "backward_compat_with_cross_talk.csv"
        output_path_without_cross_talk = Path(self.temp_dir) / "backward_compat_without_cross_talk.csv"
        
        # Write CSV with cross-talk columns enabled but turns without cross-talk info
        write_conversation_csv(self.turns_without_cross_talk, output_path_with_cross_talk, include_cross_talk=True)
        
        # Write CSV without cross-talk columns
        write_conversation_csv(self.turns_without_cross_talk, output_path_without_cross_talk, include_cross_talk=False)
        
        # Read the CSV files
        with open(output_path_with_cross_talk, 'r', encoding='utf-8') as csvfile:
            reader_with = csv.reader(csvfile)
            rows_with = list(reader_with)
        
        with open(output_path_without_cross_talk, 'r', encoding='utf-8') as csvfile:
            reader_without = csv.reader(csvfile)
            rows_without = list(reader_without)
        
        # Verify both files have the same content (since turns don't have cross-talk info)
        self.assertEqual(len(rows_with), len(rows_without), "Both files should have same number of rows")
        
        # Verify headers
        self.assertEqual(rows_with[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'], 
                         "With cross-talk: Header should include cross-talk columns")
        self.assertEqual(rows_without[0], ['Interviewer', 'Participant'], 
                         "Without cross-talk: Header should not include cross-talk columns")
        
        # Verify data content is the same (except for extra columns)
        for i in range(1, len(rows_with)):
            # Text content should be the same
            self.assertEqual(rows_with[i][0], rows_without[i][0], f"Row {i} interviewer text should match")
            self.assertEqual(rows_with[i][1], rows_without[i][1], f"Row {i} participant text should match")
            
            # Cross-talk columns should have default values
            self.assertEqual(rows_with[i][2], "False", f"Row {i} cross_talk should be False by default")
            self.assertEqual(rows_with[i][3], "1.000", f"Row {i} confidence should be 1.000 by default")
    
    def test_csv_writer_mixed_speakers(self):
        """Test that CSV writer handles non-Interviewer/Participant speaker names correctly."""
        output_path = Path(self.temp_dir) / "mixed_speakers.csv"
        
        # Write CSV with cross-talk columns
        write_conversation_csv(self.turns_mixed_speakers, output_path, include_cross_talk=True)
        
        # Read the CSV file
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Verify header
        self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'], 
                         "Header should include cross-talk columns")
        
        # Verify data rows - non-standard speakers should go to interviewer column
        self.assertEqual(len(rows), 3, "Should have 2 data rows plus header")
        
        # Check first row (SpeakerA: Hello world)
        self.assertEqual(rows[1][0], "Hello world", "First row should have SpeakerA text in interviewer column")
        self.assertEqual(rows[1][1], "", "First row should have empty participant column")
        self.assertEqual(rows[1][2], "False", "First row should have False cross_talk")
        self.assertEqual(rows[1][3], "1.000", "First row should have 1.000 confidence")
        
        # Check second row (SpeakerB: How are you)
        self.assertEqual(rows[2][0], "How are you", "Second row should have SpeakerB text in interviewer column")
        self.assertEqual(rows[2][1], "", "Second row should have empty participant column")
        self.assertEqual(rows[2][2], "True", "Second row should have True cross_talk")
        self.assertEqual(rows[2][3], "0.500", "Second row should have 0.500 confidence")
    
    def test_csv_writer_empty_turns(self):
        """Test that CSV writer handles empty turns list correctly."""
        output_path = Path(self.temp_dir) / "empty_turns.csv"
        
        # Write CSV with empty turns list
        write_conversation_csv([], output_path, include_cross_talk=True)
        
        # Read the CSV file
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Verify only header is present
        self.assertEqual(len(rows), 1, "Should have only header row")
        self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'], 
                         "Header should include cross-talk columns")
    
    def test_csv_writer_consecutive_same_speaker(self):
        """Test that CSV writer correctly merges consecutive turns from the same speaker."""
        # Create turns with consecutive same speaker
        consecutive_turns = [
            {
                "speaker": "Interviewer",
                "start": 0.0,
                "end": 2.0,
                "text": "Hello",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [{"text": "Hello", "start": 0.2, "end": 0.5, "speaker": "Interviewer", "cross_talk": False, "confidence": 1.0}]
            },
            {
                "speaker": "Interviewer",
                "start": 2.1,
                "end": 4.0,
                "text": "world",
                "cross_talk_present": False,
                "confidence": 1.0,
                "words": [{"text": "world", "start": 2.2, "end": 2.4, "speaker": "Interviewer", "cross_talk": False, "confidence": 1.0}]
            },
            {
                "speaker": "Participant",
                "start": 4.1,
                "end": 6.0,
                "text": "How are you",
                "cross_talk_present": True,
                "confidence": 0.5,
                "words": [
                    {"text": "How", "start": 4.2, "end": 4.4, "speaker": "Participant", "cross_talk": False, "confidence": 1.0},
                    {"text": "are", "start": 4.5, "end": 4.7, "speaker": "Participant", "cross_talk": True, "confidence": 0.5},
                    {"text": "you", "start": 4.8, "end": 5.0, "speaker": "Participant", "cross_talk": True, "confidence": 0.3}
                ]
            }
        ]
        
        output_path = Path(self.temp_dir) / "consecutive_speakers.csv"
        
        # Write CSV with cross-talk columns
        write_conversation_csv(consecutive_turns, output_path, include_cross_talk=True)
        
        # Read the CSV file
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Verify consecutive same speaker turns are merged
        self.assertEqual(len(rows), 3, "Should have 2 data rows plus header (consecutive same speaker merged)")
        
        # Check first row (Interviewer: Hello world - merged)
        self.assertEqual(rows[1][0], "Hello world", "First row should have merged interviewer text")
        self.assertEqual(rows[1][1], "", "First row should have empty participant column")
        self.assertEqual(rows[1][2], "False", "First row should have False cross_talk (merged from non-cross-talk)")
        self.assertEqual(rows[1][3], "1.000", "First row should have 1.000 confidence (merged from non-cross-talk)")
        
        # Check second row (Participant: How are you)
        self.assertEqual(rows[2][0], "", "Second row should have empty interviewer column")
        self.assertEqual(rows[2][1], "How are you", "Second row should have participant text")
        self.assertEqual(rows[2][2], "True", "Second row should have True cross_talk")
        self.assertEqual(rows[2][3], "0.500", "Second row should have 0.500 confidence")
    
    def test_write_csv_alias(self):
        """Test that the write_csv alias function works correctly."""
        output_path = Path(self.temp_dir) / "alias_function.csv"
        
        # Use the alias function
        write_csv(self.turns_with_cross_talk, str(output_path), include_cross_talk=True)
        
        # Read the CSV file
        with open(output_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        
        # Verify it works the same as write_conversation_csv
        self.assertEqual(len(rows[0]), 4, "Header should have 4 columns when cross-talk is enabled")
        self.assertEqual(rows[0], ['Interviewer', 'Participant', 'cross_talk', 'confidence'], 
                         "Header should include cross-talk columns")
        
        # Verify data is correct
        self.assertEqual(len(rows), 5, "Should have 4 data rows plus header")
        self.assertEqual(rows[1][0], "Hello world", "First row should have interviewer text")
        self.assertEqual(rows[2][1], "How are you", "Second row should have participant text")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCsvWriterCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing csv_writer cross-talk functionality...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All csv_writer cross-talk tests passed! CSV output formatting works correctly.")
    else:
        print("❌ Some csv_writer cross-talk tests failed. Please review the implementation.")
        sys.exit(1)