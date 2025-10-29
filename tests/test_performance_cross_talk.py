#!/usr/bin/env python3
"""
Performance tests for cross-talk detection functionality.
This script tests that cross-talk detection doesn't significantly impact performance.
"""

import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diarize import diarize_mixed
from cross_talk import detect_basic_cross_talk, assign_words_with_basic_cross_talk
from turns import build_turns


class TestPerformanceCrossTalk(unittest.TestCase):
    """Test class for cross-talk performance."""
    
    def setUp(self):
        """Set up test data for each test method."""
        # Create test data of various sizes
        self.small_data = self._create_test_data(100)  # 100 words
        self.medium_data = self._create_test_data(1000)  # 1000 words
        self.large_data = self._create_test_data(5000)  # 5000 words
        
        # Create a mock audio file
        self.temp_dir = Path("/tmp")  # Use /tmp for performance tests
        self.mock_audio_path = self.temp_dir / "test_audio.wav"
    
    def _create_test_data(self, num_words):
        """Create test data with specified number of words."""
        words = []
        segments = []
        
        for i in range(num_words):
            # Create words with 0.3 second duration each
            start_time = i * 0.4  # 0.1 second gap between words
            end_time = start_time + 0.3
            
            words.append({
                "text": f"word{i}",
                "start": start_time,
                "end": end_time,
                "speaker": None
            })
            
            # Create segments with overlaps for cross-talk testing
            if i % 10 == 0:  # Create a segment every 10 words
                speaker = "SpeakerA" if (i // 10) % 2 == 0 else "SpeakerB"
                segments.append({
                    "start": start_time - 0.1,
                    "end": end_time + 3.0,  # Make segments long to create overlaps
                    "label": speaker
                })
        
        return {"words": words, "segments": segments}
    
    def test_cross_talk_detection_performance_small(self):
        """Test cross-talk detection performance with small dataset."""
        data = self.small_data
        words = data["words"]
        segments = data["segments"]
        
        # Measure cross-talk detection performance
        start_time = time.time()
        cross_talk_segments = detect_basic_cross_talk(segments)
        detection_time = time.time() - start_time
        
        # Measure word assignment performance
        start_time = time.time()
        enhanced_words = assign_words_with_basic_cross_talk(words, segments, cross_talk_segments)
        assignment_time = time.time() - start_time
        
        # Measure turn building performance
        start_time = time.time()
        turns = build_turns(enhanced_words, speaker_label="SpeakerA")
        turn_building_time = time.time() - start_time
        
        total_time = detection_time + assignment_time + turn_building_time
        
        # Verify results are correct
        self.assertIsInstance(cross_talk_segments, list, "Should return list of cross-talk segments")
        self.assertIsInstance(enhanced_words, list, "Should return list of enhanced words")
        self.assertIsInstance(turns, list, "Should return list of turns")
        
        self.assertEqual(len(enhanced_words), len(words), "Should have same number of enhanced words")
        
        # Performance assertions for small dataset (should be very fast)
        self.assertLess(detection_time, 0.1, f"Cross-talk detection should be fast for small dataset: {detection_time:.4f}s")
        self.assertLess(assignment_time, 0.1, f"Word assignment should be fast for small dataset: {assignment_time:.4f}s")
        self.assertLess(turn_building_time, 0.1, f"Turn building should be fast for small dataset: {turn_building_time:.4f}s")
        self.assertLess(total_time, 0.3, f"Total processing should be fast for small dataset: {total_time:.4f}s")
        
        print(f"Small dataset ({len(words)} words) performance:")
        print(f"  - Cross-talk detection: {detection_time:.4f}s")
        print(f"  - Word assignment: {assignment_time:.4f}s")
        print(f"  - Turn building: {turn_building_time:.4f}s")
        print(f"  - Total: {total_time:.4f}s")
    
    def test_cross_talk_detection_performance_medium(self):
        """Test cross-talk detection performance with medium dataset."""
        data = self.medium_data
        words = data["words"]
        segments = data["segments"]
        
        # Measure cross-talk detection performance
        start_time = time.time()
        cross_talk_segments = detect_basic_cross_talk(segments)
        detection_time = time.time() - start_time
        
        # Measure word assignment performance
        start_time = time.time()
        enhanced_words = assign_words_with_basic_cross_talk(words, segments, cross_talk_segments)
        assignment_time = time.time() - start_time
        
        # Measure turn building performance
        start_time = time.time()
        turns = build_turns(enhanced_words, speaker_label="SpeakerA")
        turn_building_time = time.time() - start_time
        
        total_time = detection_time + assignment_time + turn_building_time
        
        # Verify results are correct
        self.assertIsInstance(cross_talk_segments, list, "Should return list of cross-talk segments")
        self.assertIsInstance(enhanced_words, list, "Should return list of enhanced words")
        self.assertIsInstance(turns, list, "Should return list of turns")
        
        self.assertEqual(len(enhanced_words), len(words), "Should have same number of enhanced words")
        
        # Performance assertions for medium dataset
        self.assertLess(detection_time, 1.0, f"Cross-talk detection should be fast for medium dataset: {detection_time:.4f}s")
        self.assertLess(assignment_time, 1.0, f"Word assignment should be fast for medium dataset: {assignment_time:.4f}s")
        self.assertLess(turn_building_time, 1.0, f"Turn building should be fast for medium dataset: {turn_building_time:.4f}s")
        self.assertLess(total_time, 3.0, f"Total processing should be fast for medium dataset: {total_time:.4f}s")
        
        print(f"Medium dataset ({len(words)} words) performance:")
        print(f"  - Cross-talk detection: {detection_time:.4f}s")
        print(f"  - Word assignment: {assignment_time:.4f}s")
        print(f"  - Turn building: {turn_building_time:.4f}s")
        print(f"  - Total: {total_time:.4f}s")
    
    def test_cross_talk_detection_performance_large(self):
        """Test cross-talk detection performance with large dataset."""
        data = self.large_data
        words = data["words"]
        segments = data["segments"]
        
        # Measure cross-talk detection performance
        start_time = time.time()
        cross_talk_segments = detect_basic_cross_talk(segments)
        detection_time = time.time() - start_time
        
        # Measure word assignment performance
        start_time = time.time()
        enhanced_words = assign_words_with_basic_cross_talk(words, segments, cross_talk_segments)
        assignment_time = time.time() - start_time
        
        # Measure turn building performance
        start_time = time.time()
        turns = build_turns(enhanced_words, speaker_label="SpeakerA")
        turn_building_time = time.time() - start_time
        
        total_time = detection_time + assignment_time + turn_building_time
        
        # Verify results are correct
        self.assertIsInstance(cross_talk_segments, list, "Should return list of cross-talk segments")
        self.assertIsInstance(enhanced_words, list, "Should return list of enhanced words")
        self.assertIsInstance(turns, list, "Should return list of turns")
        
        self.assertEqual(len(enhanced_words), len(words), "Should have same number of enhanced words")
        
        # Performance assertions for large dataset (more lenient)
        self.assertLess(detection_time, 5.0, f"Cross-talk detection should be reasonable for large dataset: {detection_time:.4f}s")
        self.assertLess(assignment_time, 5.0, f"Word assignment should be reasonable for large dataset: {assignment_time:.4f}s")
        self.assertLess(turn_building_time, 5.0, f"Turn building should be reasonable for large dataset: {turn_building_time:.4f}s")
        self.assertLess(total_time, 15.0, f"Total processing should be reasonable for large dataset: {total_time:.4f}s")
        
        print(f"Large dataset ({len(words)} words) performance:")
        print(f"  - Cross-talk detection: {detection_time:.4f}s")
        print(f"  - Word assignment: {assignment_time:.4f}s")
        print(f"  - Turn building: {turn_building_time:.4f}s")
        print(f"  - Total: {total_time:.4f}s")
    
    def test_cross_talk_vs_no_cross_talk_performance(self):
        """Test performance comparison between cross-talk and no cross-talk processing."""
        data = self.medium_data
        words = data["words"]
        segments = data["segments"]
        
        # Measure performance WITH cross-talk detection
        start_time = time.time()
        cross_talk_segments = detect_basic_cross_talk(segments)
        enhanced_words = assign_words_with_basic_cross_talk(words, segments, cross_talk_segments)
        turns_with_cross_talk = build_turns(enhanced_words, speaker_label="SpeakerA")
        with_cross_talk_time = time.time() - start_time
        
        # Measure performance WITHOUT cross-talk detection (simulate by skipping those steps)
        start_time = time.time()
        # Simulate standard processing without cross-talk
        turns_without_cross_talk = build_turns(words, speaker_label="SpeakerA")
        without_cross_talk_time = time.time() - start_time
        
        # Verify both approaches produce valid results
        self.assertIsInstance(turns_with_cross_talk, list, "Should return list of turns with cross-talk")
        self.assertIsInstance(turns_without_cross_talk, list, "Should return list of turns without cross-talk")
        
        # Performance comparison: cross-talk should not be significantly slower
        performance_ratio = with_cross_talk_time / without_cross_talk_time
        
        # Cross-talk processing should not be more than 3x slower than standard processing
        self.assertLess(performance_ratio, 3.0, 
                       f"Cross-talk processing should not be more than 3x slower: ratio={performance_ratio:.2f}")
        
        print(f"Performance comparison:")
        print(f"  - With cross-talk: {with_cross_talk_time:.4f}s")
        print(f"  - Without cross-talk: {without_cross_talk_time:.4f}s")
        print(f"  - Performance ratio: {performance_ratio:.2f}x")
    
    @patch('diarize.Pipeline')
    def test_diarize_mixed_cross_talk_performance(self, mock_pipeline_class):
        """Test diarize_mixed performance with cross-talk detection."""
        # Mock the pipeline and its behavior
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock diarization result
        mock_diar_result = MagicMock()
        mock_diar_result.speaker_diarization = MagicMock()
        
        # Create a mock annotation object with itertracks method
        def mock_itertracks(yield_label=True):
            # Create segments for medium dataset
            for i in range(100):  # 100 segments
                start = i * 0.5
                end = start + 2.0
                speaker = "SpeakerA" if i % 2 == 0 else "SpeakerB"
                segment = MagicMock(start=start, end=end)
                yield (segment, i, speaker)
        
        mock_diar_result.speaker_diarization.itertracks = mock_itertracks
        mock_pipeline.return_value = mock_diar_result
        
        # Mock audio loading functions
        with patch('diarize._load_waveform_mono_32f') as mock_load, \
             patch('diarize._maybe_resample') as mock_resample:
            
            mock_load.return_value = (MagicMock(), 16000)
            mock_resample.return_value = (MagicMock(), 16000)
            
            # Create medium-sized word list
            words = []
            for i in range(1000):
                start = i * 0.1
                end = start + 0.08
                words.append({
                    "text": f"word{i}",
                    "start": start,
                    "end": end,
                    "speaker": None
                })
            
            # Measure performance WITH cross-talk detection
            start_time = time.time()
            turns_with_cross_talk = diarize_mixed(
                str(self.mock_audio_path),
                words,
                detect_cross_talk=True
            )
            with_cross_talk_time = time.time() - start_time
            
            # Measure performance WITHOUT cross-talk detection
            start_time = time.time()
            turns_without_cross_talk = diarize_mixed(
                str(self.mock_audio_path),
                words,
                detect_cross_talk=False
            )
            without_cross_talk_time = time.time() - start_time
            
            # Verify both approaches produce valid results
            self.assertIsInstance(turns_with_cross_talk, list, "Should return list of turns with cross-talk")
            self.assertIsInstance(turns_without_cross_talk, list, "Should return list of turns without cross-talk")
            
            # Performance comparison
            performance_ratio = with_cross_talk_time / without_cross_talk_time
            
            # Cross-talk processing should not be more than 2x slower in diarize_mixed
            self.assertLess(performance_ratio, 2.0, 
                           f"diarize_mixed with cross-talk should not be more than 2x slower: ratio={performance_ratio:.2f}")
            
            print(f"diarize_mixed performance comparison:")
            print(f"  - With cross-talk: {with_cross_talk_time:.4f}s")
            print(f"  - Without cross-talk: {without_cross_talk_time:.4f}s")
            print(f"  - Performance ratio: {performance_ratio:.2f}x")
    
    def test_cross_talk_memory_performance(self):
        """Test that cross-talk detection doesn't use excessive memory."""
        import gc
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Measure memory before processing
        gc.collect()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large dataset
        data = self.large_data
        words = data["words"]
        segments = data["segments"]
        
        # Run cross-talk detection
        cross_talk_segments = detect_basic_cross_talk(segments)
        enhanced_words = assign_words_with_basic_cross_talk(words, segments, cross_talk_segments)
        turns = build_turns(enhanced_words, speaker_label="SpeakerA")
        
        # Force garbage collection
        del cross_talk_segments
        del enhanced_words
        del turns
        gc.collect()
        
        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(memory_increase, 100, 
                       f"Memory increase should be reasonable: {memory_increase:.2f}MB")
        
        print(f"Memory performance:")
        print(f"  - Memory before: {memory_before:.2f}MB")
        print(f"  - Memory after: {memory_after:.2f}MB")
        print(f"  - Memory increase: {memory_increase:.2f}MB")
    
    def test_cross_talk_scalability(self):
        """Test that cross-talk detection scales linearly with input size."""
        # Test with different dataset sizes
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            # Create test data
            words = []
            segments = []
            
            for i in range(size):
                start_time = i * 0.4
                end_time = start_time + 0.3
                
                words.append({
                    "text": f"word{i}",
                    "start": start_time,
                    "end": end_time,
                    "speaker": None
                })
                
                if i % 20 == 0:
                    speaker = "SpeakerA" if (i // 20) % 2 == 0 else "SpeakerB"
                    segments.append({
                        "start": start_time - 0.1,
                        "end": end_time + 3.0,
                        "label": speaker
                    })
            
            # Measure performance
            start_time = time.time()
            cross_talk_segments = detect_basic_cross_talk(segments)
            enhanced_words = assign_words_with_basic_cross_talk(words, segments, cross_talk_segments)
            turns = build_turns(enhanced_words, speaker_label="SpeakerA")
            processing_time = time.time() - start_time
            
            times.append(processing_time)
            
            print(f"Size {size}: {processing_time:.4f}s")
        
        # Check that performance scales roughly linearly
        # (i.e., doubling input size shouldn't more than quadruple processing time)
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            
            # Allow for some overhead, but it shouldn't be quadratic
            max_acceptable_ratio = size_ratio * 2  # Allow 2x the linear ratio
            
            self.assertLess(time_ratio, max_acceptable_ratio,
                           f"Performance should scale roughly linearly: size {sizes[i-1]}->{sizes[i]}, "
                           f"time ratio {time_ratio:.2f} (max acceptable: {max_acceptable_ratio:.2f})")
        
        print(f"Scalability test passed for sizes: {sizes}")


def run_tests():
    """Run all tests and return the result."""
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceCrossTalk)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Testing cross-talk performance...")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All performance tests passed! Cross-talk detection has acceptable performance.")
    else:
        print("❌ Some performance tests failed. Please review the implementation.")
        sys.exit(1)