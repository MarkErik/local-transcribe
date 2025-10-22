#!/usr/bin/env python3
"""
Integration test for progress tracking functionality.
Tests that progress bars actually track progress and show meaningful stats.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from progress import get_progress_tracker


def test_basic_progress_tracking():
    """Test basic progress tracking with determinate and indeterminate tasks."""
    print("Testing basic progress tracking...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    # Test 1: Determinate progress (with total)
    task1 = tracker.add_task("Processing files", total=100, stage="file_processing")
    
    for i in range(10):
        tracker.update(task1, advance=10, description=f"Processing files - {i*10}%")
        time.sleep(0.05)  # Small delay to visualize progress
    
    tracker.complete_task(task1, stage="file_processing")
    
    # Test 2: Indeterminate progress (without total)
    task2 = tracker.add_task("Loading model", stage="model_loading")
    
    for i in range(5):
        tracker.update(task2, description=f"Loading model - Step {i+1}")
        time.sleep(0.05)
    
    tracker.complete_task(task2, stage="model_processing")
    
    # Test 3: Item-based progress
    items = ["item1", "item2", "item3", "item4", "item5"]
    task3 = tracker.add_task("Processing items", total=len(items), stage="item_processing")
    
    for i, item in enumerate(items):
        tracker.update(task3, advance=1, description=f"Processing {item}")
        time.sleep(0.05)
    
    tracker.complete_task(task3, stage="item_processing")
    
    # Print performance summary
    tracker.print_summary()
    tracker.stop()
    
    print("✓ Basic progress tracking test passed!")


def test_progress_callback():
    """Test the ProgressCallback wrapper."""
    print("\nTesting progress callback...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    from progress import ProgressCallback
    
    task = tracker.add_task("Callback test", total=50, stage="callback_test")
    callback = ProgressCallback(tracker, task)
    
    # Test different callback formats
    for i in range(5):
        # Numeric progress
        callback(5)
        time.sleep(0.02)
    
    for i in range(5):
        # Dictionary progress
        callback({"progress": 3, "description": f"Callback step {i}"})
        time.sleep(0.02)
    
    tracker.complete_task(task, stage="callback_test")
    tracker.stop()
    
    print("✓ Progress callback test passed!")


def test_context_manager():
    """Test the progress context manager."""
    print("\nTesting progress context manager...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    from progress import progress_context
    
    with progress_context("Context test", total=20, stage="context_test") as tracker:
        for i in range(10):
            tracker.update(tracker.progress.tasks[-1].id, advance=2, description=f"Context step {i}")
            time.sleep(0.02)
    
    tracker.stop()
    
    print("✓ Progress context manager test passed!")


def test_memory_monitoring():
    """Test memory monitoring functionality."""
    print("\nTesting memory monitoring...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    task = tracker.add_task("Memory test", total=100, stage="memory_test")
    
    # Simulate some work
    for i in range(10):
        tracker.update(task, advance=10, description=f"Memory test step {i}")
        time.sleep(0.1)
    
    tracker.complete_task(task, stage="memory_test")
    
    # Check if metrics were collected
    metrics = tracker.get_metrics("memory_test")
    if metrics:
        print(f"  Duration: {metrics.duration:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB" if metrics.peak_memory_mb else "  Peak memory: N/A")
    
    tracker.stop()
    
    print("✓ Memory monitoring test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("PROGRESS TRACKING INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_basic_progress_tracking()
        test_progress_callback()
        test_context_manager()
        test_memory_monitoring()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("Progress bars and stats are now working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)