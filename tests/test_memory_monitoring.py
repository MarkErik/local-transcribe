#!/usr/bin/env python3
"""
Test script to verify memory monitoring and duration tracking fixes.
"""

import sys
import os
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from progress import get_progress_tracker


def test_memory_monitoring():
    """Test that memory monitoring actually tracks memory usage."""
    print("Testing memory monitoring...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    # Create a task that will run long enough to collect memory data
    task = tracker.add_task("Memory test task", total=100, stage="memory_test")
    
    # Simulate some work that might use memory
    data = []
    for i in range(10):
        # Allocate some memory
        data.append([0] * 100000)  # Allocate ~400KB per iteration
        tracker.update(task, advance=10, description=f"Memory test step {i}")
        time.sleep(0.2)  # Give memory monitoring time to run
    
    tracker.complete_task(task, stage="memory_test")
    
    # Check metrics
    metrics = tracker.get_metrics("memory_test")
    if metrics:
        print(f"  Duration: {metrics.duration:.2f}s")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Current memory: {metrics.current_memory_mb:.1f}MB")
        
        if metrics.peak_memory_mb and metrics.peak_memory_mb > 0:
            print("✓ Memory monitoring is working!")
        else:
            print("❌ Memory monitoring still showing 0 values")
    else:
        print("❌ No metrics found")
    
    tracker.stop()


def test_duration_tracking():
    """Test that duration tracking works for both quick and slow tasks."""
    print("\nTesting duration tracking...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    # Test quick task
    quick_task = tracker.add_task("Quick task", total=5, stage="quick_test")
    for i in range(5):
        tracker.update(quick_task, advance=1)
        time.sleep(0.001)  # Very quick
    tracker.complete_task(quick_task, stage="quick_test")
    
    # Test slower task
    slow_task = tracker.add_task("Slower task", total=5, stage="slow_test")
    for i in range(5):
        tracker.update(slow_task, advance=1)
        time.sleep(0.1)  # Slower
    tracker.complete_task(slow_task, stage="slow_test")
    
    # Check metrics
    quick_metrics = tracker.get_metrics("quick_test")
    slow_metrics = tracker.get_metrics("slow_test")
    
    print(f"  Quick task duration: {quick_metrics.duration:.4f}s")
    print(f"  Slow task duration: {slow_metrics.duration:.2f}s")
    
    if quick_metrics.duration > 0 and slow_metrics.duration > 0:
        print("✓ Duration tracking is working!")
    else:
        print("❌ Duration tracking showing 0 values")
    
    tracker.print_summary()
    tracker.stop()


def test_concurrent_tasks():
    """Test memory monitoring with concurrent tasks."""
    print("\nTesting concurrent tasks...")
    
    tracker = get_progress_tracker()
    tracker.start()
    
    # Create multiple tasks running "concurrently"
    task1 = tracker.add_task("Concurrent task 1", total=10, stage="concurrent_1")
    task2 = tracker.add_task("Concurrent task 2", total=10, stage="concurrent_2")
    
    for i in range(10):
        tracker.update(task1, advance=1, description=f"Task 1 - Step {i}")
        tracker.update(task2, advance=1, description=f"Task 2 - Step {i}")
        time.sleep(0.1)
    
    tracker.complete_task(task1, stage="concurrent_1")
    tracker.complete_task(task2, stage="concurrent_2")
    
    # Check metrics for both tasks
    metrics1 = tracker.get_metrics("concurrent_1")
    metrics2 = tracker.get_metrics("concurrent_2")
    
    print(f"  Task 1 - Duration: {metrics1.duration:.2f}s, Memory: {metrics1.peak_memory_mb:.1f}MB")
    print(f"  Task 2 - Duration: {metrics2.duration:.2f}s, Memory: {metrics2.peak_memory_mb:.1f}MB")
    
    if metrics1.peak_memory_mb and metrics2.peak_memory_mb:
        print("✓ Concurrent task memory monitoring working!")
    else:
        print("❌ Memory monitoring not working for concurrent tasks")
    
    tracker.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY MONITORING AND DURATION TRACKING TESTS")
    print("=" * 60)
    
    try:
        test_memory_monitoring()
        test_duration_tracking()
        test_concurrent_tasks()
        
        print("\n" + "=" * 60)
        print("TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)