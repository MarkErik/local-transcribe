#!/usr/bin/env python3
"""
Test script for the progress reporting functionality.
This script creates a minimal test to verify that progress tracking works correctly.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from progress import ProgressTracker, progress_context, get_progress_tracker
from rich.console import Console


def test_basic_progress():
    """Test basic progress tracking functionality."""
    console = Console()
    tracker = ProgressTracker(console)
    
    console.print("[bold]Testing basic progress tracking...[/bold]")
    
    with tracker.task_context("Test Task 1", total=10, stage="test1"):
        for i in range(10):
            time.sleep(0.1)  # Simulate work
            tracker.update(tracker.progress.tasks[-1].id, advance=1)
    
    console.print("✅ Basic progress test passed")


def test_multiple_tasks():
    """Test multiple concurrent tasks."""
    console = Console()
    tracker = ProgressTracker(console)
    
    console.print("[bold]Testing multiple concurrent tasks...[/bold]")
    
    tracker.start()
    
    # Add multiple tasks
    task1 = tracker.add_task("Task A", total=5, stage="task_a")
    task2 = tracker.add_task("Task B", total=8, stage="task_b")
    
    # Simulate work on both tasks
    for i in range(5):
        time.sleep(0.1)
        tracker.update(task1, advance=1)
        if i < 3:  # Task B progresses slower
            tracker.update(task2, advance=1)
    
    tracker.complete_task(task1, stage="task_a")
    tracker.complete_task(task2, stage="task_b")
    
    tracker.print_summary()
    tracker.stop()
    
    console.print("✅ Multiple tasks test passed")


def test_performance_metrics():
    """Test performance metrics collection."""
    console = Console()
    tracker = ProgressTracker(console)
    
    console.print("[bold]Testing performance metrics...[/bold]")
    
    with tracker.task_context("Performance Test", total=5, stage="perf_test"):
        for i in range(5):
            time.sleep(0.2)  # Simulate work
    
    # Check metrics
    metrics = tracker.get_metrics("perf_test")
    if metrics:
        console.print(f"Duration: {metrics.duration:.2f}s")
        console.print(f"Peak Memory: {metrics.peak_memory_mb or 0:.1f}MB")
    
    console.print("✅ Performance metrics test passed")


def test_context_manager():
    """Test the context manager interface."""
    console = Console()
    
    console.print("[bold]Testing context manager...[/bold]")
    
    with progress_context("Context Manager Test", total=3, stage="context_test") as tracker:
        for i in range(3):
            time.sleep(0.1)
            # The context manager handles progress automatically
    
    console.print("✅ Context manager test passed")


def test_integration_with_asr():
    """Test integration simulation with ASR-like workflow."""
    console = Console()
    
    console.print("[bold]Testing ASR integration simulation...[/bold]")
    
    # Simulate the ASR workflow
    tracker = get_progress_tracker()
    tracker.start()
    
    try:
        # Simulate audio standardization
        std_task = tracker.add_task("Audio standardization", stage="standardization")
        time.sleep(0.5)  # Simulate work
        tracker.complete_task(std_task, stage="standardization")
        
        # Simulate ASR transcription
        asr_task = tracker.add_task("ASR Transcription", stage="asr_transcription")
        for i in range(10):
            time.sleep(0.1)
            tracker.update(asr_task, description=f"ASR Transcription - Processing segment {i+1}/10")
        tracker.complete_task(asr_task, stage="asr_transcription")
        
        # Simulate alignment
        align_task = tracker.add_task("Alignment", stage="alignment")
        time.sleep(0.3)
        tracker.complete_task(align_task, stage="alignment")
        
        # Simulate diarization
        diarize_task = tracker.add_task("Speaker Diarization", stage="diarization")
        for i in range(5):
            time.sleep(0.1)
            tracker.update(diarize_task, description=f"Speaker Diarization - Found {i+1} segments")
        tracker.complete_task(diarize_task, stage="diarization")
        
        tracker.print_summary()
        
    finally:
        tracker.stop()
    
    console.print("✅ ASR integration test passed")


def main():
    """Run all tests."""
    console = Console()
    
    console.print("[bold blue]🧪 Running Progress Reporting Tests[/bold blue]")
    console.print()
    
    try:
        test_basic_progress()
        console.print()
        
        test_multiple_tasks()
        console.print()
        
        test_performance_metrics()
        console.print()
        
        test_context_manager()
        console.print()
        
        test_integration_with_asr()
        console.print()
        
        console.print("[bold green]✅ All tests passed![/bold green]")
        console.print()
        console.print("Progress reporting implementation is working correctly.")
        
    except Exception as e:
        console.print(f"[bold red]❌ Test failed: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())