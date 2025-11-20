#!/usr/bin/env python3
from __future__ import annotations
import time
import psutil
import threading
from typing import Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.console import Console


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    start_time: float
    end_time: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    current_memory_mb: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class ProgressTracker:
    """Enhanced progress tracking with performance monitoring."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        )
        self.metrics: dict[str, PerformanceMetrics] = {}
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        """Start the progress display."""
        self.progress.start()
        
    def stop(self) -> None:
        """Stop the progress display and monitoring."""
        self.progress.stop()
        self._stop_monitoring()
        
    def add_task(
        self,
        description: str,
        total: Optional[int] = None,
        stage: str = "processing"
    ) -> TaskID:
        """Add a new progress task."""
        task_id = self.progress.add_task(description, total=total)
        self.metrics[stage] = PerformanceMetrics(start_time=time.time())
        # Start memory monitoring for all tasks
        self._start_monitoring()
        return task_id
        
    def update(
        self, 
        task_id: TaskID, 
        advance: int = 1,
        description: Optional[str] = None,
        stage: Optional[str] = None
    ) -> None:
        """Update progress for a task."""
        self.progress.update(task_id, advance=advance, description=description)
        
    def complete_task(self, task_id: TaskID, stage: Optional[str] = None) -> None:
        """Mark a task as complete and hide it from the display."""
        # Mark the task as completed
        self.progress.update(task_id, completed=self.progress.tasks[task_id].total)
        # Stop the task to prevent it from being redrawn
        self.progress.stop_task(task_id)
        # Hide the task from the display by setting visible=False
        self.progress.update(task_id, visible=False)
        # Refresh the display to immediately reflect the changes
        self.progress.refresh()
        if stage and stage in self.metrics:
            self.metrics[stage].end_time = time.time()
            # Update memory one final time before stopping
            try:
                import psutil
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.metrics[stage].current_memory_mb = current_memory
                if self.metrics[stage].peak_memory_mb is None or current_memory > self.metrics[stage].peak_memory_mb:
                    self.metrics[stage].peak_memory_mb = current_memory
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
    @contextmanager
    def task_context(
        self,
        description: str,
        total: Optional[int] = None,
        stage: str = "processing"
    ):
        """Context manager for automatic task lifecycle management."""
        task_id = self.add_task(description, total=total, stage=stage)
        # Memory monitoring already started in add_task
        try:
            yield task_id
        finally:
            self.complete_task(task_id, stage)
            # Don't stop monitoring here as other tasks might be running
            
    def _start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self._monitor_thread.start()
            
    def _stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=0.1)
            
    def _monitor_memory(self) -> None:
        """Background thread to monitor memory usage."""
        process = psutil.Process()
        peak_memory = 0.0
        
        while self._monitoring_active:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                peak_memory = max(peak_memory, current_memory)
                
                # Update metrics for all active stages
                for stage, metrics in self.metrics.items():
                    if metrics.end_time is None:  # Still active
                        metrics.current_memory_mb = current_memory
                        metrics.peak_memory_mb = peak_memory
                        
                time.sleep(0.5)  # Monitor every 500ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
                
    def get_metrics(self, stage: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific stage."""
        return self.metrics.get(stage)
        
    def print_summary(self) -> None:
        """Print a summary of all performance metrics."""
        self.console.print("\n[bold]Performance Summary:[/bold]")
        
        for stage, metrics in self.metrics.items():
            duration = metrics.duration
            peak_mem = metrics.peak_memory_mb or 0
            
            # Format duration to show meaningful precision
            if duration < 0.01:
                duration_str = "< 0.01s"
            else:
                duration_str = f"{duration:.2f}s"
            
            # Format memory to show meaningful values
            if peak_mem < 0.1:
                mem_str = "< 0.1MB"
            else:
                mem_str = f"{peak_mem:.1f}MB"
            
            self.console.print(
                f"  {stage}: {duration_str}, "
                f"Peak Memory: {mem_str}"
            )


class ProgressCallback:
    """Callback wrapper for integrating with external libraries."""
    
    def __init__(self, tracker: ProgressTracker, task_id: TaskID):
        self.tracker = tracker
        self.task_id = task_id
        
    def __call__(self, progress_data: Any) -> None:
        """Handle progress callback from external libraries."""
        # Handle different callback formats
        if isinstance(progress_data, (int, float)):
            # Simple numeric progress
            self.tracker.update(self.task_id, advance=int(progress_data))
        elif isinstance(progress_data, dict):
            # Dictionary with progress info
            if "progress" in progress_data:
                self.tracker.update(self.task_id, advance=int(progress_data["progress"]))
            if "description" in progress_data:
                self.tracker.update(self.task_id, description=progress_data["description"])


# Global progress tracker instance
_global_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get or create the global progress tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker



