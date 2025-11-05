# src/progress.py
from __future__ import annotations
import time
import logging
from typing import Optional, Dict
from contextlib import contextmanager
from dataclasses import dataclass
from rich.progress import Progress, TaskID, BarColumn, TextColumn, SpinnerColumn
from rich.console import Console

logger = logging.getLogger(__name__)


@dataclass
class TaskTimer:
    """Simple timer for tracking task elapsed time."""
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def elapsed_str(self) -> str:
        """Get formatted elapsed time string."""
        elapsed = self.elapsed
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes}:{seconds:02d}"
    
    @property
    def elapsed_str_detailed(self) -> str:
        """Get detailed formatted elapsed time string with milliseconds for active tasks."""
        if self.end_time is not None:
            # For completed tasks, use simple format
            return self.elapsed_str
        else:
            # For active tasks, include milliseconds for smoother updates
            elapsed = self.elapsed
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            milliseconds = int((elapsed % 1) * 100)
            return f"{minutes}:{seconds:02d}.{milliseconds:02d}"


class ProgressTracker:
    """Simplified progress tracking with elapsed time display."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[blue]Time:[/blue] {task.fields[elapsed]}"),
            console=self.console,
            refresh_per_second=10  # Ensure more frequent updates for smooth time display
        )
        self.task_timers: Dict[TaskID, TaskTimer] = {}
        self._started = False
        
    def start(self) -> None:
        """Start the progress display."""
        if not self._started:
            self.progress.start()
            self._started = True
        
    def stop(self) -> None:
        """Stop the progress display."""
        if self._started:
            # Complete all active tasks before stopping so final state renders
            self._complete_all_tasks()
            self.progress.stop()
            self._started = False
        
    def add_task(
        self,
        description: str,
        total: Optional[int] = None
    ) -> TaskID:
        """Add a new progress task."""
        task_id = self.progress.add_task(
            description, 
            total=total,
            elapsed="0:00"  # Initial elapsed time
        )
        self.task_timers[task_id] = TaskTimer(start_time=time.time())
        return task_id
        
    def update(
        self,
        task_id: TaskID,
        advance: int = 1,
        description: Optional[str] = None
    ) -> None:
        """Update progress for a task."""
        # Validate task exists and is active (tasks is a list indexed by TaskID)
        if not (0 <= task_id < len(self.progress.tasks)):
            logger.debug(f"Task {task_id} not found in progress tracker")
            return
            
        task = self.progress.tasks[task_id]
        if task.finished:
            logger.debug(f"Task {task.description} is already finished")
            return
            
        # Update elapsed time
        if task_id in self.task_timers:
            try:
                timer = self.task_timers[task_id]
                # Use detailed time for active tasks, simple format for completed
                elapsed_str = timer.elapsed_str_detailed if timer.end_time is None else timer.elapsed_str
                update_kwargs = {"advance": advance, "elapsed": elapsed_str}
                
                # Update description if provided
                if description:
                    update_kwargs["description"] = description
                    
                self.progress.update(task_id, **update_kwargs)
            except Exception as e:
                if hasattr(self, 'console') and self.console:
                    self.console.print(f"[red]Error updating task {task.description}: {e}[/red]")
        
    def set_total(
        self,
        task_id: TaskID,
        total: int,
        description: Optional[str] = None
    ) -> None:
        """
        Set or update the total for a task (useful when total is not known at add_task time).
        Also updates elapsed time and optionally the description.
        """
        # Validate task exists
        if not (0 <= task_id < len(self.progress.tasks)):
            logger.debug(f"Task {task_id} not found in progress tracker")
            return
        
        # Build update kwargs
        update_kwargs = {"total": total}
        
        # Update elapsed time if we have a timer
        if task_id in self.task_timers:
            timer = self.task_timers[task_id]
            elapsed_str = timer.elapsed_str_detailed if timer.end_time is None else timer.elapsed_str
            update_kwargs["elapsed"] = elapsed_str
        
        # Update description if provided
        if description:
            update_kwargs["description"] = description
        
        try:
            self.progress.update(task_id, **update_kwargs)
        except Exception as e:
            if hasattr(self, 'console') and self.console:
                task = self.progress.tasks[task_id]
                self.console.print(f"[red]Error setting total for task {task.description}: {e}[/red]")
    
    def complete_task(self, task_id: TaskID) -> None:
        """Mark a task as complete."""
        # Validate task exists and is not already finished (tasks is a list indexed by TaskID)
        if not (0 <= task_id < len(self.progress.tasks)):
            logger.debug(f"Task {task_id} not found in progress tracker")
            return
            
        task = self.progress.tasks[task_id]
        if task.finished:
            logger.debug(f"Task {task.description} is already finished")
            return
            
        try:
            # Mark task as complete
            if task.total is not None:
                self.progress.update(task_id, completed=task.total)
                
            # Record end time and update elapsed display
            if task_id in self.task_timers:
                self.task_timers[task_id].end_time = time.time()
                elapsed_str = self.task_timers[task_id].elapsed_str
                self.progress.update(task_id, elapsed=elapsed_str)
        except Exception as e:
            if hasattr(self, 'console') and self.console:
                self.console.print(f"[red]Error completing task {task.description}: {e}[/red]")
            
    @contextmanager
    def task_context(
        self,
        description: str,
        total: Optional[int] = None
    ):
        """Context manager for automatic task lifecycle management."""
        task_id = self.add_task(description, total=total)
        try:
            yield task_id
        finally:
            self.complete_task(task_id)
            
    def _complete_all_tasks(self) -> None:
        """Complete all active tasks."""
        current_time = time.time()
        
        for task_id, timer in self.task_timers.items():
            # Validate task exists in the list
            if not (0 <= task_id < len(self.progress.tasks)):
                continue
                
            if timer.end_time is None:
                # Record end time first
                timer.end_time = current_time
                
                # Mark task as complete
                task = self.progress.tasks[task_id]
                if task.total is not None:
                    self.progress.update(task_id, completed=task.total)
                
                # Update elapsed display separately using fields parameter
                self.progress.update(task_id, elapsed=timer.elapsed_str)
                
    def get_task_elapsed(self, task_id: TaskID) -> Optional[float]:
        """Get elapsed time for a specific task."""
        if task_id not in self.task_timers:
            return None
        return self.task_timers[task_id].elapsed

    def print_summary(self) -> None:
        """Print a summary of completed tasks, their durations, and overall performance metrics."""
        if not self.task_timers:
            self.console.print("[yellow]No tasks completed yet.[/yellow]")
            return
            
        self.console.print("\n[bold blue]Progress Summary[/bold blue]")
        self.console.print("=" * 50)
        
        total_elapsed = 0.0
        completed_tasks = 0
        
        for task_id, timer in self.task_timers.items():
            # Validate task exists in the list
            if not (0 <= task_id < len(self.progress.tasks)):
                continue
                
            task = self.progress.tasks[task_id]
            status = "Completed" if timer.end_time is not None else "In Progress"
            
            self.console.print(f"[cyan]Task:[/cyan] {task.description}")
            self.console.print(f"  [white]Status:[/white] {status}")
            self.console.print(f"  [white]Duration:[/white] {timer.elapsed_str}")
            
            if timer.end_time is not None:
                total_elapsed += timer.elapsed
                completed_tasks += 1
            
            if task.total is not None:
                progress_pct = (task.completed / task.total) * 100 if task.total > 0 else 0
                self.console.print(f"  [white]Progress:[/white] {task.completed}/{task.total} ({progress_pct:.1f}%)")
            
            self.console.print()
        
        if completed_tasks > 0:
            avg_elapsed = total_elapsed / completed_tasks
            minutes = int(total_elapsed // 60)
            seconds = int(total_elapsed % 60)
            
            self.console.print(f"[bold green]Summary:[/bold green]")
            self.console.print(f"  [white]Total Tasks:[/white] {len(self.task_timers)}")
            self.console.print(f"  [white]Completed Tasks:[/white] {completed_tasks}")
            self.console.print(f"  [white]Total Time:[/white] {minutes}:{seconds:02d}")
            self.console.print(f"  [white]Average Time per Task:[/white] {avg_elapsed:.1f} seconds")
        else:
            self.console.print("[yellow]No completed tasks to summarize.[/yellow]")
        
        self.console.print("=" * 50)


# Global progress tracker instance
_global_tracker: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Get or create the global progress tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker


def progress_task(
    description: str, 
    total: Optional[int] = None
):
    """Decorator for adding progress tracking to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = get_progress_tracker()
            with tracker.task_context(description, total=total):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def progress_context(
    description: str, 
    total: Optional[int] = None
):
    """Context manager for progress tracking."""
    tracker = get_progress_tracker()
    with tracker.task_context(description, total=total):
        yield tracker