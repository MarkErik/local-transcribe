#!/usr/bin/env python3
"""
Base classes and utilities for pipeline stages.

This module provides the abstract PipelineStage class and related utilities
that all concrete stage implementations inherit from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Callable, Union
from enum import Enum

from local_transcribe.framework.pipeline_context import PipelineContext


# Type for progress callback: (event_type: str, data: dict) -> None
ProgressCallback = Callable[[str, Dict[str, Any]], None]


class StageStatus(Enum):
    """Status of a pipeline stage execution."""
    NOT_STARTED = "not_started"
    SKIPPED = "skipped"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StageResult:
    """Result from executing a pipeline stage."""
    stage_name: str
    status: StageStatus
    message: str = ""
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if stage completed successfully."""
        return self.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)


class StageError(Exception):
    """Exception raised when a pipeline stage fails."""
    
    def __init__(self, stage_name: str, message: str, cause: Optional[Exception] = None):
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"[{stage_name}] {message}")


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage represents a discrete step in the transcription pipeline.
    Stages can be executed independently or chained together.
    
    Subclasses must implement:
    - name: Unique identifier for the stage
    - description: Human-readable description
    - required_inputs: List of context attributes required
    - produces_outputs: List of context attributes produced
    - execute: The actual stage logic
    
    Stages may optionally override:
    - can_execute: Custom validation logic
    - is_optional: Whether the stage can be skipped
    - applicable_modes: Which processing modes this stage applies to
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier for this stage."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this stage."""
        pass
    
    @property
    @abstractmethod
    def required_inputs(self) -> List[str]:
        """Return list of context attributes required for this stage."""
        pass
    
    @property
    @abstractmethod
    def produces_outputs(self) -> List[str]:
        """Return list of context attributes this stage produces."""
        pass
    
    @property
    def is_optional(self) -> bool:
        """
        Return True if this stage is optional and can be skipped.
        
        Optional stages (like de-identification, transcript cleanup) won't
        cause pipeline failure if they can't execute due to missing config.
        """
        return False
    
    @property
    def applicable_modes(self) -> List[str]:
        """
        Return list of processing modes this stage applies to.
        
        Empty list means the stage applies to all modes.
        """
        return []  # Empty = applies to all modes
    
    def applies_to_mode(self, mode: str) -> bool:
        """Check if this stage applies to the given processing mode."""
        if not self.applicable_modes:
            return True
        return mode in self.applicable_modes
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        """
        Check if this stage can execute given the current context.
        
        Args:
            context: The pipeline context
            
        Returns:
            Tuple of (can_execute, reason_if_not)
        """
        # Check if stage applies to current mode
        if not self.applies_to_mode(context.mode):
            return False, f"Stage does not apply to mode: {context.mode}"
        
        # Check required inputs
        for required in self.required_inputs:
            if not hasattr(context, required) or getattr(context, required) is None:
                return False, f"Missing required input: {required}"
        
        return True, ""
    
    @abstractmethod
    def execute(
        self,
        context: PipelineContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineContext:
        """
        Execute this pipeline stage.
        
        Args:
            context: The pipeline context with all required inputs
            progress_callback: Optional callback for progress events.
                              Called as: progress_callback(event_type, data_dict)
            
        Returns:
            Updated context with this stage's outputs
            
        Raises:
            StageError: If execution fails
        """
        pass
    
    def execute_safe(
        self,
        context: PipelineContext,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> tuple[PipelineContext, StageResult]:
        """
        Execute this stage with error handling and result tracking.
        
        This is the preferred method for the executor to call.
        
        Args:
            context: The pipeline context
            progress_callback: Optional callback for progress events
            
        Returns:
            Tuple of (updated_context, result)
        """
        from local_transcribe.lib.program_logger import log_status, log_completion
        
        # Check if stage should be skipped
        if context.should_skip_stage(self.name):
            return context, StageResult(
                stage_name=self.name,
                status=StageStatus.SKIPPED,
                message="Skipped (before re-entry point)"
            )
        
        # Check if stage can execute
        can_run, reason = self.can_execute(context)
        if not can_run:
            if self.is_optional:
                return context, StageResult(
                    stage_name=self.name,
                    status=StageStatus.SKIPPED,
                    message=f"Skipped (optional): {reason}"
                )
            else:
                return context, StageResult(
                    stage_name=self.name,
                    status=StageStatus.FAILED,
                    message=reason,
                    error=StageError(self.name, reason)
                )
        
        # Execute the stage
        log_status(f"[{self.name}] {self.description}")
        
        try:
            context = self.execute(context, progress_callback=progress_callback)
            context.mark_stage_complete(self.name)
            
            return context, StageResult(
                stage_name=self.name,
                status=StageStatus.COMPLETED,
                message="Success"
            )
            
        except StageError as e:
            return context, StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                message=str(e),
                error=e
            )
        except Exception as e:
            error = StageError(self.name, str(e), cause=e)
            return context, StageResult(
                stage_name=self.name,
                status=StageStatus.FAILED,
                message=str(e),
                error=error
            )
    
    def execute_dry_run(self, context: PipelineContext) -> StageResult:
        """
        Perform a dry run of this stage (validation only, no execution).
        
        Args:
            context: The pipeline context
            
        Returns:
            StageResult with validation status
        """
        if context.should_skip_stage(self.name):
            status = StageStatus.SKIPPED
            message = "Would skip (before re-entry point)"
        else:
            can_run, reason = self.can_execute(context)
            if can_run:
                status = StageStatus.NOT_STARTED
                message = "Ready to execute"
            elif self.is_optional:
                status = StageStatus.SKIPPED
                message = f"Would skip (optional): {reason}"
            else:
                status = StageStatus.FAILED
                message = f"Cannot execute: {reason}"
        
        symbol = {
            StageStatus.NOT_STARTED: "○",
            StageStatus.SKIPPED: "◌",
            StageStatus.FAILED: "✗",
        }.get(status, "?")
        
        print(f"  {symbol} [{self.name}] {message}")
        
        return StageResult(
            stage_name=self.name,
            status=status,
            message=message
        )
