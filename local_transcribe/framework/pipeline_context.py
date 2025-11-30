#!/usr/bin/env python3
"""
Pipeline context object for managing state across pipeline stages.

This module provides the PipelineContext dataclass that flows through
pipeline stages, carrying configuration, data, and state information.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import argparse


@dataclass
class PipelineContext:
    """
    Context object that flows through pipeline stages.
    
    Carries all the state needed for pipeline execution, including:
    - Input configuration (args, api, paths)
    - Data flowing through stages (segments, transcripts)
    - Execution state (completed stages, re-entry info)
    """
    
    # Core configuration
    args: argparse.Namespace
    api: Dict[str, Any]
    root: Path
    
    # Paths configuration
    paths: Dict[str, Path] = field(default_factory=dict)
    
    # Processing mode
    mode: str = "combined_audio"
    
    # Speaker/audio file mapping
    speaker_files: Dict[str, str] = field(default_factory=dict)
    
    # Providers (set during setup)
    transcriber_provider: Optional[Any] = None
    aligner_provider: Optional[Any] = None
    diarization_provider: Optional[Any] = None
    transcript_cleanup_provider: Optional[Any] = None
    
    # Data flowing through pipeline
    standardized_audio: Optional[Path] = None
    word_segments: Optional[List[Any]] = None  # List[WordSegment]
    diarized_segments: Optional[List[Any]] = None  # List[WordSegment] with speakers
    transcript: Optional[Any] = None  # TranscriptFlow
    
    # Execution state
    completed_stages: List[str] = field(default_factory=list)
    start_from_stage: Optional[str] = None
    dry_run: bool = False
    
    # Re-entry configuration
    input_checkpoint_path: Optional[Path] = None
    checkpoint_metadata: Optional[Dict[str, Any]] = None
    
    # Models directory
    models_dir: Optional[Path] = None
    
    def mark_stage_complete(self, stage_name: str) -> None:
        """Mark a stage as completed."""
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
    
    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if a stage has been completed."""
        return stage_name in self.completed_stages
    
    def should_skip_stage(self, stage_name: str) -> bool:
        """
        Determine if a stage should be skipped.
        
        A stage is skipped if:
        - It's before the start_from_stage (for re-entry)
        - It's in the skip_stages list
        """
        # If we have a start_from_stage, skip all stages before it
        if self.start_from_stage:
            stage_order = get_stage_order()
            if stage_name in stage_order and self.start_from_stage in stage_order:
                stage_idx = stage_order.index(stage_name)
                start_idx = stage_order.index(self.start_from_stage)
                if stage_idx < start_idx:
                    return True
        
        # Check skip_stages in args
        skip_stages = getattr(self.args, 'skip_stages', None)
        if skip_stages and stage_name in skip_stages:
            return True
        
        return False
    
    def get_output_dir(self) -> Path:
        """Get the main output directory."""
        return self.paths.get("root", self.root)
    
    def get_intermediate_dir(self) -> Optional[Path]:
        """Get the intermediate outputs directory."""
        return self.paths.get("intermediate")


def get_stage_order() -> List[str]:
    """
    Return the canonical order of pipeline stages.
    
    This defines the execution order and is used for:
    - Determining which stages to skip on re-entry
    - Validating stage names
    - Displaying available stages
    """
    return [
        "audio_standardization",
        "transcription_alignment",
        "de_identification",
        "diarization",
        "turn_building",
        "speaker_naming",
        "output_generation",
        "transcript_preparation",
        "transcript_cleanup",
    ]


def get_stage_descriptions() -> Dict[str, str]:
    """Return human-readable descriptions for each stage."""
    return {
        "audio_standardization": "Convert audio to standardized format (16kHz mono WAV)",
        "transcription_alignment": "Transcribe audio and align words with timestamps",
        "de_identification": "Remove personal names and identifiers (optional)",
        "diarization": "Assign speaker labels to word segments",
        "turn_building": "Group word segments into conversational turns",
        "speaker_naming": "Map speaker IDs to human-readable names",
        "output_generation": "Generate output files in selected formats",
        "transcript_preparation": "Prepare transcript for LLM processing",
        "transcript_cleanup": "LLM-based transcript cleanup (optional)",
    }


def validate_stage_name(stage_name: str) -> bool:
    """Check if a stage name is valid."""
    return stage_name in get_stage_order()


def get_stages_from(start_stage: str) -> List[str]:
    """Get all stages from a given starting point (inclusive)."""
    stages = get_stage_order()
    if start_stage not in stages:
        raise ValueError(f"Unknown stage: {start_stage}. Valid stages: {', '.join(stages)}")
    start_idx = stages.index(start_stage)
    return stages[start_idx:]
