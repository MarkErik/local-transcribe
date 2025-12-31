#!/usr/bin/env python3
"""
Pipeline stages for modular transcription processing.

This module provides backward-compatible imports from the new stages package.
For new code, import directly from local_transcribe.framework.stages.

This module re-exports:
- PipelineStage: Abstract base class for all stages
- Stage implementations: TurnBuildingStage, SpeakerNamingStage, etc.
- Utility functions: get_stage, get_stages_for_reentry
"""

from typing import List

# Re-export base class and utilities from new stages package
from local_transcribe.framework.stages.base import (
    PipelineStage,
    StageResult,
    StageError,
    StageStatus,
)

# Re-export late stages (these are used by pipeline_reentry.py)
from local_transcribe.framework.stages.late_stages import (
    TurnBuildingStage,
    SpeakerNamingStage,
    OutputGenerationStage,
    TranscriptPreparationStage,
    TranscriptCleanupStage,
)

# Re-export early stages
from local_transcribe.framework.stages.early_stages import (
    AudioStandardizationStage,
    TranscriptionAlignmentStage,
    DeIdentificationStage,
    DiarizationStage,
)

# Import for re-export
from local_transcribe.framework.pipeline_context import PipelineContext
from local_transcribe.lib.program_logger import log_status, log_progress, log_intermediate_save, log_completion


# Stage registry for easy access - includes all stages
STAGE_REGISTRY = {
    # Early stages
    "audio_standardization": AudioStandardizationStage,
    "transcription_alignment": TranscriptionAlignmentStage,
    "de_identification": DeIdentificationStage,
    "diarization": DiarizationStage,
    # Late stages
    "turn_building": TurnBuildingStage,
    "speaker_naming": SpeakerNamingStage,
    "output_generation": OutputGenerationStage,
    "transcript_preparation": TranscriptPreparationStage,
    "transcript_cleanup": TranscriptCleanupStage,
}


def get_stage(stage_name: str) -> PipelineStage:
    """
    Get a stage instance by name.
    
    Args:
        stage_name: Name of the stage
        
    Returns:
        Instance of the requested stage
        
    Raises:
        ValueError: If stage name is not recognized
    """
    if stage_name not in STAGE_REGISTRY:
        available = ', '.join(STAGE_REGISTRY.keys())
        raise ValueError(f"Unknown stage: {stage_name}. Available stages: {available}")
    
    return STAGE_REGISTRY[stage_name]()


def get_stages_for_reentry(start_stage: str = "turn_building") -> List[PipelineStage]:
    """
    Get the list of stages to execute for pipeline re-entry.
    
    Args:
        start_stage: The stage to start from
        
    Returns:
        List of PipelineStage instances in execution order
    """
    from local_transcribe.framework.pipeline_context import get_stage_order
    
    stage_order = get_stage_order()
    
    if start_stage not in stage_order:
        raise ValueError(f"Unknown stage: {start_stage}")
    
    start_idx = stage_order.index(start_stage)
    stages_to_run = stage_order[start_idx:]
    
    # Filter to only stages we have implementations for
    stages = []
    for stage_name in stages_to_run:
        if stage_name in STAGE_REGISTRY:
            stages.append(STAGE_REGISTRY[stage_name]())
    
    return stages


# For backward compatibility, export all names that may have been imported
__all__ = [
    'PipelineStage',
    'StageResult',
    'StageError',
    'StageStatus',
    'TurnBuildingStage',
    'SpeakerNamingStage',
    'OutputGenerationStage',
    'TranscriptPreparationStage',
    'TranscriptCleanupStage',
    'AudioStandardizationStage',
    'TranscriptionAlignmentStage',
    'DeIdentificationStage',
    'DiarizationStage',
    'STAGE_REGISTRY',
    'get_stage',
    'get_stages_for_reentry',
]
