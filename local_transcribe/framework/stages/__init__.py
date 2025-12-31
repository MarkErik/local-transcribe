#!/usr/bin/env python3
"""
Pipeline stages package.

This package contains all pipeline stage implementations, organized into:
- base: Abstract PipelineStage class and utilities
- early_stages: Audio standardization, transcription, de-identification, diarization
- late_stages: Turn building, speaker naming, output generation, cleanup
- executor: PipelineExecutor for running stage sequences
"""

from local_transcribe.framework.stages.base import (
    PipelineStage,
    StageResult,
    StageError,
    StageStatus,
)

from local_transcribe.framework.stages.early_stages import (
    AudioStandardizationStage,
    TranscriptionAlignmentStage,
    DeIdentificationStage,
    DiarizationStage,
)

from local_transcribe.framework.stages.late_stages import (
    TurnBuildingStage,
    SpeakerNamingStage,
    OutputGenerationStage,
    SingleSpeakerOutputStage,
    TranscriptCleanupStage,
    CleanedOutputGenerationStage,
    CleanupStage,
)

from local_transcribe.framework.stages.executor import (
    PipelineExecutor,
    create_pipeline_for_mode,
)

__all__ = [
    # Base
    'PipelineStage',
    'StageResult',
    'StageError',
    'StageStatus',
    # Early stages
    'AudioStandardizationStage',
    'TranscriptionAlignmentStage',
    'DeIdentificationStage',
    'DiarizationStage',
    # Late stages
    'TurnBuildingStage',
    'SpeakerNamingStage',
    'OutputGenerationStage',
    'SingleSpeakerOutputStage',
    'TranscriptCleanupStage',
    'CleanedOutputGenerationStage',
    'CleanupStage',
    # Executor
    'PipelineExecutor',
    'create_pipeline_for_mode',
]
