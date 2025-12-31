#!/usr/bin/env python3
"""
Pipeline executor for running stage sequences.

This module provides the PipelineExecutor class which orchestrates the
execution of pipeline stages in the correct order, handling errors,
skipping, and dry runs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from local_transcribe.framework.pipeline_context import PipelineContext
from local_transcribe.framework.stages.base import (
    PipelineStage, 
    StageResult, 
    StageStatus,
    StageError,
)
from local_transcribe.lib.program_logger import log_status, log_completion, log_progress, log_intermediate_save


@dataclass
class PipelineResult:
    """Result from executing a complete pipeline."""
    success: bool
    completed_stages: List[str] = field(default_factory=list)
    skipped_stages: List[str] = field(default_factory=list)
    failed_stage: Optional[str] = None
    error: Optional[Exception] = None
    stage_results: List[StageResult] = field(default_factory=list)
    
    @property
    def exit_code(self) -> int:
        """Return appropriate exit code."""
        return 0 if self.success else 1


class PipelineExecutor:
    """
    Executes a sequence of pipeline stages.
    
    The executor handles:
    - Running stages in order
    - Skipping stages based on context
    - Error handling and reporting
    - Dry run mode for validation
    """
    
    def __init__(self, stages: List[PipelineStage]):
        """
        Initialize the executor with a list of stages.
        
        Args:
            stages: List of PipelineStage instances in execution order
        """
        self.stages = stages
    
    def execute(self, context: PipelineContext) -> PipelineResult:
        """
        Execute all stages in sequence.
        
        Args:
            context: The pipeline context
            
        Returns:
            PipelineResult with execution summary
        """
        result = PipelineResult(success=True)
        
        log_status(f"Starting pipeline with {len(self.stages)} stages")
        
        for stage in self.stages:
            # Skip stages that don't apply to this mode
            if not stage.applies_to_mode(context.mode):
                result.skipped_stages.append(stage.name)
                continue
            
            # Execute the stage
            context, stage_result = stage.execute_safe(context)
            result.stage_results.append(stage_result)
            
            if stage_result.status == StageStatus.COMPLETED:
                result.completed_stages.append(stage.name)
            elif stage_result.status == StageStatus.SKIPPED:
                result.skipped_stages.append(stage.name)
            elif stage_result.status == StageStatus.FAILED:
                result.success = False
                result.failed_stage = stage.name
                result.error = stage_result.error
                
                log_status(f"Pipeline failed at stage: {stage.name}", "ERROR")
                if stage_result.error:
                    log_status(f"Error: {stage_result.error}", "ERROR")
                
                break
        
        if result.success:
            log_completion(f"Pipeline complete: {len(result.completed_stages)} stages executed")
        
        return result
    
    def execute_dry_run(self, context: PipelineContext) -> PipelineResult:
        """
        Perform a dry run to validate all stages without executing.
        
        Args:
            context: The pipeline context
            
        Returns:
            PipelineResult with validation summary
        """
        result = PipelineResult(success=True)
        
        print("\n=== Pipeline Dry Run ===")
        print(f"Mode: {context.mode}")
        print(f"Stages: {len(self.stages)}\n")
        
        for stage in self.stages:
            if not stage.applies_to_mode(context.mode):
                print(f"  - [{stage.name}] N/A for mode {context.mode}")
                result.skipped_stages.append(stage.name)
                continue
            
            stage_result = stage.execute_dry_run(context)
            result.stage_results.append(stage_result)
            
            if stage_result.status == StageStatus.NOT_STARTED:
                # Would run successfully
                result.completed_stages.append(stage.name)
            elif stage_result.status == StageStatus.SKIPPED:
                result.skipped_stages.append(stage.name)
            elif stage_result.status == StageStatus.FAILED:
                result.success = False
                result.failed_stage = stage.name
        
        print(f"\n{'✓' if result.success else '✗'} Dry run {'passed' if result.success else 'failed'}")
        
        if result.failed_stage:
            print(f"  First failure: {result.failed_stage}")
        
        return result


def create_pipeline_for_mode(mode: str) -> List[PipelineStage]:
    """
    Create the appropriate pipeline stages for a given processing mode.
    
    Args:
        mode: Processing mode (combined_audio, split_audio, single_speaker_audio, vad_split_audio)
        
    Returns:
        List of PipelineStage instances in execution order
    
    Pipeline Order:
        1. Audio Standardization - Convert audio to standard format
        2. Transcription/Alignment - Transcribe and get word-level timing
        3. De-identification - Remove personal identifiers (optional)
        4. Diarization - Assign speakers (combined_audio only)
        5. Turn Building - Group words into conversational turns
        6. Speaker Naming - Map speaker IDs to names
        7. Output Generation - Write raw transcript files to Transcript_Raw/
        8. Transcript Cleanup - LLM-based cleanup (optional, disabled by default)
        9. Cleaned Output Generation - Write cleaned files to Transcript_Processed/ (if cleanup ran)
        10. Cleanup - Remove temporary files
    """
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
    
    if mode == "single_speaker_audio":
        return [
            AudioStandardizationStage(),
            TranscriptionAlignmentStage(),
            DeIdentificationStage(),
            SingleSpeakerOutputStage(),
            CleanupStage(),
        ]
    
    elif mode == "combined_audio":
        return [
            AudioStandardizationStage(),
            TranscriptionAlignmentStage(),
            DeIdentificationStage(),
            DiarizationStage(),
            TurnBuildingStage(),
            SpeakerNamingStage(),
            OutputGenerationStage(),
            # LLM cleanup stages (optional, skipped if --enable-cleanup not set)
            TranscriptCleanupStage(),
            CleanedOutputGenerationStage(),
            CleanupStage(),
        ]
    
    elif mode == "split_audio":
        return [
            AudioStandardizationStage(),
            TranscriptionAlignmentStage(),
            DeIdentificationStage(),
            # No diarization needed - speakers already known
            TurnBuildingStage(),
            SpeakerNamingStage(),
            OutputGenerationStage(),
            # LLM cleanup stages (optional, skipped if --enable-cleanup not set)
            TranscriptCleanupStage(),
            CleanedOutputGenerationStage(),
            CleanupStage(),
        ]
    
    elif mode == "vad_split_audio":
        # VAD pipeline has its own transcription flow through build_turns_vad_split_audio
        # So we use a specialized stage set
        return [
            # VAD pipeline handles audio/transcription internally via VADTranscriptionStage
            VADTranscriptionStage(),
            DeIdentificationStage(),
            # Turn building happens in VADTranscriptionStage
            SpeakerNamingStage(),
            OutputGenerationStage(),
            # LLM cleanup stages (optional, skipped if --enable-cleanup not set)
            TranscriptCleanupStage(),
            CleanedOutputGenerationStage(),
            CleanupStage(),
        ]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


class VADTranscriptionStage(PipelineStage):
    """
    Combined VAD + transcription + turn building stage for VAD pipeline.
    
    This stage encapsulates the entire VAD-first pipeline which handles
    audio processing, VAD segmentation, transcription, and turn building
    as an integrated unit.
    """
    
    @property
    def name(self) -> str:
        return "vad_transcription"
    
    @property
    def description(self) -> str:
        return "VAD-based transcription and turn building"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["speaker_files", "transcriber_provider"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["transcript", "standardized_speaker_files"]
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["vad_split_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.vad import VADBlockBuilderConfig
        from local_transcribe.processing.turn_building import build_turns_vad_split_audio
        from local_transcribe.lib.environment import ensure_file
        
        args = context.args
        intermediate_dir = context.get_intermediate_dir()
        registry = context.api.get("registry")
        
        log_status("Using VAD-first pipeline for split audio files")
        
        # Build VAD config from CLI args
        vad_config = VADBlockBuilderConfig(
            merge_gap_threshold_ms=getattr(args, 'vad_merge_gap_ms', 500),
        )
        
        # Build speaker audio files dict (need absolute paths)
        speaker_audio_paths = {}
        for speaker_name, audio_file in context.speaker_files.items():
            speaker_audio_paths[speaker_name] = str(ensure_file(audio_file, speaker_name))
        
        context.standardized_speaker_files = speaker_audio_paths
        
        # Build additional transcription kwargs
        transcription_kwargs = {}
        if getattr(args, 'transcriber_model', None):
            transcription_kwargs['transcriber_model'] = args.transcriber_model
        
        # Pass include_disfluencies setting if specified (for remote transcriber)
        include_disfluencies = getattr(args, 'include_disfluencies', None)
        if include_disfluencies is not None:
            transcription_kwargs['include_disfluencies'] = include_disfluencies
        
        # Run VAD pipeline
        transcript = build_turns_vad_split_audio(
            speaker_audio_files=speaker_audio_paths,
            transcriber_provider=context.transcriber_provider,
            config=vad_config,
            intermediate_dir=intermediate_dir,
            models_dir=context.models_dir,
            vad_threshold=getattr(args, 'vad_threshold', 0.5),
            **transcription_kwargs,
        )
        
        context.transcript = transcript
        
        # Save turns
        if intermediate_dir and registry:
            json_turns_writer = registry.get_output_writer("turns-json")
            turns_file = intermediate_dir / "turns" / "vad_turns.json"
            turns_file.parent.mkdir(parents=True, exist_ok=True)
            json_turns_writer.write(transcript, turns_file)
            log_intermediate_save(str(turns_file), "VAD turns saved to")
        
        log_completion("VAD transcription and turn building complete")
        return context
