#!/usr/bin/env python3
"""
Pipeline stages for modular transcription processing.

This module defines the abstract PipelineStage class and concrete
implementations for each stage of the transcription pipeline.
Each stage is a self-contained unit that can be executed independently
or as part of the full pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from pathlib import Path

from local_transcribe.framework.pipeline_context import PipelineContext
from local_transcribe.lib.program_logger import log_status, log_progress, log_intermediate_save, log_completion


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage represents a discrete step in the transcription pipeline.
    Stages can be executed independently or chained together.
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
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        """
        Check if this stage can execute given the current context.
        
        Returns:
            Tuple of (can_execute, reason_if_not)
        """
        for required in self.required_inputs:
            if not hasattr(context, required) or getattr(context, required) is None:
                return False, f"Missing required input: {required}"
        return True, ""
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute this pipeline stage.
        
        Args:
            context: The pipeline context with all required inputs
            
        Returns:
            Updated context with this stage's outputs
        """
        pass
    
    def execute_dry_run(self, context: PipelineContext) -> None:
        """
        Perform a dry run of this stage (validation only, no execution).
        
        Args:
            context: The pipeline context
        """
        can_run, reason = self.can_execute(context)
        status = "✓ Ready" if can_run else f"✗ {reason}"
        print(f"  [{self.name}] {status}")


class TurnBuildingStage(PipelineStage):
    """Stage for building conversation turns from diarized segments."""
    
    @property
    def name(self) -> str:
        return "turn_building"
    
    @property
    def description(self) -> str:
        return "Group word segments into conversational turns"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["diarized_segments", "mode"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["transcript"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.turn_building import build_turns
        
        log_status("Building conversation turns")
        
        # Prepare kwargs
        turn_kwargs = {}
        if context.get_intermediate_dir():
            turn_kwargs['intermediate_dir'] = context.get_intermediate_dir()
        
        # Build turns
        transcript = build_turns(context.diarized_segments, mode=context.mode, **turn_kwargs)
        
        # Save turns
        intermediate_dir = context.get_intermediate_dir()
        if intermediate_dir:
            turns_dir = intermediate_dir / "turns"
            turns_dir.mkdir(parents=True, exist_ok=True)
            
            json_turns_writer = context.api["registry"].get_output_writer("turns-json")
            turns_file = turns_dir / "raw_turns.json"
            json_turns_writer.write(transcript, turns_file)
            log_intermediate_save(str(turns_file), "Raw turns saved to")
        
        context.transcript = transcript
        context.mark_stage_complete(self.name)
        
        return context


class SpeakerNamingStage(PipelineStage):
    """Stage for assigning human-readable names to speakers."""
    
    @property
    def name(self) -> str:
        return "speaker_naming"
    
    @property
    def description(self) -> str:
        return "Map speaker IDs to human-readable names"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcript"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["transcript"]  # Modified in place
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.lib.speaker_namer import assign_speaker_names
        
        log_status("Assigning speaker names")
        
        interactive = getattr(context.args, 'interactive', False)
        context.transcript = assign_speaker_names(context.transcript, interactive, context.mode)
        
        context.mark_stage_complete(self.name)
        
        return context


class OutputGenerationStage(PipelineStage):
    """Stage for generating output files in various formats."""
    
    @property
    def name(self) -> str:
        return "output_generation"
    
    @property
    def description(self) -> str:
        return "Generate output files in selected formats"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcript", "paths"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return []  # Outputs are files, not context attributes
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.framework.output_manager import OutputManager
        
        # Create raw output directory
        raw_dir = context.paths["root"] / "Transcript_Raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_status(f"Writing raw outputs to {raw_dir}")
        
        output_manager = OutputManager.get_instance(context.api["registry"])
        
        # Determine audio config for video generation
        audio_config = None
        if context.standardized_audio:
            audio_config = context.standardized_audio
        elif context.mode == "split_audio" and context.speaker_files:
            audio_config = context.speaker_files
        
        # Get selected outputs
        selected_outputs = getattr(context.args, 'selected_outputs', [])
        if not selected_outputs:
            # Default to all outputs
            selected_outputs = list(context.api["registry"].list_output_writers().keys())
        
        log_progress(f"Writing outputs with formats: {selected_outputs}")
        
        # Write outputs
        output_manager.write_selected_outputs(
            context.transcript,
            {**context.paths, "merged": raw_dir},
            selected_outputs,
            audio_config,
            generate_video=True,
            word_segments=context.diarized_segments
        )
        
        context.mark_stage_complete(self.name)
        
        return context


class TranscriptPreparationStage(PipelineStage):
    """Stage for preparing transcript for LLM processing."""
    
    @property
    def name(self) -> str:
        return "transcript_preparation"
    
    @property
    def description(self) -> str:
        return "Prepare transcript for LLM processing"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcript"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["transcript", "prep_result"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.pre_LLM_transcript_preparation import prepare_transcript_for_llm
        
        log_status("Preparing transcript for LLM processing")
        
        try:
            prep_result = prepare_transcript_for_llm(
                context.transcript,
                max_words_per_segment=getattr(context.args, 'max_words_per_segment', 500),
                preparation_mode=getattr(context.args, 'preparation_mode', 'basic'),
                standardize_speakers=getattr(context.args, 'standardize_speakers', True),
                normalize_whitespace=getattr(context.args, 'normalize_whitespace', True),
                handle_special_chars=getattr(context.args, 'handle_special_chars', True)
            )
            
            # Update transcript with processed turns
            context.transcript = prep_result['turns']
            
            # Store prep_result for cleanup stage
            if not hasattr(context, 'prep_result'):
                context.prep_result = None
            context.prep_result = prep_result
            
            log_completion(f"Transcript preparation complete: {prep_result['stats']['segments_created']} segments created", {
                "original_turns": prep_result['stats']['original_turns'],
                "words_processed": prep_result['stats']['words_processed'],
                "turns_split": prep_result['stats']['turns_split']
            })
        except Exception as e:
            log_status(f"Warning: Error during transcript preparation: {str(e)}", "WARNING")
            log_progress("Continuing with original transcript")
        
        context.mark_stage_complete(self.name)
        
        return context


class TranscriptCleanupStage(PipelineStage):
    """Stage for LLM-based transcript cleanup."""
    
    @property
    def name(self) -> str:
        return "transcript_cleanup"
    
    @property
    def description(self) -> str:
        return "LLM-based transcript cleanup (optional)"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcript", "transcript_cleanup_provider"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return []  # Outputs are files
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        # This stage is optional - only runs if provider is configured
        if not context.transcript_cleanup_provider:
            return False, "No transcript cleanup provider configured (optional stage)"
        return True, ""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.transcript_cleanup_provider:
            log_progress("No transcript cleanup selected, skipping.")
            context.mark_stage_complete(self.name)
            return context
        
        log_status(f"Cleaning up transcript with {context.args.transcript_cleanup_provider}")
        
        prep_result = getattr(context, 'prep_result', None)
        if not prep_result or 'segments' not in prep_result:
            log_progress("No prepared segments available, skipping cleanup.")
            context.mark_stage_complete(self.name)
            return context
        
        log_progress(f"Processing {len(prep_result['segments'])} segments")
        
        # Process each segment through LLM
        cleaned_segments = []
        for idx, segment in enumerate(prep_result['segments']):
            log_progress(f"[{idx+1}/{len(prep_result['segments'])}] Processing: {segment[:60]}...")
            cleaned = context.transcript_cleanup_provider.transcript_cleanup_segment(segment)
            cleaned_segments.append(cleaned)
            log_progress(f"[{idx+1}/{len(prep_result['segments'])}] Cleaned: {cleaned[:60]}...")
        
        # Write cleaned transcript to processed directory
        processed_dir = context.paths["root"] / "Transcript_Processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        cleaned_text_file = processed_dir / "transcript_cleaned.txt"
        cleaned_text_file.write_text('\n\n'.join(cleaned_segments) + '\n', encoding='utf-8')
        
        log_completion(f"Transcript cleanup complete: {cleaned_text_file}")
        log_progress("Raw transcript with timestamps available in Transcript_Raw/")
        
        context.mark_stage_complete(self.name)
        
        return context


# Stage registry for easy access
STAGE_REGISTRY = {
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
