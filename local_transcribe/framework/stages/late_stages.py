#!/usr/bin/env python3
"""
Late pipeline stages: turn building, speaker naming, output generation, cleanup.

These stages handle post-processing of transcripts after diarization.
"""

from typing import List, Optional, Any, Dict
from pathlib import Path

from local_transcribe.framework.pipeline_context import PipelineContext
from local_transcribe.framework.stages.base import PipelineStage, StageError
from local_transcribe.lib.program_logger import (
    log_status, log_progress, log_intermediate_save, log_completion
)


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
        return ["mode"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["transcript"]
    
    @property
    def applicable_modes(self) -> List[str]:
        # All modes except single_speaker_audio (which outputs CSV directly)
        return ["combined_audio", "split_audio", "vad_split_audio"]
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        can_run, reason = super().can_execute(context)
        if not can_run:
            return can_run, reason
        
        # Need either diarized_segments (combined) or word_segments (split)
        if context.mode == "combined_audio":
            if not context.diarized_segments:
                return False, "Missing diarized_segments for combined_audio mode"
        else:
            if not context.word_segments:
                return False, "Missing word_segments"
        
        return True, ""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.turn_building import build_turns
        
        intermediate_dir = context.get_intermediate_dir()
        registry = context.api.get("registry")
        
        turn_kwargs = {}
        if intermediate_dir:
            turn_kwargs['intermediate_dir'] = intermediate_dir
        
        # Select input based on mode
        if context.mode == "combined_audio":
            input_segments = context.diarized_segments
            if not input_segments:
                raise StageError(self.name, "Missing diarized_segments for combined_audio mode")
        else:
            # For split_audio, use word_segments (which already have speaker labels)
            input_segments = context.word_segments
            if not input_segments:
                raise StageError(self.name, "Missing word_segments")
        
        log_progress(f"Building turns from {len(input_segments)} segments")
        
        transcript = build_turns(input_segments, mode=context.mode, **turn_kwargs)
        context.transcript = transcript
        
        # Save turns
        if intermediate_dir and registry:
            json_turns_writer = registry.get_output_writer("turns-json")
            
            if context.mode == "combined_audio":
                turns_file = intermediate_dir / "turns" / "raw_turns.json"
            else:
                turns_file = intermediate_dir / "turns" / "merged_turns.json"
            
            turns_file.parent.mkdir(parents=True, exist_ok=True)
            json_turns_writer.write(transcript, turns_file)
            log_intermediate_save(str(turns_file), "Turns saved to")
        
        log_completion("Turn building complete")
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
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["combined_audio", "split_audio", "vad_split_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.lib.speaker_namer import assign_speaker_names
        
        interactive = getattr(context.args, 'interactive', False)
        
        log_progress("Assigning speaker names" + (" (interactive)" if interactive else ""))
        
        context.transcript = assign_speaker_names(
            context.transcript, 
            interactive, 
            context.mode
        )
        
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
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["combined_audio", "split_audio", "vad_split_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.framework.output_manager import OutputManager
        
        registry = context.api.get("registry")
        if registry is None:
            raise StageError(self.name, "Registry not found in api")
        
        # Create raw output directory
        raw_dir = context.paths["root"] / "Transcript_Raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        log_status(f"Writing raw outputs to {raw_dir}")
        
        output_manager = OutputManager.get_instance(registry)
        if output_manager is None:
            raise StageError(self.name, "Output manager not available")
        
        # Determine audio config for video generation
        audio_config = self._get_audio_config(context)
        
        # Get selected outputs
        selected_outputs = getattr(context.args, 'selected_outputs', [])
        if not selected_outputs:
            selected_outputs = list(registry.list_output_writers().keys())
        
        # Determine word segments for detailed timing
        word_segments = self._get_word_segments(context)
        
        # Check if video generation is supported for this mode
        generate_video = context.mode != "vad_split_audio"
        
        log_progress(f"Writing outputs with formats: {selected_outputs}")
        
        if context.transcript is None:
            raise StageError(self.name, "No transcript available for output generation")
        
        output_manager.write_selected_outputs(
            context.transcript,
            {**context.paths, "merged": raw_dir},
            selected_outputs,
            audio_config,
            generate_video=generate_video,
            word_segments=word_segments
        )
        
        log_completion("Output generation complete")
        return context
    
    def _get_audio_config(self, context: PipelineContext) -> Any:
        """Get appropriate audio config based on mode."""
        if context.mode == "combined_audio":
            return context.standardized_audio
        elif context.mode == "split_audio":
            return context.speaker_files
        elif context.mode == "vad_split_audio":
            return context.standardized_speaker_files
        return None
    
    def _get_word_segments(self, context: PipelineContext) -> Optional[List[Any]]:
        """Get word segments for detailed timing."""
        if context.mode == "combined_audio":
            return context.diarized_segments
        elif context.mode == "split_audio":
            return context.word_segments
        return None


class SingleSpeakerOutputStage(PipelineStage):
    """Stage for generating CSV output for single speaker mode."""
    
    @property
    def name(self) -> str:
        return "single_speaker_output"
    
    @property
    def description(self) -> str:
        return "Generate CSV output for single speaker transcript"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["word_segments", "paths"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return []  # Output is a file
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["single_speaker_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        import csv
        
        outdir = context.get_output_dir()
        transcript = context.word_segments  # This is text for single speaker mode
        
        # Split transcript into words and save as CSV
        words = str(transcript).split()
        csv_path = outdir / "transcript.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Line', 'Word'])
            for i, word in enumerate(words, 1):
                writer.writerow([i, word])
        
        log_completion(f"Transcript saved to {csv_path}")
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
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["combined_audio", "split_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.pre_LLM_transcript_preparation import prepare_transcript_for_llm
        
        args = context.args
        
        try:
            prep_result = prepare_transcript_for_llm(
                context.transcript,
                max_words_per_segment=getattr(args, 'max_words_per_segment', 500),
                preparation_mode=getattr(args, 'preparation_mode', 'basic'),
                standardize_speakers=getattr(args, 'standardize_speakers', True),
                normalize_whitespace=getattr(args, 'normalize_whitespace', True),
                handle_special_chars=getattr(args, 'handle_special_chars', True)
            )
            
            # Update transcript with processed turns
            context.transcript = prep_result['turns']
            
            # Store prep_result for cleanup stage
            context.prep_result = prep_result
            
            log_completion(f"Transcript preparation complete: {prep_result['stats']['segments_created']} segments created", {
                "original_turns": prep_result['stats']['original_turns'],
                "words_processed": prep_result['stats']['words_processed'],
                "turns_split": prep_result['stats']['turns_split']
            })
            
        except Exception as e:
            log_status(f"Warning: Error during transcript preparation: {str(e)}", "WARNING")
            log_progress("Continuing with original transcript")
            context.prep_result = None
        
        return context


class TranscriptCleanupStage(PipelineStage):
    """Stage for LLM-based transcript cleanup."""
    
    @property
    def name(self) -> str:
        return "transcript_cleanup"
    
    @property
    def description(self) -> str:
        return "LLM-based transcript cleanup"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcript", "transcript_cleanup_provider"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return []  # Outputs are files
    
    @property
    def is_optional(self) -> bool:
        return True
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["combined_audio", "split_audio"]
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        # Check base requirements
        can_run, reason = super().can_execute(context)
        if not can_run:
            return can_run, reason
        
        # Need prep_result with segments
        prep_result = getattr(context, 'prep_result', None)
        if not prep_result or 'segments' not in prep_result:
            return False, "No prepared segments available"
        
        return True, ""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        args = context.args
        prep_result = context.prep_result
        
        if not prep_result or 'segments' not in prep_result:
            raise StageError(self.name, "No prepared segments available")
        
        if context.transcript_cleanup_provider is None:
            raise StageError(self.name, "No transcript cleanup provider configured")
        
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
        
        return context


class CleanupStage(PipelineStage):
    """Stage for cleaning up temporary files."""
    
    @property
    def name(self) -> str:
        return "cleanup"
    
    @property
    def description(self) -> str:
        return "Clean up temporary files"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["paths"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return []
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.lib.audio_processor import cleanup_temp_audio
        
        outdir = context.get_output_dir()
        cleanup_temp_audio(outdir)
        
        log_completion("Temporary files cleaned up")
        return context
