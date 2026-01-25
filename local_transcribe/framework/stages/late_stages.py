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
        
        # Video generation is now supported for all multi-speaker modes
        # VAD mode uses vad-video writer, others use standard video writer
        generate_video = True
        
        log_progress(f"Writing outputs with formats: {selected_outputs}")
        
        if context.transcript is None:
            raise StageError(self.name, "No transcript available for output generation")
        
        output_manager.write_selected_outputs(
            context.transcript,
            {**context.paths, "merged": raw_dir},
            selected_outputs,
            audio_config,
            generate_video=generate_video,
            word_segments=word_segments,
            mode=context.mode,
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


class TranscriptCleanupStage(PipelineStage):
    """
    Stage for LLM-based transcript cleanup.
    
    This stage processes the TranscriptFlow through an LLM to clean up
    the transcript text while preserving turn structure. It:
    - Performs health check on LLM server
    - Batches turns for efficient processing
    - Creates a cleaned TranscriptFlow with smoothed text
    - Preserves speaker assignments and timing boundaries
    - Clears word-level timing (no longer valid after cleanup)
    
    The cleaned transcript is stored in context.cleaned_transcript and
    can be written to Transcript_Processed/ by CleanedOutputGenerationStage.
    """
    
    @property
    def name(self) -> str:
        return "transcript_cleanup"
    
    @property
    def description(self) -> str:
        return "LLM-based transcript cleanup (optional)"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcript"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["cleaned_transcript"]
    
    @property
    def is_optional(self) -> bool:
        return True
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["combined_audio", "split_audio", "vad_split_audio"]
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        """Check if cleanup stage should run."""
        # Check if cleanup is explicitly enabled
        enable_cleanup = getattr(context.args, 'enable_cleanup', False)
        if not enable_cleanup:
            return False, "Transcript cleanup is disabled (use --enable-cleanup to enable)"
        
        # Check if transcript is available
        if context.transcript is None:
            return False, "No transcript available for cleanup"
        
        # Check if cleanup provider is configured
        if context.transcript_cleanup_provider is None:
            return False, "No transcript cleanup provider configured"
        
        return True, ""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.transcript_cleanup import BatchProcessor
        from local_transcribe.processing.transcript_cleanup.batch_processor import BatchConfig
        
        if context.transcript_cleanup_provider is None:
            raise StageError(self.name, "No transcript cleanup provider configured")
        
        if context.transcript is None:
            raise StageError(self.name, "No transcript available for cleanup")
        
        # Perform health check on LLM server
        log_progress("Checking LLM server availability...")
        
        if hasattr(context.transcript_cleanup_provider, 'health_check'):
            server_info = context.transcript_cleanup_provider.health_check(timeout=10.0)
            
            if not server_info.available:
                error_msg = server_info.error or "Server not responding"
                log_status(f"LLM server health check failed: {error_msg}", "WARNING")
                log_progress("Skipping transcript cleanup - LLM server unavailable")
                return context
            
            if server_info.is_harmony_format:
                log_progress(f"LLM server uses Harmony format (model: {server_info.model_name})")
            else:
                log_progress(f"LLM server available (model: {server_info.model_name or 'unknown'})")
        else:
            log_progress("LLM provider does not support health check, proceeding anyway...")
        
        # Get batch configuration from args
        max_words = getattr(context.args, 'cleanup_batch_words', 500)
        max_turns = getattr(context.args, 'cleanup_batch_turns', 20)
        
        config = BatchConfig(
            max_words_per_batch=max_words,
            max_turns_per_batch=max_turns
        )
        
        # Create batch processor
        processor = BatchProcessor(config)
        
        # Progress callback
        def progress_callback(batch_num: int, total_batches: int, status: str):
            log_progress(f"[{batch_num}/{total_batches}] {status}")
        
        log_progress(f"Processing transcript with {len(context.transcript.turns)} turns...")
        
        try:
            # Process transcript through LLM
            cleaned_transcript = processor.process_transcript(
                context.transcript,
                context.transcript_cleanup_provider,
                progress_callback=progress_callback
            )
            
            # Store cleaned transcript in context
            context.cleaned_transcript = cleaned_transcript
            
            # Log statistics
            original_words = sum(len(t.text.split()) for t in context.transcript.turns)
            cleaned_words = sum(len(t.text.split()) for t in cleaned_transcript.turns)
            
            log_completion(f"Transcript cleanup complete", {
                "turns_processed": len(cleaned_transcript.turns),
                "original_words": original_words,
                "cleaned_words": cleaned_words,
                "word_difference": cleaned_words - original_words
            })
            
        except Exception as e:
            log_status(f"Error during transcript cleanup: {str(e)}", "ERROR")
            log_progress("Cleaned transcript will not be available")
            # Don't fail the pipeline, just skip cleanup
            context.cleaned_transcript = None
        
        return context


class CleanedOutputGenerationStage(PipelineStage):
    """
    Stage for generating output files from cleaned transcript.
    
    This stage writes the LLM-cleaned transcript to Transcript_Processed/
    using the existing output writers. Files are named with '_cleaned' suffix
    to distinguish them from raw transcripts.
    """
    
    @property
    def name(self) -> str:
        return "cleaned_output_generation"
    
    @property
    def description(self) -> str:
        return "Generate output files from cleaned transcript"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["cleaned_transcript", "paths"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return []  # Outputs are files
    
    @property
    def is_optional(self) -> bool:
        return True
    
    @property
    def applicable_modes(self) -> List[str]:
        return ["combined_audio", "split_audio", "vad_split_audio"]
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        """Check if cleaned output generation should run."""
        # Only run if we have a cleaned transcript
        cleaned_transcript = getattr(context, 'cleaned_transcript', None)
        if cleaned_transcript is None:
            return False, "No cleaned transcript available"
        
        return True, ""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        registry = context.api.get("registry")
        if registry is None:
            raise StageError(self.name, "Registry not found in api")
        
        # Create processed output directory
        processed_dir = context.paths["root"] / "Transcript_Processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        log_status(f"Writing cleaned outputs to {processed_dir}")
        
        cleaned_transcript = context.cleaned_transcript
        
        # Get selected outputs (same as raw, but with _cleaned suffix)
        selected_outputs = getattr(context.args, 'selected_outputs', [])
        if not selected_outputs:
            selected_outputs = ['timestamped-txt', 'plain-txt', 'turns-json']
        
        # Write each format with _cleaned suffix
        try:
            if 'timestamped-txt' in selected_outputs:
                writer = registry.get_output_writer("timestamped-txt")
                writer.write(cleaned_transcript, processed_dir / "transcript_cleaned.timestamped.txt")
                log_progress("Written: transcript_cleaned.timestamped.txt")
            
            if 'plain-txt' in selected_outputs:
                writer = registry.get_output_writer("plain-txt")
                writer.write(cleaned_transcript, processed_dir / "transcript_cleaned.txt")
                log_progress("Written: transcript_cleaned.txt")
            
            if 'markdown' in selected_outputs:
                writer = registry.get_output_writer("markdown")
                writer.write(cleaned_transcript, processed_dir / "transcript_cleaned.md")
                log_progress("Written: transcript_cleaned.md")
            
            if 'dialogue-script' in selected_outputs:
                writer = registry.get_output_writer("dialogue-script")
                writer.write(cleaned_transcript, processed_dir / "transcript_cleaned.script.txt")
                log_progress("Written: transcript_cleaned.script.txt")
            
            if 'turns-json' in selected_outputs:
                writer = registry.get_output_writer("turns-json")
                writer.write(cleaned_transcript, processed_dir / "transcript_cleaned.turns.json")
                log_progress("Written: transcript_cleaned.turns.json")
            
            # Note: We skip html-timeline and video for cleaned transcripts
            # because they rely on word-level timing which is no longer valid
            if 'html-timeline' in selected_outputs:
                log_progress("Skipping HTML timeline for cleaned transcript (word timing not available)")
            
            if 'video' in selected_outputs:
                log_progress("Skipping video generation for cleaned transcript (word timing not available)")
            
        except Exception as e:
            log_status(f"Error writing cleaned output: {e}", "ERROR")
            raise StageError(self.name, f"Failed to write cleaned outputs: {e}")
        
        log_completion("Cleaned output generation complete")
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
