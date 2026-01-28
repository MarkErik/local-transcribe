#!/usr/bin/env python3
"""
Pipeline re-entry runner for resuming from checkpoints.

This module provides functionality to resume the transcription pipeline
from intermediate checkpoint files (e.g., corrected diarized JSON or 
edited TranscriptFlow JSON).
"""

from pathlib import Path
from typing import Optional, Dict, Any
import os

from local_transcribe.framework.pipeline_context import (
    PipelineContext,
    get_stage_order,
    get_stage_descriptions,
    get_stages_from
)
from local_transcribe.framework.checkpoint_loader import (
    load_diarized_checkpoint,
    validate_checkpoint_for_reentry,
    print_checkpoint_summary,
    get_mode_from_checkpoint,
    CheckpointValidationError,
    detect_checkpoint_type,
    load_transcript_flow_checkpoint,
    print_transcript_flow_summary
)
from local_transcribe.framework.pipeline_stages import get_stages_for_reentry
from local_transcribe.lib.speaker_mapper import (
    apply_speaker_mapping,
    create_speaker_mapping_from_args
)
from local_transcribe.lib.program_logger import log_status, log_progress, log_completion
from local_transcribe.lib.create_directories import ensure_session_dirs


def run_pipeline_from_checkpoint(args, api, root) -> int:
    """
    Resume pipeline from a checkpoint file.
    
    This function automatically detects the checkpoint type:
    - TranscriptFlow checkpoint (has 'turns' array) -> starts from de_identification or speaker_naming
    - Word segments checkpoint (has 'words' array) -> starts from turn_building
    
    Args:
        args: Parsed command line arguments (must include from_diarized_json)
        api: Pipeline API dictionary with registry and utilities
        root: Repository root path
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    checkpoint_path = Path(args.from_diarized_json)
    
    # Detect checkpoint type
    log_status(f"Detecting checkpoint type: {checkpoint_path}")
    
    try:
        checkpoint_type = detect_checkpoint_type(checkpoint_path)
    except CheckpointValidationError as e:
        print(f"ERROR: Failed to detect checkpoint type: {e}")
        return 1
    
    log_progress(f"Checkpoint type detected: {checkpoint_type}")
    
    # Route to appropriate handler based on checkpoint type
    if checkpoint_type == "transcript_flow":
        return run_pipeline_from_transcript_flow_checkpoint(args, api, root)
    elif checkpoint_type == "word_segments":
        return run_pipeline_from_word_segments_checkpoint(args, api, root)
    else:
        print(f"ERROR: Unknown checkpoint format. Expected 'turns' or 'words' array in JSON.")
        return 1


def run_pipeline_from_word_segments_checkpoint(args, api, root) -> int:
    """
    Resume pipeline from a word segments checkpoint file.
    
    This is the original re-entry path for diarized word segment JSON files.
    Starts from the turn_building stage.
    
    Args:
        args: Parsed command line arguments (must include from_diarized_json)
        api: Pipeline API dictionary with registry and utilities
        root: Repository root path
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    from local_transcribe.lib.environment import ensure_outdir
    from local_transcribe.framework.cli import interactive_reentry_prompt
    
    checkpoint_path = Path(args.from_diarized_json)
    
    # Load and validate checkpoint
    log_status(f"Loading word segments checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint_result = load_diarized_checkpoint(checkpoint_path)
    except CheckpointValidationError as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return 1
    
    # Print checkpoint summary
    print_checkpoint_summary(checkpoint_result)
    
    # Validate for re-entry
    is_valid, messages = validate_checkpoint_for_reentry(checkpoint_result, "turn_building")
    
    if messages:
        print("\n--- Validation Results ---")
        for msg in messages:
            print(f"  {msg}")
    
    if not is_valid:
        print("\nERROR: Checkpoint validation failed. Cannot proceed.")
        return 1
    
    # Determine mode
    mode = None
    if hasattr(args, 'mode') and args.mode:
        mode = args.mode
        log_progress(f"Using mode from command line: {mode}")
    else:
        mode = get_mode_from_checkpoint(checkpoint_result)
        if mode:
            log_progress(f"Detected mode from checkpoint: {mode}")
        else:
            mode = "combined_audio"
            log_progress(f"Could not detect mode, defaulting to: {mode}")
    
    args.mode = mode
    
    # Dry run - just show what would happen
    if args.dry_run:
        return run_dry_run(args, checkpoint_result)
    
    # Interactive mode for re-entry
    if args.interactive:
        args = interactive_reentry_prompt(args, api, checkpoint_result)
    
    # Apply speaker mapping if provided
    segments = checkpoint_result.segments
    
    if hasattr(args, 'speaker_mapping') and args.speaker_mapping:
        log_progress("Applying speaker name mapping")
        segments = apply_speaker_mapping(segments, args.speaker_mapping)
    elif hasattr(args, 'speaker_map') and args.speaker_map:
        # Parse from command line argument
        mapping = create_speaker_mapping_from_args(segments, args.speaker_map)
        if mapping:
            log_progress(f"Applying speaker mapping from --speaker-map: {mapping}")
            segments = apply_speaker_mapping(segments, mapping)
    
    # Ensure output directory
    outdir = ensure_outdir(args.outdir)
    
    # Write settings to file if DEBUG log level is set
    if args.log_level == "DEBUG":
        settings_path = os.path.join(outdir, "settings.txt")
        with open(settings_path, 'w') as f:
            f.write("Local-Transcribe Settings (Re-entry Mode)\n")
            f.write("=" * 40 + "\n\n")
            
            # Write all command line arguments
            f.write("Command Line Arguments:\n")
            f.write("-" * 25 + "\n")
            for key, value in vars(args).items():
                if key not in ['audio_files', 'outdir', 'from_diarized_json']:  # Skip these as they can be long
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Write processing mode
            f.write("Processing Mode:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Mode: {mode} (Re-entry)\n")
            f.write(f"Starting Stage: turn_building\n")
            f.write(f"Selected Outputs: {', '.join(args.selected_outputs)}\n")
            f.write("\n")
            
            # Write checkpoint info
            f.write("Checkpoint Information:\n")
            f.write("-" * 23 + "\n")
            f.write(f"Checkpoint File: {os.path.basename(str(checkpoint_path))}\n")
            f.write(f"Number of Segments: {len(segments)}\n")
        
        print(f"[DEBUG] Settings written to {settings_path}")
    
    # Set up directory structure for re-entry
    # We need minimal directories since we're starting from diarization
    capabilities = {
        "mode": mode,
        "has_builtin_alignment": False,
        "aligner": False,
        "diarization": False  # Already done
    }
    
    paths = ensure_session_dirs(outdir, mode, {}, capabilities)
    
    # Ensure turns directory exists
    turns_dir = paths["intermediate"] / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default outputs if not specified
    if not hasattr(args, 'selected_outputs') or not args.selected_outputs:
        if getattr(args, 'only_final_transcript', False):
            args.selected_outputs = ['timestamped-txt']
        else:
            from local_transcribe.framework.cli import get_available_writers
            # Get only writers compatible with the current mode
            available_writers = get_available_writers(mode, api["registry"], exclude_internal=True)
            args.selected_outputs = list(available_writers.keys())
    
    # Configure logging
    api["configure_global_logging"](log_level=args.log_level)
    
    # Build pipeline context
    context = PipelineContext(
        args=args,
        api=api,
        root=root,
        paths=paths,
        mode=mode,
        diarized_segments=segments,
        start_from_stage="turn_building",
        checkpoint_metadata=checkpoint_result.metadata,
        input_checkpoint_path=checkpoint_path
    )
    
    # Handle audio for video generation
    if hasattr(args, 'audio_for_video') and args.audio_for_video:
        audio_path = Path(args.audio_for_video)
        if audio_path.exists():
            context.standardized_audio = audio_path
        else:
            log_progress(f"Warning: Audio file not found: {audio_path}")
    
    # Get transcript cleanup provider if specified
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider:
        try:
            context.transcript_cleanup_provider = api["registry"].get_transcript_cleanup_provider(
                args.transcript_cleanup_provider
            )
        except ValueError:
            log_progress(f"Warning: Transcript cleanup provider not found: {args.transcript_cleanup_provider}")
    
    # Execute pipeline stages
    log_status("Starting pipeline from turn_building stage")
    
    stages = get_stages_for_reentry("turn_building")
    
    for stage in stages:
        stage_name = stage.name
        
        # Check if we should skip this stage
        if context.should_skip_stage(stage_name):
            log_progress(f"Skipping stage: {stage_name}")
            continue
        
        # Check if stage can execute
        can_run, reason = stage.can_execute(context)
        if not can_run:
            if stage_name == "transcript_cleanup":
                # This stage is optional
                log_progress(f"Skipping optional stage {stage_name}: {reason}")
                continue
            else:
                print(f"ERROR: Cannot execute stage {stage_name}: {reason}")
                return 1
        
        log_status(f"Executing stage: {stage_name}")
        
        try:
            context = stage.execute(context)
        except Exception as e:
            print(f"ERROR: Stage {stage_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Final summary
    log_completion("Pipeline re-entry complete")
    print(f"[i] Artifacts written to: {paths['root']}")
    
    return 0


def run_pipeline_from_transcript_flow_checkpoint(args, api, root) -> int:
    """
    Resume pipeline from a TranscriptFlow checkpoint file.
    
    This is for re-entry from edited transcripts (turns-json format).
    Starts from de_identification stage (if enabled) or speaker_naming stage.
    
    Args:
        args: Parsed command line arguments (must include from_diarized_json)
        api: Pipeline API dictionary with registry and utilities
        root: Repository root path
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    from local_transcribe.lib.environment import ensure_outdir
    
    checkpoint_path = Path(args.from_diarized_json)
    
    # Load and validate TranscriptFlow checkpoint
    log_status(f"Loading TranscriptFlow checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint_result = load_transcript_flow_checkpoint(checkpoint_path)
    except CheckpointValidationError as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return 1
    
    # Print checkpoint summary
    print_transcript_flow_summary(checkpoint_result)
    
    if checkpoint_result.warnings:
        print("\n--- Validation Warnings ---")
        for warning in checkpoint_result.warnings:
            print(f"  {warning}")
    
    # Determine mode - VAD split audio is the primary mode for TranscriptFlow checkpoints
    mode = checkpoint_result.metadata.get('mode', 'vad_split_audio')
    if hasattr(args, 'mode') and args.mode:
        mode = args.mode
        log_progress(f"Using mode from command line: {mode}")
    else:
        log_progress(f"Using mode from checkpoint metadata: {mode}")
    
    args.mode = mode
    
    # Determine start stage based on de-identification setting
    enable_de_id = getattr(args, 'enable_de_identification', False) or \
                   getattr(args, 'de_identify', False)
    
    if enable_de_id:
        start_stage = "de_identification"
        log_progress("De-identification enabled - starting from de_identification stage")
    else:
        start_stage = "speaker_naming"
        log_progress("De-identification disabled - starting from speaker_naming stage")
    
    # Allow override via --start-stage argument
    if hasattr(args, 'start_stage') and args.start_stage:
        start_stage = args.start_stage
        log_progress(f"Using start stage from command line: {start_stage}")
    
    # Dry run - just show what would happen
    if args.dry_run:
        return run_transcript_flow_dry_run(args, checkpoint_result, start_stage)
    
    # Ensure output directory
    outdir = ensure_outdir(args.outdir)
    
    # Write settings to file if DEBUG log level is set
    if args.log_level == "DEBUG":
        settings_path = os.path.join(outdir, "settings.txt")
        with open(settings_path, 'w') as f:
            f.write("Local-Transcribe Settings (TranscriptFlow Re-entry Mode)\n")
            f.write("=" * 50 + "\n\n")
            
            # Write all command line arguments
            f.write("Command Line Arguments:\n")
            f.write("-" * 25 + "\n")
            for key, value in vars(args).items():
                if key not in ['audio_files', 'outdir', 'from_diarized_json']:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Write processing mode
            f.write("Processing Mode:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Mode: {mode} (TranscriptFlow Re-entry)\n")
            f.write(f"Starting Stage: {start_stage}\n")
            f.write(f"Selected Outputs: {', '.join(args.selected_outputs)}\n")
            f.write("\n")
            
            # Write checkpoint info
            f.write("Checkpoint Information:\n")
            f.write("-" * 23 + "\n")
            f.write(f"Checkpoint File: {os.path.basename(str(checkpoint_path))}\n")
            f.write(f"Number of Turns: {checkpoint_result.total_turns}\n")
            f.write(f"Number of Interjections: {checkpoint_result.total_interjections}\n")
        
        print(f"[DEBUG] Settings written to {settings_path}")
    
    # Set up directory structure for re-entry
    capabilities = {
        "mode": mode,
        "has_builtin_alignment": False,
        "aligner": False,
        "diarization": False
    }
    
    paths = ensure_session_dirs(outdir, mode, {}, capabilities)
    
    # Ensure turns directory exists
    turns_dir = paths["intermediate"] / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default outputs if not specified
    if not hasattr(args, 'selected_outputs') or not args.selected_outputs:
        if getattr(args, 'only_final_transcript', False):
            args.selected_outputs = ['timestamped-txt']
        else:
            from local_transcribe.framework.cli import get_available_writers
            available_writers = get_available_writers(mode, api["registry"], exclude_internal=True)
            args.selected_outputs = list(available_writers.keys())
    
    # Configure logging
    api["configure_global_logging"](log_level=args.log_level)
    
    # Build pipeline context with TranscriptFlow already loaded
    context = PipelineContext(
        args=args,
        api=api,
        root=root,
        paths=paths,
        mode=mode,
        transcript=checkpoint_result.transcript,  # TranscriptFlow is already loaded
        start_from_stage=start_stage,
        checkpoint_metadata=checkpoint_result.metadata,
        input_checkpoint_path=checkpoint_path
    )
    
    # Handle audio for video generation
    if hasattr(args, 'audio_for_video') and args.audio_for_video:
        audio_path = Path(args.audio_for_video)
        if audio_path.exists():
            context.standardized_audio = audio_path
        else:
            log_progress(f"Warning: Audio file not found: {audio_path}")
    
    # Get transcript cleanup provider if specified
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider:
        try:
            context.transcript_cleanup_provider = api["registry"].get_transcript_cleanup_provider(
                args.transcript_cleanup_provider
            )
        except ValueError:
            log_progress(f"Warning: Transcript cleanup provider not found: {args.transcript_cleanup_provider}")
    
    # Execute pipeline stages
    log_status(f"Starting pipeline from {start_stage} stage")
    
    stages = get_stages_for_reentry(start_stage)
    
    for stage in stages:
        stage_name = stage.name
        
        # Check if we should skip this stage
        if context.should_skip_stage(stage_name):
            log_progress(f"Skipping stage: {stage_name}")
            continue
        
        # Check if stage can execute
        can_run, reason = stage.can_execute(context)
        if not can_run:
            if stage_name in ("transcript_cleanup", "de_identification"):
                # These stages are optional
                log_progress(f"Skipping optional stage {stage_name}: {reason}")
                continue
            else:
                print(f"ERROR: Cannot execute stage {stage_name}: {reason}")
                return 1
        
        log_status(f"Executing stage: {stage_name}")
        
        try:
            context = stage.execute(context)
        except Exception as e:
            print(f"ERROR: Stage {stage_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Final summary
    log_completion("Pipeline re-entry from TranscriptFlow complete")
    print(f"[i] Artifacts written to: {paths['root']}")
    
    return 0


def run_transcript_flow_dry_run(args, checkpoint_result, start_stage: str) -> int:
    """
    Perform a dry run for TranscriptFlow re-entry.
    
    Args:
        args: Parsed command line arguments
        checkpoint_result: Loaded TranscriptFlow checkpoint result
        start_stage: The stage to start from
        
    Returns:
        Exit code (0 for success)
    """
    print("\n" + "=" * 60)
    print("DRY RUN (TranscriptFlow) - No changes will be made")
    print("=" * 60)
    
    mode = args.mode
    
    print(f"\n📁 Checkpoint: {args.from_diarized_json}")
    print(f"📂 Output directory: {args.outdir}")
    print(f"🔧 Mode: {mode}")
    print(f"🎬 Starting stage: {start_stage}")
    
    print("\n--- Stages that would execute ---")
    
    stages_to_run = get_stages_from(start_stage)
    stage_descriptions = get_stage_descriptions()
    
    for i, stage_name in enumerate(stages_to_run, 1):
        desc = stage_descriptions.get(stage_name, "")
        
        # Determine status
        if stage_name == "transcript_cleanup":
            has_provider = hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider
            status = "✓ Will run" if has_provider else "○ Skipped (no provider)"
        elif stage_name == "de_identification":
            enable_de_id = getattr(args, 'enable_de_identification', False) or \
                          getattr(args, 'de_identify', False)
            status = "✓ Will run" if enable_de_id else "○ Skipped (not enabled)"
        else:
            status = "✓ Will run"
        
        print(f"  {i}. [{status}] {stage_name}")
        print(f"      {desc}")
    
    print("\n--- Checkpoint information ---")
    print(f"  Total turns: {checkpoint_result.total_turns}")
    print(f"  Total interjections: {checkpoint_result.total_interjections}")
    print(f"  Duration: {checkpoint_result.duration_seconds:.1f} seconds")
    print(f"  Speakers: {', '.join(checkpoint_result.speakers_found)}")
    
    if checkpoint_result.warnings:
        print("\n--- Warnings ---")
        for warning in checkpoint_result.warnings:
            print(f"  {warning}")
    
    print("\n" + "=" * 60)
    print("Dry run complete. Use without --dry-run to execute.")
    print("=" * 60)
    
    return 0


def run_dry_run(args, checkpoint_result) -> int:
    """
    Perform a dry run - validate and show what would happen.
    
    Args:
        args: Parsed command line arguments
        checkpoint_result: Loaded checkpoint result
        
    Returns:
        Exit code (0 for success)
    """
    print("\n" + "=" * 60)
    print("DRY RUN - No changes will be made")
    print("=" * 60)
    
    mode = args.mode
    
    print(f"\n📁 Checkpoint: {args.from_diarized_json}")
    print(f"📂 Output directory: {args.outdir}")
    print(f"🔧 Mode: {mode}")
    
    print("\n--- Stages that would execute ---")
    
    stages_to_run = get_stages_from("turn_building")
    stage_descriptions = get_stage_descriptions()
    
    for i, stage_name in enumerate(stages_to_run, 1):
        desc = stage_descriptions.get(stage_name, "")
        
        # Determine status
        if stage_name == "transcript_cleanup":
            has_provider = hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider
            status = "✓ Will run" if has_provider else "○ Skipped (no provider)"
        else:
            status = "✓ Will run"
        
        print(f"  {i}. [{status}] {stage_name}")
        print(f"      {desc}")
    
    print("\n--- Checkpoint validation ---")
    print(f"  Total words: {checkpoint_result.total_words:,}")
    print(f"  Duration: {checkpoint_result.duration_seconds:.1f} seconds")
    print(f"  Speakers: {', '.join(checkpoint_result.speakers_found)}")
    
    if checkpoint_result.warnings:
        print("\n--- Warnings ---")
        for warning in checkpoint_result.warnings:
            print(f"  {warning}")
    
    print("\n" + "=" * 60)
    print("Dry run complete. Use without --dry-run to execute.")
    print("=" * 60)
    
    return 0


def check_reentry_requirements(args) -> tuple[bool, str]:
    """
    Check if re-entry requirements are met.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (requirements_met, error_message)
    """
    if not args.from_diarized_json:
        return False, "No checkpoint file specified"
    
    checkpoint_path = Path(args.from_diarized_json)
    if not checkpoint_path.exists():
        return False, f"Checkpoint file not found: {checkpoint_path}"
    
    if not args.outdir:
        return False, "Output directory (-o/--outdir) is required"
    
    return True, ""
