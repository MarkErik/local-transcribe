#!/usr/bin/env python3
"""
Pipeline re-entry runner for resuming from checkpoints.

This module provides functionality to resume the transcription pipeline
from intermediate checkpoint files (e.g., corrected diarized JSON).
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
    CheckpointValidationError
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
    
    This function:
    1. Loads and validates the checkpoint
    2. Determines the mode (from checkpoint or args)
    3. Sets up output directories
    4. Runs remaining pipeline stages
    
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
    log_status(f"Loading checkpoint from: {checkpoint_path}")
    
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
            all_writers = list(api["registry"].list_output_writers().keys())
            args.selected_outputs = all_writers
    
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
