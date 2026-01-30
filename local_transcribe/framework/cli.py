#!/usr/bin/env python3
"""
CLI argument parsing and interactive prompts.

This module provides:
- Command-line argument parsing (parse_args)
- Pipeline stage listing (list_stages)
- Pipeline re-entry prompts (interactive_reentry_prompt)
- Main interactive prompt entry point (interactive_prompt)

The actual implementation details are split into submodules:
- cli_prompts.py: Reusable prompt helpers
- cli_providers.py: Provider selection helpers
- cli_modes.py: Mode-specific interactive flows
"""

import argparse
from typing import Optional

from local_transcribe.lib.environment import get_available_system_capabilities

# Import and re-export from submodules for backward compatibility
from local_transcribe.framework.cli_prompts import (
    prompt_selection,
    prompt_yes_no,
    prompt_url,
    print_mode_header,
    # Backward compatibility aliases
    _prompt_selection,
    _prompt_yes_no,
    _prompt_url,
    _print_mode_header,
)

from local_transcribe.framework.cli_providers import (
    select_provider,
    select_transcriber_provider,
    select_transcriber_model,
    select_aligner_provider,
    select_diarization_provider,
    configure_transcriber,
    configure_aligner_if_needed,
    configure_diarization,
    configure_num_speakers,
    prompt_remote_transcriber_url,
    # Backward compatibility aliases
    _select_provider,
    _configure_transcriber,
    _configure_aligner_if_needed,
    _configure_diarization,
    _configure_num_speakers,
)

from local_transcribe.framework.cli_modes import (
    PipelineMode,
    prompt_system_capability,
    prompt_de_identification,
    get_available_writers,
    filter_incompatible_writers,
    prompt_output_formats,
    prompt_transcript_cleanup,
    interactive_single_speaker,
    interactive_vad_split_audio,
    interactive_combined_audio,
    interactive_split_audio,
    display_configuration_summary,
)


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="local-transcribe: offline transcription."
    )
    p.add_argument("-i", "--interactive", action="store_true", help="Interactive mode: prompt for provider and output selections.")
    p.add_argument("-l", "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING", help="Set logging level (DEBUG, INFO, WARNING, ERROR) [Default: WARNING]")
  
    p.add_argument("-a", "--audio-files", nargs='+', metavar="AUDIO_FILE", help="Audio files to process. One file = mixed audio with multiple speakers. Multiple files = separate tracks (2 files: interviewer + participant, 3+ files: prompt for speaker names).")
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("-s", "--single-speaker-audio", action="store_true", help="Process a single speaker audio file for transcription only, output as CSV.")
    p.add_argument("-n", "--num-speakers", type=int, help="Number of speakers expected in the audio (for diarization) [Default: 2]")
    p.add_argument("-x", "--system", choices=["cuda", "mps", "cpu"], help="System capability to use for ML acceleration [Default: auto-detected preference: MPS > CUDA > CPU]")
    p.add_argument("-d", "--de-identify", action="store_true", help="Enable de-identification to replace people's names with [REDACTED]. Automatically runs two-pass processing for multi-speaker transcripts. [Default: prompt in interactive mode]")

    p.add_argument("--transcriber-provider", help="Transcriber provider to use [Default: auto-selected]")
    p.add_argument("--transcriber-model", help="Transcriber model to use (if provider supports multiple models) [Default: provider-specific]")
    p.add_argument("--aligner-provider", help="Aligner provider to use (required if transcriber doesn't have built-in alignment) [Default: auto-selected if needed]")
    p.add_argument("--diarization-provider", help="Diarization provider to use (required for single audio files with multiple speakers) [Default: auto-selected if needed]")
    p.add_argument("--transcript-cleanup-provider", help="Transcript cleanup provider to use for LLM-based transcript cleaning [Default: none]")
    
    p.add_argument("--llm-de-identifier-url", default="http://0.0.0.0:8080", help="URL for LLM personal information de-identifier processor (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    p.add_argument("--llm-transcript-cleanup-url", default="http://0.0.0.0:8080", help="URL for remote transcript cleanup provider (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    
    p.add_argument("--remote-transcriber-url", default="http://0.0.0.0:7070", help="URL for remote transcription server when using 'remote' transcriber provider [Default: http://0.0.0.0:7070]")

    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")

    # Pipeline re-entry arguments
    p.add_argument("--from-diarized-json", metavar="JSON_FILE", help="Resume pipeline from a corrected diarized word segments JSON file. Starts from turn-building stage.")
    p.add_argument("--audio-for-video", metavar="AUDIO_FILE", help="Original audio file path for video generation when resuming from checkpoint.")
    p.add_argument("--mode", choices=["combined_audio", "split_audio", "vad_split_audio"], help="Pipeline mode. Overrides mode detected from checkpoint metadata.")
    p.add_argument("--speaker-map", metavar="MAPPING", help="Speaker name mapping for re-entry (e.g., 'SPEAKER_00=Interviewer,SPEAKER_01=Participant')")
    p.add_argument("--dry-run", action="store_true", help="Validate checkpoint and show what stages would run without executing.")
    p.add_argument("--list-stages", action="store_true", help="List available pipeline stages and exit.")

    # VAD pipeline arguments
    p.add_argument("--vad-pipeline", action="store_true", help="Use VAD-first pipeline for split audio files.")

    # LLM cleanup arguments
    p.add_argument("--enable-cleanup", action="store_true", help="Enable LLM-based transcript cleanup stage (disabled by default). Requires --transcript-cleanup-provider to be set.")

    args = p.parse_args(argv)
    
    # Track whether certain arguments were explicitly provided via CLI
    # This is used to determine if we should prompt for values in interactive mode
    import sys
    raw_args = argv if argv is not None else sys.argv[1:]
    args._llm_de_identifier_url_set = any(
        arg.startswith('--llm-de-identifier-url') for arg in raw_args
    )

    return args


# =============================================================================
# Pipeline Stage Listing
# =============================================================================

def list_stages():
    """Display available pipeline stages."""
    from local_transcribe.framework.pipeline_context import get_stage_order, get_stage_descriptions
    
    print("\n=== Pipeline Stages ===")
    print("\nStages execute in the following order:\n")
    
    stages = get_stage_order()
    descriptions = get_stage_descriptions()
    
    for i, stage in enumerate(stages, 1):
        desc = descriptions.get(stage, "No description")
        print(f"  {i}. {stage}")
        print(f"     {desc}")
    
    print("\n--- Re-entry Points ---")
    print("\nCurrently supported re-entry points:")
    print("  • turn_building - Resume from corrected diarized JSON file")
    print("                    Use: --from-diarized-json <file>")
    
    print("\nExample usage:")
    print("  # Resume from corrected diarization with interactive prompts")
    print("  python main.py -o ./output --from-diarized-json ./corrected.json -i")
    print("")
    print("  # Dry run to validate checkpoint")
    print("  python main.py -o ./output --from-diarized-json ./corrected.json --dry-run")


# =============================================================================
# Pipeline Re-entry Prompts
# =============================================================================

def interactive_reentry_prompt(args, api, checkpoint_result):
    """
    Interactive prompts specific to pipeline re-entry.
    
    Only prompts for configuration needed from the re-entry point onward.
    
    Args:
        args: Parsed command line arguments
        api: Pipeline API dictionary
        checkpoint_result: Loaded checkpoint result
        
    Returns:
        Updated args namespace
    """
    from local_transcribe.lib.speaker_mapper import (
        create_speaker_mapping_interactive,
        detect_speakers_in_segments
    )
    
    registry = api["registry"]
    
    print("\n" + "=" * 60)
    print("PIPELINE RE-ENTRY - INTERACTIVE MODE")
    print("=" * 60)
    
    print(f"\nResuming from: {args.from_diarized_json}")
    print(f"Output directory: {args.outdir}")
    
    # 1. Speaker name assignment
    print("\n" + "-" * 40)
    print("STEP 1: Speaker Name Assignment")
    print("-" * 40)
    
    response = input("\nWould you like to assign names to speakers? [Y/n]: ").strip().lower()
    if response != 'n':
        mode = getattr(args, 'mode', None) or checkpoint_result.metadata.get('mode', 'combined_audio')
        speaker_mapping = create_speaker_mapping_interactive(
            checkpoint_result.segments,
            mode,
            show_samples=True
        )
        args.speaker_mapping = speaker_mapping
    else:
        args.speaker_mapping = {}
        print("  ✓ Keeping original speaker IDs")
    
    # 2. Output format selection
    print("\n" + "-" * 40)
    print("STEP 2: Output Format Selection")
    print("-" * 40)
    
    # Get mode-compatible writers (mode was determined from checkpoint metadata or CLI)
    mode = getattr(args, 'mode', None) or checkpoint_result.metadata.get('mode', 'combined_audio')
    filtered_writers = get_available_writers(mode, registry, exclude_internal=True)
    
    print("\nAvailable Output Formats:")
    for i, (name, desc) in enumerate(filtered_writers.items(), 1):
        print(f"  {i}. {name}: {desc}")
    
    print("\nEnter numbers separated by commas (e.g., 1,3,5), or press Enter for all formats [Default: all]:")
    choice = input("Select output formats: ").strip()
    
    if not choice:
        args.selected_outputs = list(filtered_writers.keys())
        print("  ✓ Selected: All output formats [Default]")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(filtered_writers)]
            args.selected_outputs = [list(filtered_writers.keys())[i] for i in valid_indices]
            if not args.selected_outputs:
                print("  Error: No valid choices, selecting all.")
                args.selected_outputs = list(filtered_writers.keys())
            else:
                print(f"  ✓ Selected: {', '.join(args.selected_outputs)}")
        except ValueError:
            print("  Error: Invalid input, selecting all.")
            args.selected_outputs = list(filtered_writers.keys())
    
    # 3. Video generation (if video is in selected outputs)
    if 'video' in args.selected_outputs:
        print("\n" + "-" * 40)
        print("STEP 3: Video Generation")
        print("-" * 40)
        
        if not args.audio_for_video:
            print("\nVideo output requires the original audio file.")
            audio_path = input("Enter path to audio file (or press Enter to skip video): ").strip()
            if audio_path:
                args.audio_for_video = audio_path
                print(f"  ✓ Audio for video: {audio_path}")
            else:
                # Remove video from outputs
                args.selected_outputs = [o for o in args.selected_outputs if o != 'video']
                print("  ✓ Video output skipped")
        else:
            print(f"  ✓ Using audio file: {args.audio_for_video}")
    
    # 4. Transcript cleanup (optional)
    print("\n" + "-" * 40)
    print("STEP 4: Transcript Cleanup (Optional)")
    print("-" * 40)
    
    transcript_cleanup_providers = registry.list_transcript_cleanup_providers()
    if transcript_cleanup_providers:
        print("\nTranscript Cleanup Providers (optional LLM-based transcript cleaning):")
        print("  0. None (skip cleanup) [Default]")
        for i, (name, desc) in enumerate(transcript_cleanup_providers.items(), 1):
            print(f"  {i}. {name}: {desc}")
        
        while True:
            try:
                choice_input = input("\nSelect transcript cleanup provider (number) [Default: 0]: ").strip()
                
                if not choice_input:
                    choice = 0
                else:
                    choice = int(choice_input)
                
                if choice == 0:
                    args.transcript_cleanup_provider = None
                    print("  ✓ Selected: None [Default]")
                    break
                elif 1 <= choice <= len(transcript_cleanup_providers):
                    args.transcript_cleanup_provider = list(transcript_cleanup_providers.keys())[choice - 1]
                    
                    # If remote provider, ask for URL
                    if args.transcript_cleanup_provider == "llm_transcript_cleanup":
                        default_url = getattr(args, 'llm_transcript_cleanup_url', 'http://0.0.0.0:8080')
                        url = input(f"Enter LLM server URL [Default: {default_url}]: ").strip()
                        if url:
                            if not url.startswith(('http://', 'https://')):
                                url = f"http://{url}"
                            args.llm_transcript_cleanup_url = url
                        else:
                            args.llm_transcript_cleanup_url = default_url
                    
                    print(f"  ✓ Selected: {args.transcript_cleanup_provider}")
                    break
                else:
                    print("  Error: Please enter a number from the list.")
            except ValueError:
                print("  Error: Please enter a valid number.")
    else:
        args.transcript_cleanup_provider = None
        print("  No transcript cleanup providers available.")
    
    # Summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\n  Checkpoint: {args.from_diarized_json}")
    print(f"  Output directory: {args.outdir}")
    print(f"  Mode: {getattr(args, 'mode', 'auto-detected')}")
    print(f"  Output formats: {', '.join(args.selected_outputs)}")
    if args.audio_for_video:
        print(f"  Audio for video: {args.audio_for_video}")
    if args.transcript_cleanup_provider:
        print(f"  Transcript cleanup: {args.transcript_cleanup_provider}")
    if hasattr(args, 'speaker_mapping') and args.speaker_mapping:
        print(f"  Speaker mapping: {len(args.speaker_mapping)} speakers renamed")
    
    print("\n" + "=" * 60)
    
    return args


# =============================================================================
# Pipeline Mode Determination
# =============================================================================

def determine_pipeline_mode(args) -> str:
    """
    Determine the pipeline mode based on CLI arguments.
    
    Returns:
        str: One of PipelineMode constants
    """
    # Single speaker mode takes precedence
    if getattr(args, 'single_speaker_audio', False):
        return PipelineMode.SINGLE_SPEAKER
    
    num_files = len(args.audio_files) if hasattr(args, 'audio_files') and args.audio_files else 0
    
    if num_files == 0:
        return PipelineMode.COMBINED_AUDIO  # Will error later, but need a default
    elif num_files == 1:
        return PipelineMode.COMBINED_AUDIO
    else:
        # Multiple files - check for VAD pipeline flag
        if getattr(args, 'vad_pipeline', False):
            return PipelineMode.VAD_SPLIT_AUDIO
        return PipelineMode.SPLIT_AUDIO


def apply_cli_implications(args) -> argparse.Namespace:
    """
    Apply logical implications from CLI arguments.
    """
    # No special implications needed anymore - remote is just another transcriber provider
    return args


# =============================================================================
# Main Interactive Prompt Entry Point
# =============================================================================

def interactive_prompt(args, api):
    """
    Main interactive prompt entry point.
    
    Routes to mode-specific interactive flows based on CLI arguments
    and audio file configuration.
    """
    registry = api["registry"]
    
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    
    # Apply CLI implications (e.g., --remote-granite implies granite transcriber)
    args = apply_cli_implications(args)
    
    # Determine pipeline mode
    mode = determine_pipeline_mode(args)
    
    # Show what was already configured via CLI
    print("\nCLI Configuration Detected:")
    cli_items = []
    if args.transcriber_provider:
        cli_items.append(f"Transcriber: {args.transcriber_provider}")
    if args.transcriber_model:
        cli_items.append(f"Model: {args.transcriber_model}")
    if getattr(args, 'vad_pipeline', False):
        cli_items.append("VAD Pipeline: Enabled")
    if args.aligner_provider:
        cli_items.append(f"Aligner: {args.aligner_provider}")
    if args.diarization_provider:
        cli_items.append(f"Diarization: {args.diarization_provider}")
    if getattr(args, 'de_identify', False):
        cli_items.append("De-identification: Enabled")
    if args.system:
        cli_items.append(f"System: {args.system.upper()}")
    
    if cli_items:
        for item in cli_items:
            print(f"  • {item}")
    else:
        print("  (none)")
    
    # Route to mode-specific interactive flow
    if mode == PipelineMode.SINGLE_SPEAKER:
        args = interactive_single_speaker(args, api)
    elif mode == PipelineMode.VAD_SPLIT_AUDIO:
        args = interactive_vad_split_audio(args, api)
    elif mode == PipelineMode.COMBINED_AUDIO:
        args = interactive_combined_audio(args, api)
    else:  # SPLIT_AUDIO
        args = interactive_split_audio(args, api)
    
    # Display final configuration summary
    display_configuration_summary(args, mode)
    
    return args
