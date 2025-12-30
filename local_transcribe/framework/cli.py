#!/usr/bin/env python3
# framework/cli.py - CLI argument parsing and interactive prompts

import argparse
from typing import Optional

from local_transcribe.lib.environment import get_available_system_capabilities

# ---------- CLI ----------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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
    
    # Remote transcription server URL (used when selecting 'remote' transcriber)
    p.add_argument("--remote-transcriber-url", default="http://0.0.0.0:7070", help="URL for remote transcription server when using 'remote' transcriber provider [Default: http://0.0.0.0:7070]")

    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--show-defaults", action="store_true", help="Show all default values and exit.")

    # Pipeline re-entry arguments
    p.add_argument("--from-diarized-json", metavar="JSON_FILE", help="Resume pipeline from a corrected diarized word segments JSON file. Starts from turn-building stage.")
    p.add_argument("--audio-for-video", metavar="AUDIO_FILE", help="Original audio file path for video generation when resuming from checkpoint.")
    p.add_argument("--mode", choices=["combined_audio", "split_audio", "vad_split_audio"], help="Pipeline mode. Overrides mode detected from checkpoint metadata.")
    p.add_argument("--speaker-map", metavar="MAPPING", help="Speaker name mapping for re-entry (e.g., 'SPEAKER_00=Interviewer,SPEAKER_01=Participant')")
    p.add_argument("--dry-run", action="store_true", help="Validate checkpoint and show what stages would run without executing.")
    p.add_argument("--list-stages", action="store_true", help="List available pipeline stages and exit.")

    # VAD pipeline arguments
    p.add_argument("--vad-pipeline", action="store_true", help="Use VAD-first pipeline for split audio files (recommended for interviews).")
    p.add_argument("--skip-alignment", action="store_true", default=True, help="Skip word-level alignment (default: True for VAD pipeline).")
    p.add_argument("--vad-threshold", type=float, default=0.5, help="VAD speech probability threshold (0-1) [Default: 0.5]")
    p.add_argument("--vad-merge-gap-ms", type=int, default=600, help="Maximum gap between VAD segments to merge (ms) [Default: 600]")

    args = p.parse_args(argv)

    return args

def show_defaults():
    """Display all default values used by the application."""
    print("\n=== Default Values ===")
    print("\nSystem Capability:")
    print("  - Default: Auto-detected preference (MPS > CUDA > CPU)")
    
    print("\nProviders:")
    print("  - Transcriber Provider: Auto-selected based on availability")
    print("  - Transcriber Model: Provider-specific default model")
    print("  - Aligner Provider: Auto-selected if needed based on transcriber")
    print("  - Diarization Provider: Auto-selected if needed for single audio files")
    print("  - Transcript Cleanup Provider: None (disabled)")
    
    print("\nConfiguration:")
    print("  - Number of Speakers: 2")
    print("  - Output Formats: All available formats")
    print("  - Single Speaker Audio: Disabled (use -s to enable)")
    print("  - Chunking (Granite): Always enabled with local stitching")
    
    print("\nVAD Pipeline Settings:")
    print("  - VAD Pipeline: Disabled (use --vad-pipeline for split audio)")
    print("  - VAD Threshold: 0.5 (speech probability)")
    print("  - VAD Merge Gap: 500ms (gap threshold for merging segments)")
    print("  - Skip Alignment: True (word alignment skipped in VAD mode)")
    
    print("\nURLs:")
    print("  - LLM Turn Builder URL: http://0.0.0.0:8080")
    print("  - LLM Transcript Cleanup URL: http://0.0.0.0:8080")
    print("  - Remote Transcriber URL: http://0.0.0.0:7070")
    
    print("\nNote: Some defaults may be overridden by system capabilities or provider availability.")


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
    
    output_writers = registry.list_output_writers()
    # Filter out SRT as it's handled internally by video
    filtered_writers = {name: desc for name, desc in output_writers.items()
                       if name not in ['srt']}
    
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

class PipelineMode:
    """Enumeration of pipeline processing modes."""
    SINGLE_SPEAKER = "single_speaker_audio"
    COMBINED_AUDIO = "combined_audio"
    SPLIT_AUDIO = "split_audio"
    VAD_SPLIT_AUDIO = "vad_split_audio"


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
# Reusable Prompt Helpers
# =============================================================================

def _prompt_selection(options: list, prompt_text: str, default_index: Optional[int] = None, 
                      allow_none: bool = False, none_label: str = "None") -> int:
    """
    Generic selection prompt with consistent default handling.
    
    Args:
        options: List of (name, display_name) tuples
        prompt_text: Text to display for the prompt
        default_index: 0-based index of default option (None if no default, -1 for None option)
        allow_none: If True, option 0 is "None/skip"
        none_label: Label for the none option
        
    Returns:
        Selected index (0-based), or -1 if none selected
    """
    if allow_none:
        default_marker = " [Default]" if default_index == -1 else ""
        print(f"  0. {none_label}{default_marker}")
    
    for i, (name, display_name) in enumerate(options):
        is_default = (default_index == i)
        marker = " [Default]" if is_default else ""
        # When allow_none, options start at 1; otherwise at 1 as well
        display_num = i + 1
        print(f"  {display_num}. {display_name}{marker}")
    
    # Build prompt with default info
    if default_index is not None:
        if default_index == -1 and allow_none:
            default_display = "0"
        else:
            default_display = str(default_index + 1)
        full_prompt = f"\n{prompt_text} [Default: {default_display}]: "
    else:
        full_prompt = f"\n{prompt_text}: "
    
    while True:
        choice_input = input(full_prompt).strip()
        
        # Handle default (Enter key)
        if not choice_input:
            if default_index is not None:
                return default_index
            else:
                print("  Error: A selection is required.")
                continue
        
        try:
            choice = int(choice_input)
            
            if allow_none and choice == 0:
                return -1
            
            # Convert 1-based input to 0-based index
            adjusted = choice - 1
            if 0 <= adjusted < len(options):
                return adjusted
            else:
                print("  Error: Please enter a number from the list.")
        except ValueError:
            print("  Error: Please enter a valid number.")


def _prompt_yes_no(prompt_text: str, default: bool = True) -> bool:
    """
    Yes/No prompt with consistent default handling.
    
    Args:
        prompt_text: Text to display
        default: Default value (True=Yes, False=No)
        
    Returns:
        bool: User's choice
    """
    default_hint = "Y/n" if default else "y/N"
    full_prompt = f"{prompt_text} [{default_hint}]: "
    
    while True:
        response = input(full_prompt).strip().lower()
        
        if not response:
            return default
        elif response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        else:
            print("  Error: Please enter 'y' or 'n'.")


def _prompt_url(prompt_text: str, default_url: str) -> str:
    """
    URL input prompt with validation.
    
    Args:
        prompt_text: Text to display
        default_url: Default URL value
        
    Returns:
        str: Validated URL
    """
    full_prompt = f"{prompt_text} [Default: {default_url}]: "
    url = input(full_prompt).strip()
    
    if not url:
        return default_url
    
    # Add http:// if not present
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    
    return url


# =============================================================================
# Provider Selection Helpers
# =============================================================================

def select_transcriber_provider(registry, filter_pure_only: bool = False, default_provider: Optional[str] = None):
    """
    Select a transcriber provider with optional filtering.
    
    Args:
        registry: Plugin registry
        filter_pure_only: If True, only show providers without built-in alignment
        default_provider: Name of default provider (if any)
        
    Returns:
        str: Selected provider name
    """
    providers = registry.list_transcriber_providers()
    
    # Apply filter if needed
    if filter_pure_only:
        filtered = {}
        for name, desc in providers.items():
            provider = registry.get_transcriber_provider(name)
            if not provider.has_builtin_alignment:
                filtered[name] = desc
        providers = filtered
    
    if not providers:
        raise ValueError("No suitable transcriber providers available.")
    
    # Build options list with display names
    options = []
    default_index = None
    for i, (name, desc) in enumerate(providers.items()):
        provider = registry.get_transcriber_provider(name)
        display_name = getattr(provider, 'short_name', desc)
        options.append((name, display_name))
        if name == default_provider:
            default_index = i
    
    print("\nAvailable Transcriber Providers:")
    selected = _prompt_selection(options, "Select transcriber (number)", default_index)
    
    return options[selected][0]


def select_transcriber_model(registry, provider_name: str, default_model: Optional[str] = None) -> Optional[str]:
    """
    Select a model for the given transcriber provider.
    
    Args:
        registry: Plugin registry
        provider_name: Name of transcriber provider
        default_model: Default model name (if any)
        
    Returns:
        str: Selected model name, or None if only one model
    """
    provider = registry.get_transcriber_provider(provider_name)
    available_models = provider.get_available_models()
    
    if not available_models:
        return None
    
    if len(available_models) == 1:
        print(f"  ✓ Using model: {available_models[0]}")
        return available_models[0]
    
    # Find default index
    default_index = None
    
    # For granite, default to 8b
    if provider_name == "granite" and default_model is None:
        default_model = "granite-8b"
    
    if default_model and default_model in available_models:
        default_index = available_models.index(default_model)
    elif default_index is None:
        default_index = 0  # First model as default
    
    options = [(m, m) for m in available_models]
    
    print(f"\nAvailable models for {getattr(provider, 'short_name', provider_name)}:")
    selected = _prompt_selection(options, "Select model (number)", default_index)
    
    return available_models[selected]


def select_aligner_provider(registry, default_provider: Optional[str] = None):
    """Select an aligner provider."""
    providers = registry.list_aligner_providers()
    
    if not providers:
        raise ValueError("No aligner providers available.")
    
    options = []
    default_index = None
    for i, (name, desc) in enumerate(providers.items()):
        provider = registry.get_aligner_provider(name)
        display_name = getattr(provider, 'short_name', desc)
        options.append((name, display_name))
        if name == default_provider:
            default_index = i
    
    # Default to first if not specified
    if default_index is None:
        default_index = 0
    
    print("\nAvailable Aligner Providers:")
    selected = _prompt_selection(options, "Select aligner (number)", default_index)
    
    return options[selected][0]


def select_diarization_provider(registry, default_provider: Optional[str] = None):
    """Select a diarization provider."""
    providers = registry.list_diarization_providers()
    
    if not providers:
        raise ValueError("No diarization providers available.")
    
    options = []
    default_index = None
    for i, (name, desc) in enumerate(providers.items()):
        provider = registry.get_diarization_provider(name)
        display_name = getattr(provider, 'short_name', desc)
        options.append((name, display_name))
        if name == default_provider:
            default_index = i
    
    # Default to first if not specified
    if default_index is None:
        default_index = 0
    
    print("\nAvailable Diarization Providers:")
    selected = _prompt_selection(options, "Select diarization provider (number)", default_index)
    
    return options[selected][0]


# =============================================================================
# Common Prompt Sections
# =============================================================================

def prompt_system_capability(args) -> argparse.Namespace:
    """Prompt for system capability (MPS/CUDA/CPU) if not already set."""
    if args.system:
        print(f"  ✓ System: {args.system.upper()} (set via --system)")
        return args
    
    available_capabilities = get_available_system_capabilities()
    
    # Determine preferred default: MPS > CUDA > CPU
    if "mps" in available_capabilities:
        default_capability = "mps"
    elif "cuda" in available_capabilities:
        default_capability = "cuda"
    else:
        default_capability = "cpu"
    
    if len(available_capabilities) == 1:
        args.system = available_capabilities[0]
        print(f"  ✓ System: {args.system.upper()} (only option)")
        return args
    
    default_index = available_capabilities.index(default_capability)
    options = [(cap, cap.upper()) for cap in available_capabilities]
    
    print("\nSystem Capability:")
    selected = _prompt_selection(options, "Select system capability (number)", default_index)
    args.system = available_capabilities[selected]
    print(f"  ✓ System: {args.system.upper()}")
    
    return args


def prompt_remote_transcriber_url(args) -> argparse.Namespace:
    """Prompt for remote transcriber server URL if remote transcriber is selected."""
    # Only relevant when remote transcriber is selected
    if args.transcriber_provider != "remote":
        return args
    
    print("\n--- Remote Transcription Server ---")
    
    default_url = getattr(args, 'remote_transcriber_url', 'http://0.0.0.0:7070')
    args.remote_transcriber_url = _prompt_url("Enter remote transcription server URL", default_url)
    
    # Check server availability
    from local_transcribe.providers.transcribers.remote_transcriber import check_remote_transcriber_available
    print(f"  Checking connection to {args.remote_transcriber_url}...")
    
    if check_remote_transcriber_available(args.remote_transcriber_url):
        print(f"  ✓ Remote server is available")
    else:
        print(f"  ⚠ Remote server not available at {args.remote_transcriber_url}")
        fallback = _prompt_yes_no("Continue anyway (will fail if server unavailable)?", default=False)
        if not fallback:
            # User wants to choose a different transcriber
            print("  ✓ Please select a different transcriber")
            args.transcriber_provider = None  # Reset to force re-selection
    
    return args


def prompt_de_identification(args, mode: str) -> argparse.Namespace:
    """Prompt for de-identification settings if not already set.
    
    De-identification now automatically runs two-pass processing when enabled
    for multi-speaker transcripts, so there's no need to prompt separately.
    """
    # De-identification not supported in VAD mode yet
    if mode == PipelineMode.VAD_SPLIT_AUDIO:
        print("\n  ⚠ Note: De-identification not yet implemented for VAD pipeline")
        args.de_identify = False
        return args
    
    # Prompt for de-identification if not already set via CLI
    if not args.de_identify:
        args.de_identify = _prompt_yes_no(
            "\nEnable de-identification (replace names with [REDACTED])?",
            default=True
        )
        if args.de_identify:
            if mode == PipelineMode.SINGLE_SPEAKER:
                print("  ✓ De-identification enabled")
            else:
                print("  ✓ De-identification enabled (includes two-pass processing)")
        else:
            print("  ✓ De-identification disabled")
    else:
        if mode == PipelineMode.SINGLE_SPEAKER:
            print("  ✓ De-identification enabled (set via --de-identify)")
        else:
            print("  ✓ De-identification enabled (set via --de-identify, includes two-pass processing)")
    
    return args


def prompt_output_formats(args, registry) -> argparse.Namespace:
    """Prompt for output format selection if not already set."""
    if hasattr(args, 'selected_outputs') and args.selected_outputs:
        print(f"  ✓ Output formats: {', '.join(args.selected_outputs)} (pre-configured)")
        return args
    
    # Filter out SRT as it's handled internally by video
    output_writers = registry.list_output_writers()
    filtered_writers = {name: desc for name, desc in output_writers.items()
                       if name not in ['srt']}
    
    print("\nAvailable Output Formats:")
    for i, (name, desc) in enumerate(filtered_writers.items(), 1):
        print(f"  {i}. {name}: {desc}")
    
    print("\n  Enter numbers separated by commas (e.g., 1,3,5), or press Enter for all formats")
    choice = input("  Select output formats [Default: all]: ").strip()
    
    if not choice:
        args.selected_outputs = list(filtered_writers.keys())
        print("  ✓ Selected: All output formats")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(filtered_writers)]
            args.selected_outputs = [list(filtered_writers.keys())[i] for i in valid_indices]
            if not args.selected_outputs:
                print("  Error: No valid choices, selecting all.")
                args.selected_outputs = list(filtered_writers.keys())
            print(f"  ✓ Selected: {', '.join(args.selected_outputs)}")
        except ValueError:
            print("  Error: Invalid input, selecting all.")
            args.selected_outputs = list(filtered_writers.keys())
            print(f"  ✓ Selected: {', '.join(args.selected_outputs)}")
    
    return args


def prompt_transcript_cleanup(args, registry) -> argparse.Namespace:
    """Prompt for optional transcript cleanup provider."""
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider is not None:
        if args.transcript_cleanup_provider:
            print(f"  ✓ Transcript cleanup: {args.transcript_cleanup_provider} (set via CLI)")
        else:
            print("  ✓ Transcript cleanup: None (set via CLI)")
        return args
    
    providers = registry.list_transcript_cleanup_providers()
    
    if not providers:
        args.transcript_cleanup_provider = None
        print("  ✓ No transcript cleanup providers available")
        return args
    
    print("\nTranscript Cleanup (optional LLM-based cleaning):")
    
    options = [(name, f"{name}: {desc}") for name, desc in providers.items()]
    
    # Default is None (index -1)
    selected = _prompt_selection(
        options, 
        "Select transcript cleanup provider (number)", 
        default_index=-1,
        allow_none=True,
        none_label="None (skip cleanup)"
    )
    
    if selected == -1:
        args.transcript_cleanup_provider = None
        print("  ✓ Transcript cleanup: None")
    else:
        args.transcript_cleanup_provider = options[selected][0]
        
        # If remote provider, ask for URL
        if args.transcript_cleanup_provider == "llm_transcript_cleanup":
            default_url = getattr(args, 'llm_transcript_cleanup_url', 'http://0.0.0.0:8080')
            args.llm_transcript_cleanup_url = _prompt_url(
                "Enter LLM server URL",
                default_url
            )
        
        print(f"  ✓ Transcript cleanup: {args.transcript_cleanup_provider}")
    
    return args


# =============================================================================
# Mode-Specific Interactive Flows
# =============================================================================

def interactive_single_speaker(args, api) -> argparse.Namespace:
    """Interactive prompts for single speaker audio mode."""
    registry = api["registry"]
    
    print("\n" + "-" * 50)
    print("MODE: Single Speaker Audio")
    print("Transcription only, output as CSV")
    print("-" * 50)
    
    # System capability
    args = prompt_system_capability(args)
    
    # Transcriber selection (pure transcribers only)
    if args.transcriber_provider is None:
        args.transcriber_provider = select_transcriber_provider(
            registry, 
            filter_pure_only=True,
            default_provider="granite"
        )
        print(f"  ✓ Transcriber: {args.transcriber_provider}")
    else:
        # Validate it's a pure transcriber
        provider = registry.get_transcriber_provider(args.transcriber_provider)
        if provider.has_builtin_alignment:
            print(f"  ⚠ Provider '{args.transcriber_provider}' has built-in alignment.")
            print("    Single speaker mode requires a pure transcriber (granite or openai_whisper).")
            args.transcriber_provider = select_transcriber_provider(
                registry,
                filter_pure_only=True,
                default_provider="granite"
            )
        else:
            print(f"  ✓ Transcriber: {args.transcriber_provider} (set via CLI)")
    
    # Model selection
    if args.transcriber_model is None:
        args.transcriber_model = select_transcriber_model(registry, args.transcriber_provider)
    else:
        print(f"  ✓ Model: {args.transcriber_model} (set via CLI)")
    
    # Remote transcriber URL prompt (if remote transcriber selected)
    args = prompt_remote_transcriber_url(args)
    
    # Granite-specific settings
    if args.transcriber_provider == "granite":
        args.output_format = "chunked"
        print("  ✓ Using chunk stitching for Granite")
    
    # De-identification
    args = prompt_de_identification(args, PipelineMode.SINGLE_SPEAKER)
    
    # Output is fixed to CSV for single speaker
    args.selected_outputs = ['csv']
    print("  ✓ Output format: CSV (fixed for single speaker mode)")
    
    return args


def interactive_vad_split_audio(args, api) -> argparse.Namespace:
    """Interactive prompts for VAD-first pipeline with split audio."""
    registry = api["registry"]
    
    print("\n" + "-" * 50)
    print("MODE: VAD Pipeline (Split Audio)")
    print("Using Voice Activity Detection for turn segmentation")
    print("Requires: Transcriber only (no aligner or diarization)")
    print("-" * 50)
    
    # System capability
    args = prompt_system_capability(args)
    
    # Transcriber selection
    if args.transcriber_provider is None:
        args.transcriber_provider = select_transcriber_provider(
            registry,
            default_provider="granite"
        )
        print(f"  ✓ Transcriber: {args.transcriber_provider}")
    else:
        print(f"  ✓ Transcriber: {args.transcriber_provider} (set via CLI)")
    
    # Model selection
    if args.transcriber_model is None:
        args.transcriber_model = select_transcriber_model(registry, args.transcriber_provider)
    else:
        print(f"  ✓ Model: {args.transcriber_model} (set via CLI)")
    
    # Remote transcriber URL prompt (if remote transcriber selected)
    args = prompt_remote_transcriber_url(args)
    
    # Granite-specific settings
    if args.transcriber_provider == "granite":
        args.output_format = "chunked"
        print("  ✓ Using chunk stitching for Granite")
    
    # VAD settings - show current values, allow modification
    print("\n--- VAD Settings ---")
    current_threshold = getattr(args, 'vad_threshold', 0.5)
    current_merge_gap = getattr(args, 'vad_merge_gap_ms', 600)
    
    print(f"  Current VAD threshold: {current_threshold}")
    print(f"  Current merge gap: {current_merge_gap}ms")
    
    modify_vad = _prompt_yes_no("Modify VAD settings?", default=False)
    
    if modify_vad:
        # VAD threshold
        threshold_input = input(f"  VAD threshold (0-1) [Default: {current_threshold}]: ").strip()
        if threshold_input:
            try:
                args.vad_threshold = float(threshold_input)
                if not 0 <= args.vad_threshold <= 1:
                    print("  ⚠ Invalid threshold, using default")
                    args.vad_threshold = current_threshold
            except ValueError:
                print("  ⚠ Invalid input, using default")
                args.vad_threshold = current_threshold
        
        # Merge gap
        gap_input = input(f"  Merge gap (ms) [Default: {current_merge_gap}]: ").strip()
        if gap_input:
            try:
                args.vad_merge_gap_ms = int(gap_input)
            except ValueError:
                print("  ⚠ Invalid input, using default")
                args.vad_merge_gap_ms = current_merge_gap
    
    print(f"  ✓ VAD threshold: {args.vad_threshold}")
    print(f"  ✓ VAD merge gap: {args.vad_merge_gap_ms}ms")
    
    # De-identification note
    args = prompt_de_identification(args, PipelineMode.VAD_SPLIT_AUDIO)
    
    # Output formats
    args = prompt_output_formats(args, registry)
    
    # Transcript cleanup
    args = prompt_transcript_cleanup(args, registry)
    
    return args


def interactive_combined_audio(args, api) -> argparse.Namespace:
    """Interactive prompts for combined audio (single file, multiple speakers)."""
    registry = api["registry"]
    
    print("\n" + "-" * 50)
    print("MODE: Combined Audio")
    print("Single audio file with multiple speakers")
    print("Requires: Transcriber, Aligner (if needed), Diarization")
    print("-" * 50)
    
    # System capability
    args = prompt_system_capability(args)
    
    # Transcriber selection
    if args.transcriber_provider is None:
        args.transcriber_provider = select_transcriber_provider(
            registry,
            default_provider="granite_mfa"
        )
        print(f"  ✓ Transcriber: {args.transcriber_provider}")
    else:
        print(f"  ✓ Transcriber: {args.transcriber_provider} (set via CLI)")
    
    # Model selection
    if args.transcriber_model is None:
        args.transcriber_model = select_transcriber_model(registry, args.transcriber_provider)
    else:
        print(f"  ✓ Model: {args.transcriber_model} (set via CLI)")
    
    # Remote transcriber URL prompt (if remote transcriber selected)
    args = prompt_remote_transcriber_url(args)
    
    # Check if aligner is needed
    transcriber = registry.get_transcriber_provider(args.transcriber_provider)
    if not transcriber.has_builtin_alignment:
        if args.aligner_provider is None:
            args.aligner_provider = select_aligner_provider(registry)
            print(f"  ✓ Aligner: {args.aligner_provider}")
        else:
            print(f"  ✓ Aligner: {args.aligner_provider} (set via CLI)")
    else:
        print("  ✓ Aligner: Not needed (transcriber has built-in alignment)")
        args.aligner_provider = None
    
    # Diarization (always needed for combined audio)
    if args.diarization_provider is None:
        args.diarization_provider = select_diarization_provider(registry)
        print(f"  ✓ Diarization: {args.diarization_provider}")
    else:
        print(f"  ✓ Diarization: {args.diarization_provider} (set via CLI)")
    
    # Number of speakers
    if not hasattr(args, 'num_speakers') or args.num_speakers is None:
        print("\n--- Speaker Configuration ---")
        while True:
            num_input = input("  Number of speakers expected [Default: 2]: ").strip()
            if not num_input:
                args.num_speakers = 2
                break
            try:
                num = int(num_input)
                if num > 0:
                    args.num_speakers = num
                    break
                else:
                    print("  Error: Please enter a positive number.")
            except ValueError:
                print("  Error: Please enter a valid number.")
        print(f"  ✓ Number of speakers: {args.num_speakers}")
    else:
        print(f"  ✓ Number of speakers: {args.num_speakers} (set via --num-speakers)")
    
    # Granite-specific settings
    if args.transcriber_provider and 'granite' in args.transcriber_provider:
        args.output_format = "chunked"
        print("  ✓ Using chunk stitching for Granite")
    
    # De-identification
    args = prompt_de_identification(args, PipelineMode.COMBINED_AUDIO)
    
    # Output formats
    args = prompt_output_formats(args, registry)
    
    # Transcript cleanup
    args = prompt_transcript_cleanup(args, registry)
    
    return args


def interactive_split_audio(args, api) -> argparse.Namespace:
    """Interactive prompts for split audio (separate files per speaker)."""
    registry = api["registry"]
    
    print("\n" + "-" * 50)
    print("MODE: Split Audio")
    print("Separate audio files per speaker")
    print("Requires: Transcriber, Aligner (if needed)")
    print("-" * 50)
    
    # Offer VAD pipeline as an option
    print("\n--- Pipeline Selection ---")
    print("  The VAD pipeline is recommended for interviews with separate audio tracks.")
    print("  It uses Voice Activity Detection for more accurate turn segmentation.")
    
    use_vad = _prompt_yes_no("Use VAD pipeline? (recommended)", default=True)
    
    if use_vad:
        args.vad_pipeline = True
        return interactive_vad_split_audio(args, api)
    
    # Continue with standard split audio flow
    args.vad_pipeline = False
    print("\n  Continuing with standard split audio pipeline...")
    
    # System capability
    args = prompt_system_capability(args)
    
    # Transcriber selection
    if args.transcriber_provider is None:
        args.transcriber_provider = select_transcriber_provider(
            registry,
            default_provider="granite_mfa"
        )
        print(f"  ✓ Transcriber: {args.transcriber_provider}")
    else:
        print(f"  ✓ Transcriber: {args.transcriber_provider} (set via CLI)")
    
    # Model selection
    if args.transcriber_model is None:
        args.transcriber_model = select_transcriber_model(registry, args.transcriber_provider)
    else:
        print(f"  ✓ Model: {args.transcriber_model} (set via CLI)")
    
    # Remote transcriber URL prompt (if remote transcriber selected)
    args = prompt_remote_transcriber_url(args)
    
    # Check if aligner is needed
    transcriber = registry.get_transcriber_provider(args.transcriber_provider)
    if not transcriber.has_builtin_alignment:
        if args.aligner_provider is None:
            args.aligner_provider = select_aligner_provider(registry)
            print(f"  ✓ Aligner: {args.aligner_provider}")
        else:
            print(f"  ✓ Aligner: {args.aligner_provider} (set via CLI)")
    else:
        print("  ✓ Aligner: Not needed (transcriber has built-in alignment)")
        args.aligner_provider = None
    
    # No diarization needed for split audio
    args.diarization_provider = None
    print("  ✓ Diarization: Not needed (speakers are in separate files)")
    
    # Granite-specific settings
    if args.transcriber_provider and 'granite' in args.transcriber_provider:
        args.output_format = "chunked"
        print("  ✓ Using chunk stitching for Granite")
    
    # De-identification
    args = prompt_de_identification(args, PipelineMode.SPLIT_AUDIO)
    
    # Output formats
    args = prompt_output_formats(args, registry)
    
    # Transcript cleanup
    args = prompt_transcript_cleanup(args, registry)
    
    return args


# =============================================================================
# Main Interactive Prompt Entry Point
# =============================================================================

def display_configuration_summary(args, mode: str):
    """Display a summary of the selected configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    mode_names = {
        PipelineMode.SINGLE_SPEAKER: "Single Speaker Audio",
        PipelineMode.COMBINED_AUDIO: "Combined Audio (Multi-Speaker)",
        PipelineMode.SPLIT_AUDIO: "Split Audio (Standard)",
        PipelineMode.VAD_SPLIT_AUDIO: "Split Audio (VAD Pipeline)",
    }
    
    print(f"\n  Mode: {mode_names.get(mode, mode)}")
    print(f"  System: {getattr(args, 'system', 'auto').upper()}")
    
    if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
        print(f"  Transcriber: {args.transcriber_provider}")
        # Show remote server URL if using remote transcriber
        if args.transcriber_provider == "remote":
            print(f"  Remote Server: {getattr(args, 'remote_transcriber_url', 'http://0.0.0.0:7070')}")
    if hasattr(args, 'transcriber_model') and args.transcriber_model:
        print(f"  Model: {args.transcriber_model}")
    if hasattr(args, 'aligner_provider') and args.aligner_provider:
        print(f"  Aligner: {args.aligner_provider}")
    if hasattr(args, 'diarization_provider') and args.diarization_provider:
        print(f"  Diarization: {args.diarization_provider}")
    if hasattr(args, 'num_speakers') and args.num_speakers:
        print(f"  Number of speakers: {args.num_speakers}")
    
    if mode == PipelineMode.VAD_SPLIT_AUDIO:
        print(f"  VAD threshold: {getattr(args, 'vad_threshold', 0.5)}")
        print(f"  VAD merge gap: {getattr(args, 'vad_merge_gap_ms', 600)}ms")
    
    print(f"  De-identification: {'Enabled' if getattr(args, 'de_identify', False) else 'Disabled'}")
    if getattr(args, 'de_identify', False) and getattr(args, 'de_identify_second_pass', False):
        print(f"  Second-pass de-id: Enabled")
    
    if hasattr(args, 'selected_outputs') and args.selected_outputs:
        print(f"  Output formats: {', '.join(args.selected_outputs)}")
    
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider:
        print(f"  Transcript cleanup: {args.transcript_cleanup_provider}")
    
    print("\n" + "=" * 60)


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