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
    p.add_argument("-d", "--de-identify", action="store_true", help="Enable de-identification to replace people's names with [REDACTED] [Default: prompt in interactive mode]")
    p.add_argument("--de-identify-second-pass", action="store_true", help="Enable second-pass de-identification using names discovered across all speakers [Default: prompt in interactive mode if de-identify enabled]")

    p.add_argument("--transcriber-provider", help="Transcriber provider to use [Default: auto-selected]")
    p.add_argument("--transcriber-model", help="Transcriber model to use (if provider supports multiple models) [Default: provider-specific]")
    p.add_argument("--stitching-method", choices=["local", "llm"], default="local", help="Method for stitching transcript chunks [Default: local]. local=intelligent local overlap detection, llm=external LLM server stitching")
    p.add_argument("--aligner-provider", help="Aligner provider to use (required if transcriber doesn't have built-in alignment) [Default: auto-selected if needed]")
    p.add_argument("--diarization-provider", help="Diarization provider to use (required for single audio files with multiple speakers) [Default: auto-selected if needed]")
    p.add_argument("--transcript-cleanup-provider", help="Transcript cleanup provider to use for LLM-based transcript cleaning [Default: none]")
    
    p.add_argument("--llm-stitcher-url", default="http://0.0.0.0:8080", help="URL for LLM stitcher provider (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    p.add_argument("--llm-de-identifier-url", default="http://0.0.0.0:8080", help="URL for LLM personal information de-identifier processor (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    p.add_argument("--llm-transcript-cleanup-url", default="http://0.0.0.0:8080", help="URL for remote transcript cleanup provider (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")

    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--show-defaults", action="store_true", help="Show all default values and exit.")

    # Pipeline re-entry arguments
    p.add_argument("--from-diarized-json", metavar="JSON_FILE", help="Resume pipeline from a corrected diarized word segments JSON file. Starts from turn-building stage.")
    p.add_argument("--audio-for-video", metavar="AUDIO_FILE", help="Original audio file path for video generation when resuming from checkpoint.")
    p.add_argument("--mode", choices=["combined_audio", "split_audio"], help="Pipeline mode. Overrides mode detected from checkpoint metadata.")
    p.add_argument("--speaker-map", metavar="MAPPING", help="Speaker name mapping for re-entry (e.g., 'SPEAKER_00=Interviewer,SPEAKER_01=Participant')")
    p.add_argument("--dry-run", action="store_true", help="Validate checkpoint and show what stages would run without executing.")
    p.add_argument("--list-stages", action="store_true", help="List available pipeline stages and exit.")

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
    print("  - Chunking (Granite): Always enabled")
    print("  - Stitching Method (Granite): local (default) or llm available")
    
    print("\nURLs:")
    print("  - LLM Stitcher URL: http://0.0.0.0:8080")
    print("  - LLM Turn Builder URL: http://0.0.0.0:8080")
    print("  - LLM Transcript Cleanup URL: http://0.0.0.0:8080")
    
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


def select_provider(registry, provider_type):
    """Helper to select a provider from registry."""
    if provider_type == "transcriber":
        providers = registry._transcriber_providers
    elif provider_type == "aligner":
        providers = registry._aligner_providers
    elif provider_type == "diarization":
        providers = registry._diarization_providers
    elif provider_type == "transcript_cleanup":
        providers = registry._transcript_cleanup_providers
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    print(f"\nAvailable {provider_type.capitalize()} Providers:")
    for i, (name, provider) in enumerate(providers.items(), 1):
        display_name = getattr(provider, 'short_name', provider.description)
        print(f"  {i}. {display_name}")

    while True:
        try:
            choice = int(input(f"\nSelect {provider_type.lower()} (number): ").strip())
            if 1 <= choice <= len(providers):
                return list(providers.keys())[choice - 1]
            else:
                print("Error: Please enter a number from the list.")
        except ValueError:
            print("Error: Please enter a valid number.")

def interactive_prompt(args, api):
    registry = api["registry"]
    print("\n=== Interactive Mode ===")
    print("Select your processing options:\n")

    # Check for single-speaker-audio mode
    if hasattr(args, 'single_speaker_audio') and args.single_speaker_audio:
        print("Mode: Single Speaker Audio (transcription only, CSV output)")
        # Only prompt for system capability and transcriber provider
        # System capability selection
        available_capabilities = get_available_system_capabilities()
        
        # Determine preferred default: MPS > CUDA > CPU
        if "mps" in available_capabilities:
            default_capability = "mps"
        elif "cuda" in available_capabilities:
            default_capability = "cuda"
        else:
            default_capability = "cpu"
        
        # Find the index of the default capability
        default_index = available_capabilities.index(default_capability) + 1
        
        if len(available_capabilities) > 1:
            print("Available System Capabilities:")
            for i, cap in enumerate(available_capabilities, 1):
                marker = " [Default]" if cap == default_capability else ""
                print(f"  {i}. {cap.upper()}{marker}")
            
            while True:
                try:
                    choice_input = input(f"\nSelect system capability (number) [Default: {default_index}]: ").strip()
                    
                    # If user presses Enter, default to the preferred capability
                    if not choice_input:
                        args.system = default_capability
                        print(f"✓ Selected: {args.system.upper()} [Default]")
                        break
                    else:
                        choice = int(choice_input)
                        if 1 <= choice <= len(available_capabilities):
                            args.system = available_capabilities[choice - 1]
                            is_default = args.system == default_capability
                            default_marker = " [Default]" if is_default else ""
                            print(f"✓ Selected: {args.system.upper()}{default_marker}")
                            break
                        else:
                            print("Error: Please enter a number from the list.")
                except ValueError:
                    print("Error: Please enter a valid number.")
        else:
            args.system = available_capabilities[0]  # Only CPU available
            print(f"✓ Selected: {args.system.upper()} [Default]")

        # Select transcriber provider (only pure ones)
        transcriber_providers = registry.list_transcriber_providers()
        eligible_providers = {}
        for name, desc in transcriber_providers.items():
            provider = registry.get_transcriber_provider(name)
            if not provider.has_builtin_alignment:
                eligible_providers[name] = desc
        
        if not eligible_providers:
            print("Error: No pure transcriber providers available.")
            return args
        
        print("\nAvailable Pure Transcriber Providers:")
        for i, (name, desc) in enumerate(eligible_providers.items(), 1):
            provider = registry.get_transcriber_provider(name)
            display_name = getattr(provider, 'short_name', provider.description)
            print(f"  {i}. {display_name}")
        
        while True:
            try:
                choice = int(input("\nSelect transcriber provider (number): ").strip())
                if 1 <= choice <= len(eligible_providers):
                    selected_provider = list(eligible_providers.keys())[choice - 1]
                    args.transcriber_provider = selected_provider
                    provider = registry.get_transcriber_provider(selected_provider)
                    display_name = getattr(provider, 'short_name', provider.description)
                    print(f"✓ Selected: {display_name}")
                    
                    # Model selection if provider has multiple models
                    available_models = provider.get_available_models()
                    if len(available_models) > 1:
                        print(f"\nAvailable models for {display_name}:")
                        for i, model in enumerate(available_models, 1):
                            print(f"  {i}. {model}")
                        
                        while True:
                            try:
                                model_choice = int(input(f"\nSelect model (number) [Default: 1]: ").strip())
                                if not model_choice:
                                    # For granite provider, default to granite-8b (index 1) instead of first model
                                    if selected_provider == "granite" and "granite-8b" in available_models:
                                        model_choice = available_models.index("granite-8b") + 1
                                    else:
                                        model_choice = 1
                                if 1 <= model_choice <= len(available_models):
                                    args.transcriber_model = available_models[model_choice - 1]
                                    is_default = (model_choice == 1 and selected_provider != "granite") or \
                                               (selected_provider == "granite" and args.transcriber_model == "granite-8b")
                                    default_marker = " [Default]" if is_default else ""
                                    print(f"✓ Selected model: {args.transcriber_model}{default_marker}")
                                    break
                                else:
                                    print("Error: Please enter a number from the list.")
                            except ValueError:
                                print("Error: Please enter a valid number.")
                    elif len(available_models) == 1:
                        args.transcriber_model = available_models[0]
                        print(f"✓ Using default model: {args.transcriber_model}")
                    
                    # Granite-specific options
                    if selected_provider == "granite":
                        # Always use chunking, but ask for stitcher method
                        print("\nGranite Stitching Options:")
                        print("  1. Local (use intelligent local overlap detection) [Default]")
                        print("  2. LLM-based (use external LLM server for stitching)")
                        
                        while True:
                            stitch_input = input("\nSelect stitching option (number) [Default: 1]: ").strip()
                            if not stitch_input or stitch_input == "1":
                                args.stitching_method = "local"
                                args.output_format = "chunked"
                                print("✓ Selected: Local intelligent stitching")
                                break
                            elif stitch_input == "2":
                                args.stitching_method = "llm"
                                args.output_format = "chunked"  # Using chunked mode for LLM stitching
                                # Prompt for LLM server URL
                                default_url = getattr(args, 'llm_stitcher_url', 'http://0.0.0.0:8080')
                                llm_url = input(f"\nLLM server URL [Default: {default_url}]: ").strip()
                                if not llm_url:
                                    args.llm_stitcher_url = default_url
                                    is_default = True
                                else:
                                    # Add http:// if not present
                                    if not llm_url.startswith(('http://', 'https://')):
                                        llm_url = f"http://{llm_url}"
                                    args.llm_stitcher_url = llm_url
                                    is_default = False
                                default_marker = " [Default]" if is_default else ""
                                print(f"✓ Selected: LLM-based stitching with URL: {args.llm_stitcher_url}{default_marker}")
                                break
                            else:
                                print("Error: Please enter 1 or 2.")
                    
                    break
                else:
                    print("Error: Please enter a number from the list.")
            except ValueError:
                print("Error: Please enter a valid number.")
        
        return args

    # System capability selection
    available_capabilities = get_available_system_capabilities()
    
    # Determine preferred default: MPS > CUDA > CPU
    if "mps" in available_capabilities:
        default_capability = "mps"
    elif "cuda" in available_capabilities:
        default_capability = "cuda"
    else:
        default_capability = "cpu"
    
    # Find the index of the default capability
    default_index = available_capabilities.index(default_capability) + 1
    
    if len(available_capabilities) > 1:
        print("Available System Capabilities:")
        for i, cap in enumerate(available_capabilities, 1):
            marker = " [Default]" if cap == default_capability else ""
            print(f"  {i}. {cap.upper()}{marker}")
        
        while True:
            try:
                choice_input = input(f"\nSelect system capability (number) [Default: {default_index}]: ").strip()
                
                # If user presses Enter, default to the preferred capability
                if not choice_input:
                    args.system = default_capability
                    print(f"✓ Selected: {args.system.upper()} [Default]")
                    break
                else:
                    choice = int(choice_input)
                    if 1 <= choice <= len(available_capabilities):
                        args.system = available_capabilities[choice - 1]
                        is_default = args.system == default_capability
                        default_marker = " [Default]" if is_default else ""
                        print(f"✓ Selected: {args.system.upper()}{default_marker}")
                        break
                    else:
                        print("Error: Please enter a number from the list.")
            except ValueError:
                print("Error: Please enter a valid number.")
    else:
        args.system = available_capabilities[0]  # Only CPU available
        print(f"✓ Selected: {args.system.upper()} [Default]")

    # De-identification prompt (if not already set via CLI flag)
    if not args.de_identify:
        response = input("\nWould you like to de-identify the transcript (replace people's names with [REDACTED])? [Default: Y/n]: ").strip().lower()
        args.de_identify = response != 'n'  # Default to Yes
        if args.de_identify:
            print("✓ De-identification enabled")
        else:
            print("De-identification disabled")
    else:
        print("✓ De-identification enabled (set via CLI flag)")
    
    # Second-pass de-identification prompt (only if de-identify is enabled)
    if args.de_identify and not args.de_identify_second_pass:
        response = input("\nEnable second-pass de-identification? (uses names found across all speakers to catch missed instances) [Default: Y/n]: ").strip().lower()
        args.de_identify_second_pass = response != 'n'  # Default to Yes
        if args.de_identify_second_pass:
            print("✓ Second-pass de-identification enabled")
        else:
            print("Second-pass de-identification disabled")
    elif args.de_identify and args.de_identify_second_pass:
        print("✓ Second-pass de-identification enabled (set via CLI flag)")

    # Determine mode based on number of audio files
    if hasattr(args, 'audio_files') and args.audio_files:
        num_files = len(args.audio_files)
        if num_files == 1:
            mode = "combined_audio"
            print("Mode: Combined audio (single file with multiple speakers)")
        else:
            mode = "split_audio"
            print(f"Mode: Split audio tracks ({num_files} files)")
    else:
        # Fallback for when audio files aren't set yet
        mode = "combined_audio"

    # Select transcriber provider
    transcriber_providers = registry.list_transcriber_providers()
    args.transcriber_provider = select_provider(registry, "transcriber")

    # Model selection for transcriber provider
    transcriber_provider = registry.get_transcriber_provider(args.transcriber_provider)
    available_models = transcriber_provider.get_available_models()
    if len(available_models) > 1:
        print(f"\nAvailable models for {args.transcriber_provider}:")
        for i, model in enumerate(available_models, 1):
            default_marker = " [Default]" if i == 1 else ""
            print(f"  {i}. {model}{default_marker}")
        
        while True:
            choice_input = input(f"\nSelect model (number) [Default: 1]: ").strip()
            if not choice_input:
                # For granite provider, default to granite-8b (index 1) instead of first model
                if args.transcriber_provider == "granite" and "granite-8b" in available_models:
                    choice = available_models.index("granite-8b") + 1
                else:
                    choice = 1
            else:
                try:
                    choice = int(choice_input)
                except ValueError:
                    print("Error: Please enter a valid number.")
                    continue
            
            if 1 <= choice <= len(available_models):
                args.transcriber_model = available_models[choice - 1]
                is_default = (choice == 1 and args.transcriber_provider != "granite") or \
                           (args.transcriber_provider == "granite" and args.transcriber_model == "granite-8b")
                default_marker = " [Default]" if is_default else ""
                print(f"✓ Selected: {args.transcriber_model}{default_marker}")
                break
            else:
                print("Error: Please enter a number from the list.")
    else:
        args.transcriber_model = available_models[0] if available_models else None
        if args.transcriber_model:
            print(f"✓ Selected: {args.transcriber_model} [Default]")

    # Granite-specific options
    if args.transcriber_provider == "granite":
        # Always use chunking, but ask for stitcher method
        print(f"\nGranite Stitching Options:")
        print(f"  1. Local: Intelligent local overlap detection [Default]")
        print(f"  2. LLM: External AI server for intelligent chunk merging (requires LLM server)")
        
        while True:
            choice = input(f"\nSelect stitching method (number) [Default: 1]: ").strip()
            if choice in ['1', '']:
                args.output_format = 'chunked'
                args.stitching_method = "local"
                print("✓ Selected: Local intelligent stitching [Default]")
                break
            elif choice == '2':
                args.output_format = 'chunked'
                args.stitching_method = "llm"
                # Prompt for LLM URL
                default_url = getattr(args, 'llm_stitcher_url', 'http://0.0.0.0:8080')
                llm_url = input(f"LLM server URL [Default: {default_url}]: ").strip()
                if not llm_url:
                    llm_url = default_url
                    is_default = True
                else:
                    # Add http:// if not present
                    if not llm_url.startswith(('http://', 'https://')):
                        llm_url = f"http://{llm_url}"
                    is_default = False
                args.llm_stitcher_url = llm_url
                default_marker = " [Default]" if is_default else ""
                print(f"✓ Selected: LLM stitching with URL: {llm_url}{default_marker}")
                break
            else:
                print("Error: Please enter 1 or 2.")

    # Check if aligner is needed
    if not transcriber_provider.has_builtin_alignment:
        # Select aligner provider
        aligner_providers = registry.list_aligner_providers()
        args.aligner_provider = select_provider(registry, "aligner")
    else:
        args.aligner_provider = None

    # Select diarization provider (only needed for combined audio mode)
    if mode == "combined_audio":
        diarization_providers = registry.list_diarization_providers()
        args.diarization_provider = select_provider(registry, "diarization")
    else:
        args.diarization_provider = None

    # Number of speakers
    if hasattr(args, 'audio_files') and args.audio_files and len(args.audio_files) == 1:
        while True:
            try:
                num_speakers = int(input("\nNumber of speakers expected in the audio [Default: 2]: ").strip())
                if num_speakers > 0:
                    args.num_speakers = num_speakers
                    is_default = num_speakers == 2
                    default_marker = " [Default]" if is_default else ""
                    print(f"✓ Selected: {num_speakers} speakers{default_marker}")
                    break
                else:
                    print("Error: Please enter a positive number.")
            except ValueError:
                print("Error: Please enter a valid number.")

    # Transcript cleanup provider (optional)
    transcript_cleanup_providers = registry.list_transcript_cleanup_providers()
    if transcript_cleanup_providers:
        print("\nTranscript Cleanup Providers (optional LLM-based transcript cleaning):")
        print("  0. None (skip cleanup) [Default]")
        for i, (name, desc) in enumerate(transcript_cleanup_providers.items(), 1):
            print(f"  {i}. {name}: {desc}")
        
        while True:
            try:
                choice_input = input("\nSelect transcript cleanup provider (number) [Default: 0]: ").strip()
                
                # If user presses Enter, default to 0 (None)
                if not choice_input:
                    choice = 0
                else:
                    choice = int(choice_input)
                    
                if choice == 0:
                    args.transcript_cleanup_provider = None
                    print("✓ Selected: None [Default]")
                    break
                elif 1 <= choice <= len(transcript_cleanup_providers):
                    args.transcript_cleanup_provider = list(transcript_cleanup_providers.keys())[choice - 1]
                    is_default = False

                    # If remote provider, ask for URL
                    if args.transcript_cleanup_provider == "llm_transcript_cleanup":
                        default_url = getattr(args, 'llm_transcript_cleanup_url', 'http://0.0.0.0:8080')
                        url = input(f"Enter llama.cpp server URL (e.g., http://0.0.0.0:8080) [Default: {default_url}]: ").strip()
                        if url:
                            # Add http:// if not present
                            if not url.startswith(('http://', 'https://')):
                                url = f"http://{url}"
                            args.llm_transcript_cleanup_url = url
                            is_default = False
                        else:
                            args.llm_transcript_cleanup_url = default_url
                            is_default = True
                    
                    default_marker = " [Default]" if is_default else ""
                    print(f"✓ Selected: {args.transcript_cleanup_provider}{default_marker}")
                    break
                else:
                    print("Error: Please enter a number from the list.")
            except ValueError:
                print("Error: Please enter a valid number.")
    else:
        args.transcript_cleanup_provider = None
        print("✓ Selected: None [Default]")

    # Output formats - filter out SRT as it's now handled internally by video
    output_writers = registry.list_output_writers()
    filtered_writers = {name: desc for name, desc in output_writers.items()
                       if name not in ['srt']}
    
    print("\nAvailable Output Formats:")
    for i, (name, desc) in enumerate(filtered_writers.items(), 1):
        print(f"  {i}. {name}: {desc}")

    print("\nEnter numbers separated by commas (e.g., 1,3,5), or press Enter for all formats [Default: all]:")
    choice = input("Select output formats: ").strip()

    if not choice:
        args.selected_outputs = list(filtered_writers.keys())
        print("✓ Selected: All output formats [Default]")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(filtered_writers)]
            args.selected_outputs = [list(filtered_writers.keys())[i] for i in valid_indices]
            if not args.selected_outputs:
                print("Error: No valid choices, selecting all.")
                args.selected_outputs = list(filtered_writers.keys())
                print("✓ Selected: All output formats [Default]")
            else:
                selected_names = [list(filtered_writers.keys())[i] for i in valid_indices]
                print(f"✓ Selected: {', '.join(selected_names)}")
        except ValueError:
            print("Error: Invalid input, selecting all.")
            args.selected_outputs = list(filtered_writers.keys())
            print("✓ Selected: All output formats [Default]")

    provider_info = []
    if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
        provider_info.append(f"Transcriber={args.transcriber_provider}")
    if hasattr(args, 'aligner_provider') and args.aligner_provider:
        provider_info.append(f"Aligner={args.aligner_provider}")
    if hasattr(args, 'diarization_provider') and args.diarization_provider:
        provider_info.append(f"Diarization={args.diarization_provider}")
    if hasattr(args, 'turn_builder_provider') and args.turn_builder_provider:
        provider_info.append(f"Turn Builder={args.turn_builder_provider}")
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider:
        provider_info.append(f"Transcript Cleanup={args.transcript_cleanup_provider}")
    provider_str = ", ".join(provider_info) if provider_info else "Default providers"
    print(f"\nSelected: {provider_str}, Outputs={args.selected_outputs}")
    return args