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
    p.add_argument("-d", "--de-identify")

    p.add_argument("--transcriber-provider", help="Transcriber provider to use [Default: auto-selected]")
    p.add_argument("--transcriber-model", help="Transcriber model to use (if provider supports multiple models) [Default: provider-specific]")
    p.add_argument("--stitching-method", choices=["local", "llm"], default="local", help="Method for stitching transcript chunks [Default: local]. local=intelligent local overlap detection, llm=external LLM server stitching")
    p.add_argument("--aligner-provider", help="Aligner provider to use (required if transcriber doesn't have built-in alignment) [Default: auto-selected if needed]")
    p.add_argument("--diarization-provider", help="Diarization provider to use (required for single audio files with multiple speakers) [Default: auto-selected if needed]")
    p.add_argument("--turn-builder-provider", help="Turn builder provider to use (for grouping transcribed words into turns) [Default: auto-selected based on audio mode]")
    p.add_argument("--transcript-cleanup-provider", help="Transcript cleanup provider to use for LLM-based transcript cleaning [Default: none]")
    
    p.add_argument("--llm-stitcher-url", default="http://0.0.0.0:8080", help="URL for LLM stitcher provider (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    p.add_argument("--llm-de-identifier-url", default="http://0.0.0.0:8080", help="URL for LLM personal information de-identifier processor (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    p.add_argument("--llm-turn-builder-url", default="http://0.0.0.0:8080", help="URL for LLM turn builder provider (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")
    p.add_argument("--llm-transcript-cleanup-url", default="http://localhost:8080", help="URL for remote transcript cleanup provider (e.g., http://ip:port for LLM server) [Default: http://0.0.0.0:8080]")

    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--show-defaults", action="store_true", help="Show all default values and exit.")

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
    print("  - Turn Builder Provider: Auto-selected based on audio mode")
    print("  - Transcript Cleanup Provider: None (disabled)")
    
    print("\nConfiguration:")
    print("  - Number of Speakers: 2")
    print("  - Output Formats: All available formats")
    print("  - Processing Mode: Unified for single audio, Separate for multiple audio")
    print("  - Single Speaker Audio: Disabled (use -s to enable)")
    print("  - Chunking (Granite): Always enabled")
    print("  - Stitching Method (Granite): local (default) or llm available")
    
    print("\nURLs:")
    print("  - LLM Stitcher URL: http://0.0.0.0:8080")
    print("  - LLM Turn Builder URL: http://0.0.0.0:8080")
    print("  - LLM Transcript Cleanup URL: http://localhost:8080")
    
    print("\nNote: Some defaults may be overridden by system capabilities or provider availability.")

def select_provider(registry, provider_type):
    """Helper to select a provider from registry."""
    if provider_type == "transcriber":
        providers = registry._transcriber_providers
    elif provider_type == "aligner":
        providers = registry._aligner_providers
    elif provider_type == "diarization":
        providers = registry._diarization_providers
    elif provider_type == "unified":
        providers = registry._unified_providers
    elif provider_type == "transcript_cleanup":
        providers = registry._transcript_cleanup_providers
    elif provider_type == "turn_builder":
        providers = registry._turn_builder_providers
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
                                    model_choice = 1
                                if 1 <= model_choice <= len(available_models):
                                    args.transcriber_model = available_models[model_choice - 1]
                                    print(f"✓ Selected model: {args.transcriber_model}")
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

    # Check if unified mode and offer processing type choice
    unified_providers = registry.list_unified_providers()
    if unified_providers and hasattr(args, 'audio_files') and args.audio_files and len(args.audio_files) == 1:
        print("Processing Modes:")
        print("  1. Unified Provider (Transcription + Diarization in one step) [Default]")
        print("  2. Separate Providers (Transcription then Diarization)")
        
        while True:
            choice_input = input("\nSelect processing mode (number) [Default: 1]: ").strip()
            if not choice_input:
                choice = 1
            else:
                try:
                    choice = int(choice_input)
                except ValueError:
                    print("Error: Please enter a valid number.")
                    continue
            
            if choice == 1:
                args.processing_mode = "unified"
                is_default = True
                break
            elif choice == 2:
                args.processing_mode = "separate"
                is_default = False
                break
            else:
                print("Error: Please enter 1 or 2.")
        
        mode_name = "Unified Provider" if args.processing_mode == "unified" else "Separate Providers"
        default_marker = " [Default]" if is_default else ""
        print(f"✓ Selected: {mode_name}{default_marker}")
    else:
        args.processing_mode = "separate"  # For multi-file, always separate
        print("✓ Selected: Separate Providers [Default]")

    if args.processing_mode == "unified":
        # Select unified provider
        unified_providers = registry.list_unified_providers()
        args.unified_provider = select_provider(registry, "unified")

        # Model selection for unified provider
        unified_provider = registry.get_unified_provider(args.unified_provider)
        available_models = unified_provider.get_available_models()
        if len(available_models) > 1:
            print(f"\nAvailable models for {args.unified_provider}:")
            for i, model in enumerate(available_models, 1):
                default_marker = " [Default]" if i == 1 else ""
                print(f"  {i}. {model}{default_marker}")
            
            while True:
                choice_input = input(f"\nSelect model (number) [Default: 1]: ").strip()
                if not choice_input:
                    choice = 1
                else:
                    try:
                        choice = int(choice_input)
                    except ValueError:
                        print("Error: Please enter a valid number.")
                        continue
                
                if 1 <= choice <= len(available_models):
                    args.unified_model = available_models[choice - 1]
                    is_default = choice == 1
                    default_marker = " [Default]" if is_default else ""
                    print(f"✓ Selected: {args.unified_model}{default_marker}")
                    break
                else:
                    print("Error: Please enter a number from the list.")
        else:
            args.unified_model = available_models[0] if available_models else None
            if args.unified_model:
                print(f"✓ Selected: {args.unified_model} [Default]")

        # Number of speakers (for unified providers)
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

    else:
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
                    choice = 1
                else:
                    try:
                        choice = int(choice_input)
                    except ValueError:
                        print("Error: Please enter a valid number.")
                        continue
                
                if 1 <= choice <= len(available_models):
                    args.transcriber_model = available_models[choice - 1]
                    is_default = choice == 1
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

        # Select turn builder for split audio
        if mode == "split_audio":
            turn_builders = registry._turn_builder_providers
            # For split audio mode, show the split_audio_turn_builder as the primary option
            split_audio_turn_builders = {k: v for k, v in turn_builders.items() if k.startswith("split_audio")}
            # Also include single speaker turn builders as fallback options
            single_speaker_turn_builders = {k: v for k, v in turn_builders.items() if k.startswith("single_speaker")}
            
            # Combine split_audio first, then single_speaker options
            all_turn_builders = {**split_audio_turn_builders, **single_speaker_turn_builders}
            
            if len(all_turn_builders) > 1:
                print(f"\nAvailable Turn Builder Providers:")
                for i, (name, provider) in enumerate(all_turn_builders.items(), 1):
                    display_name = getattr(provider, 'short_name', provider.description)
                    # Mark the recommended option
                    if name.startswith("split_audio"):
                        marker = " [Recommended]" if i == 1 else ""
                        print(f"  {i}. {display_name}{marker}")
                    else:
                        print(f"  {i}. {display_name}")
                
                while True:
                    choice_input = input(f"\nSelect turn builder (number) [Default: 1]: ").strip()
                    if not choice_input:
                        choice = 1
                    else:
                        try:
                            choice = int(choice_input)
                        except ValueError:
                            print("Error: Please enter a valid number.")
                            continue
                    
                    if 1 <= choice <= len(all_turn_builders):
                        selected_key = list(all_turn_builders.keys())[choice - 1]
                        args.turn_builder_provider = selected_key
                        selected_provider = all_turn_builders[selected_key]
                        display_name = getattr(selected_provider, 'short_name', selected_provider.description)
                        is_default = choice == 1
                        default_marker = " [Default]" if is_default else ""
                        print(f"✓ Selected: {display_name}{default_marker}")
                        break
                    else:
                        print("Error: Please enter a number from the list.")
            else:
                # Default to split_audio_turn_builder if available, otherwise fall back
                if split_audio_turn_builders:
                    args.turn_builder_provider = list(split_audio_turn_builders.keys())[0]
                elif single_speaker_turn_builders:
                    args.turn_builder_provider = list(single_speaker_turn_builders.keys())[0]
                else:
                    args.turn_builder_provider = "split_audio_turn_builder"  # Fallback default
                print(f"✓ Selected: {args.turn_builder_provider} [Default]")

            # If LLM turn builder selected, prompt for URL
            if hasattr(args, 'turn_builder_provider') and args.turn_builder_provider in ["split_audio_llm", "split_audio_llm_turn_builder_improved"]:
                default_url = getattr(args, 'llm_turn_builder_url', 'http://0.0.0.0:8080')
                llm_url = input(f"LLM server URL for turn building [Default: {default_url}]: ").strip()
                if not llm_url:
                    llm_url = default_url
                    is_default = True
                else:
                    # Add http:// if not present
                    if not llm_url.startswith(('http://', 'https://')):
                        llm_url = f"http://{llm_url}"
                    is_default = False
                args.llm_turn_builder_url = llm_url
                default_marker = " [Default]" if is_default else ""
                print(f"✓ Selected: LLM turn builder with URL: {llm_url}{default_marker}")

        # Select diarization provider (only needed for combined audio in separate mode)
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

    if args.processing_mode == "unified":
        print(f"\nSelected: Unified={args.unified_provider} ({args.unified_model}), Outputs={args.selected_outputs}")
    else:
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