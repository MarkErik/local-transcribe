#!/usr/bin/env python3
# framework/cli.py - CLI argument parsing and interactive prompts

import argparse
from typing import Optional

from local_transcribe.lib.environment import get_available_system_capabilities

# ---------- CLI ----------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="local-transcribe: batch transcription – offline, Apple Silicon friendly."
    )
    p.add_argument("-a", "--audio-files", nargs='+', metavar="AUDIO_FILE", help="Audio files to process. One file = mixed audio with multiple speakers. Multiple files = separate tracks (2 files: interviewer + participant, 3+ files: prompt for speaker names).")
    p.add_argument("--transcriber-provider", help="Transcriber provider to use")
    p.add_argument("--transcriber-model", help="Transcriber model to use (if provider supports multiple models)")
    p.add_argument("--aligner-provider", help="Aligner provider to use (required if transcriber doesn't have built-in alignment)")
    p.add_argument("--diarization-provider", help="Diarization provider to use (required for single audio files with multiple speakers)")
    p.add_argument("--num-speakers", type=int, help="Number of speakers expected in the audio (for diarization)")
    p.add_argument("--transcript-cleanup-provider", help="Transcript cleanup provider to use for LLM-based transcript cleaning")
    p.add_argument("--transcript-cleanup-url", help="URL for remote transcript cleanup provider (e.g., http://ip:port for Llama.cpp server)")
    p.add_argument("--turn-builder-provider", help="Turn builder provider to use (for grouping transcribed words into turns)")
    p.add_argument("--llm-stitcher-url", help="URL for LLM stitcher provider (e.g., http://ip:port for LLM server)")
    p.add_argument("--llm-turn-builder-url", help="URL for LLM turn builder provider (e.g., http://ip:port for LLM server)")
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("-i", "--interactive", action="store_true", help="Interactive mode: prompt for provider and output selections.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging output.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("-s", "--system", choices=["cuda", "mps", "cpu"], help="System capability to use for ML acceleration. If not specified, will auto-detect available capabilities.")

    args = p.parse_args(argv)

    return args

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
            choice = int(input(f"\nChoose {provider_type.lower()} (number): ").strip())
            if 1 <= choice <= len(providers):
                return list(providers.keys())[choice - 1]
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Please enter a valid number.")

def interactive_prompt(args, api):
    registry = api["registry"]
    print("\n=== Interactive Mode ===")
    print("Select your processing options:\n")

    # System capability selection
    available_capabilities = get_available_system_capabilities()
    
    # Determine preferred default: MPS > CUDA > CPU
    if "mps" in available_capabilities:
        default_capability = "mps"
        default_index = 1  # MPS will be shown as option 1
    elif "cuda" in available_capabilities:
        default_capability = "cuda"
        default_index = 1  # CUDA will be shown as option 1
    else:
        default_capability = "cpu"
        default_index = 1  # CPU will be shown as option 1
    
    if len(available_capabilities) > 1:
        print("Available System Capabilities:")
        for i, cap in enumerate(available_capabilities, 1):
            marker = " (DEFAULT)" if cap == default_capability else ""
            print(f"  {i}. {cap.upper()}{marker}")
        
        while True:
            try:
                choice_input = input(f"\nChoose system capability (number, or press Enter for default): ").strip()
                
                # If user presses Enter, default to the preferred capability
                if not choice_input:
                    args.system = default_capability
                    print(f"✓ Selected system capability: {args.system.upper()} (default)")
                    break
                else:
                    choice = int(choice_input)
                    if 1 <= choice <= len(available_capabilities):
                        args.system = available_capabilities[choice - 1]
                        print(f"✓ Selected system capability: {args.system.upper()}")
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        args.system = available_capabilities[0]  # Only CPU available
        print(f"✓ Selected system capability: {args.system.upper()}")

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
        print("  1. Unified Provider (Transcription + Diarization in one step)")
        print("  2. Separate Providers (Transcription then Diarization)")
        
        while True:
            choice_input = input("\nChoose processing mode (1 or 2) [1]: ").strip()
            if not choice_input:
                choice = 1
            else:
                try:
                    choice = int(choice_input)
                except ValueError:
                    print("Please enter a valid number.")
                    continue
            
            if choice == 1:
                args.processing_mode = "unified"
                break
            elif choice == 2:
                args.processing_mode = "separate"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    else:
        args.processing_mode = "separate"  # For multi-file, always separate

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
                print(f"  {i}. {model}")
            
            while True:
                choice_input = input(f"\nChoose model (1-{len(available_models)}) [1]: ").strip()
                if not choice_input:
                    choice = 1
                else:
                    try:
                        choice = int(choice_input)
                    except ValueError:
                        print("Please enter a valid number.")
                        continue
                
                if 1 <= choice <= len(available_models):
                    args.unified_model = available_models[choice - 1]
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
        else:
            args.unified_model = available_models[0] if available_models else None

        # Number of speakers (for unified providers)
        if hasattr(args, 'audio_files') and args.audio_files and len(args.audio_files) == 1:
            while True:
                try:
                    num_speakers = int(input("\nNumber of speakers expected in the audio: ").strip())
                    if num_speakers > 0:
                        args.num_speakers = num_speakers
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Please enter a valid number.")

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
                print(f"  {i}. {model}")
            
            while True:
                choice_input = input(f"\nChoose model (1-{len(available_models)}) [1]: ").strip()
                if not choice_input:
                    choice = 1
                else:
                    try:
                        choice = int(choice_input)
                    except ValueError:
                        print("Please enter a valid number.")
                        continue
                
                if 1 <= choice <= len(available_models):
                    args.transcriber_model = available_models[choice - 1]
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
        else:
            args.transcriber_model = available_models[0] if available_models else None

        # Granite-specific options
        if args.transcriber_provider == "granite":
            print(f"\nGranite Chunking Options:")
            print(f"  Chunking processes long audio in segments to manage memory.")
            print(f"  - ON: Stable memory usage, slight quality reduction at boundaries")
            print(f"  - OFF: Maximum quality, higher memory usage (may crash on long audio)")
            
            while True:
                choice = input(f"\nEnable chunking? (y/n) [y]: ").strip().lower()
                if choice in ['y', 'yes', '']:
                    args.disable_chunking = False
                    print("✓ Chunking enabled (recommended for stability)")
                    break
                elif choice in ['n', 'no']:
                    args.disable_chunking = True
                    print("✓ Chunking disabled (maximum quality, monitor memory)")
                    break
                else:
                    print("Please enter 'y' or 'n'.")

            # Stitching method choice
            print(f"\nStitching Options:")
            print(f"  - Built-in: Fast, uses overlap detection to merge chunks")
            print(f"  - LLM: Slower, uses AI to intelligently merge chunks (requires LLM server)")
            
            while True:
                choice = input(f"\nChoose stitching method (1=Built-in, 2=LLM) [1]: ").strip()
                if choice in ['1', '']:
                    args.output_mode = 'stitched'
                    print("✓ Using built-in stitching")
                    break
                elif choice == '2':
                    args.output_mode = 'chunked'
                    # Prompt for LLM URL
                    default_url = getattr(args, 'llm_stitcher_url', 'http://0.0.0.0:8080')
                    llm_url = input(f"LLM server URL [{default_url}]: ").strip()
                    if not llm_url:
                        llm_url = default_url
                    else:
                        # Add http:// if not present
                        if not llm_url.startswith(('http://', 'https://')):
                            llm_url = f"http://{llm_url}"
                    args.llm_stitcher_url = llm_url
                    print(f"✓ Using LLM stitching with URL: {llm_url}")
                    break
                else:
                    print("Please enter 1 or 2.")

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
                        print(f"  {i}. {display_name} (RECOMMENDED)")
                    else:
                        print(f"  {i}. {display_name}")
                
                while True:
                    choice_input = input(f"\nChoose turn builder (number) [1]: ").strip()
                    if not choice_input:
                        choice = 1
                    else:
                        try:
                            choice = int(choice_input)
                        except ValueError:
                            print("Please enter a valid number.")
                            continue
                    
                    if 1 <= choice <= len(all_turn_builders):
                        args.turn_builder_provider = list(all_turn_builders.keys())[choice - 1]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
            else:
                # Default to split_audio_turn_builder if available, otherwise fall back
                if split_audio_turn_builders:
                    args.turn_builder_provider = list(split_audio_turn_builders.keys())[0]
                elif single_speaker_turn_builders:
                    args.turn_builder_provider = list(single_speaker_turn_builders.keys())[0]
                else:
                    args.turn_builder_provider = "split_audio_turn_builder"  # Fallback default

            # If LLM turn builder selected, prompt for URL
            if hasattr(args, 'turn_builder_provider') and args.turn_builder_provider == "split_audio_llm":
                default_url = getattr(args, 'llm_turn_builder_url', 'http://0.0.0.0:8080')
                llm_url = input(f"LLM server URL for turn building [{default_url}]: ").strip()
                if not llm_url:
                    llm_url = default_url
                else:
                    # Add http:// if not present
                    if not llm_url.startswith(('http://', 'https://')):
                        llm_url = f"http://{llm_url}"
                args.llm_turn_builder_url = llm_url
                print(f"✓ Using LLM turn builder with URL: {llm_url}")

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
                    num_speakers = int(input("\nNumber of speakers expected in the audio: ").strip())
                    if num_speakers > 0:
                        args.num_speakers = num_speakers
                        break
                    else:
                        print("Please enter a positive number.")
                except ValueError:
                    print("Please enter a valid number.")

    # Transcript cleanup provider (optional)
    transcript_cleanup_providers = registry.list_transcript_cleanup_providers()
    if transcript_cleanup_providers:
        print("\nTranscript Cleanup Providers (optional LLM-based transcript cleaning):")
        print("  0. None (skip cleanup)")
        for i, (name, desc) in enumerate(transcript_cleanup_providers.items(), 1):
            print(f"  {i}. {name}: {desc}")
        
        while True:
            try:
                choice_input = input("\nChoose transcript cleanup provider (0 for none, or press Enter for default): ").strip()
                
                # If user presses Enter, default to 0 (None)
                if not choice_input:
                    choice = 0
                else:
                    choice = int(choice_input)
                    
                if choice == 0:
                    args.transcript_cleanup_provider = None
                    break
                elif 1 <= choice <= len(transcript_cleanup_providers):
                    args.transcript_cleanup_provider = list(transcript_cleanup_providers.keys())[choice - 1]

                    # If remote provider, ask for URL
                    if args.transcript_cleanup_provider == "llama_cpp_remote":
                        default_url = getattr(args, 'transcript_cleanup_url', 'http://localhost:8080')
                        url = input(f"Enter Llama.cpp server URL (e.g., http://192.168.1.100:8080) [{default_url}]: ").strip()
                        if url:
                            args.transcript_cleanup_url = url
                        else:
                            args.transcript_cleanup_url = default_url
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        args.transcript_cleanup_provider = None

    # Output formats - filter out SRT as it's now handled internally by video
    output_writers = registry.list_output_writers()
    filtered_writers = {name: desc for name, desc in output_writers.items()
                       if name not in ['srt']}
    
    print("\nAvailable Output Formats:")
    for i, (name, desc) in enumerate(filtered_writers.items(), 1):
        print(f"  {i}. {name}: {desc}")

    print("\nEnter numbers separated by commas (e.g., 1,3,5), or press Enter for all formats:")
    choice = input("Choose output formats: ").strip()

    if not choice:
        args.selected_outputs = list(filtered_writers.keys())
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(filtered_writers)]
            args.selected_outputs = [list(filtered_writers.keys())[i] for i in valid_indices]
            if not args.selected_outputs:
                print("No valid choices, selecting all.")
                args.selected_outputs = list(filtered_writers.keys())
        except ValueError:
            print("Invalid input, selecting all.")
            args.selected_outputs = list(filtered_writers.keys())

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