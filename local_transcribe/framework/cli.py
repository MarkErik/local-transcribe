#!/usr/bin/env python3
# framework/cli.py - CLI argument parsing and interactive prompts

import argparse
from typing import Optional

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
    p.add_argument("--cleanup-provider", help="Cleanup provider to use for LLM-based transcript cleaning")
    p.add_argument("--cleanup-url", help="URL for remote cleanup provider (e.g., http://ip:port for Llama.cpp server)")
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--interactive", action="store_true", help="Interactive mode: prompt for provider and output selections.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")

    args = p.parse_args(argv)

    return args

def select_provider(providers, prompt):
    """Helper to select a provider from a dict."""
    print(f"\nAvailable {prompt}:")
    for i, (name, desc) in enumerate(providers.items(), 1):
        print(f"  {i}. {name}: {desc}")

    while True:
        try:
            choice = int(input(f"\nChoose {prompt.lower()} (number): ").strip())
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
            try:
                choice = int(input("\nChoose processing mode (1 or 2): ").strip())
                if choice == 1:
                    args.processing_mode = "unified"
                    break
                elif choice == 2:
                    args.processing_mode = "separate"
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        args.processing_mode = "separate"  # For multi-file, always separate

    if args.processing_mode == "unified":
        # Select unified provider
        unified_providers = registry.list_unified_providers()
        args.unified_provider = select_provider(unified_providers, "Unified Providers")

        # Model selection for unified provider
        unified_provider = registry.get_unified_provider(args.unified_provider)
        available_models = unified_provider.get_available_models()
        if len(available_models) > 1:
            print(f"\nAvailable models for {args.unified_provider}:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            
            while True:
                try:
                    choice = int(input(f"\nChoose model (1-{len(available_models)}): ").strip())
                    if 1 <= choice <= len(available_models):
                        args.unified_model = available_models[choice - 1]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Please enter a valid number.")
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
        args.transcriber_provider = select_provider(transcriber_providers, "Transcriber Providers")

        # Model selection for transcriber provider
        transcriber_provider = registry.get_transcriber_provider(args.transcriber_provider)
        available_models = transcriber_provider.get_available_models()
        if len(available_models) > 1:
            print(f"\nAvailable models for {args.transcriber_provider}:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            
            while True:
                try:
                    choice = int(input(f"\nChoose model (1-{len(available_models)}): ").strip())
                    if 1 <= choice <= len(available_models):
                        args.transcriber_model = available_models[choice - 1]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            args.transcriber_model = available_models[0] if available_models else None

        # Check if aligner is needed
        if not transcriber_provider.has_builtin_alignment:
            # Select aligner provider
            aligner_providers = registry.list_aligner_providers()
            args.aligner_provider = select_provider(aligner_providers, "Aligner Providers")
        else:
            args.aligner_provider = None

        # Select diarization provider (only needed for combined audio in separate mode)
        if mode == "combined_audio":
            diarization_providers = registry.list_diarization_providers()
            args.diarization_provider = select_provider(diarization_providers, "Diarization Providers")
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

    # Cleanup provider (optional)
    cleanup_providers = registry.list_cleanup_providers()
    if cleanup_providers:
        print("\nCleanup Providers (optional LLM-based transcript cleaning):")
        print("  0. None (skip cleanup)")
        for i, (name, desc) in enumerate(cleanup_providers.items(), 1):
            print(f"  {i}. {name}: {desc}")
        
        while True:
            try:
                choice = int(input("\nChoose cleanup provider (0 for none): ").strip())
                if choice == 0:
                    args.cleanup_provider = None
                    break
                elif 1 <= choice <= len(cleanup_providers):
                    args.cleanup_provider = list(cleanup_providers.keys())[choice - 1]
                    
                    # If remote provider, ask for URL
                    if args.cleanup_provider == "llama_cpp_remote":
                        url = input("Enter Llama.cpp server URL (e.g., http://192.168.1.100:8080): ").strip()
                        if url:
                            args.cleanup_url = url
                        else:
                            args.cleanup_url = "http://localhost:8080"
                    break
                else:
                    print("Invalid choice. Please enter a number from the list.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        args.cleanup_provider = None

    # Output formats
    output_writers = registry.list_output_writers()
    print("\nAvailable Output Formats:")
    for i, (name, desc) in enumerate(output_writers.items(), 1):
        print(f"  {i}. {name}: {desc}")

    print("\nEnter numbers separated by commas (e.g., 1,3,5), or press Enter for all formats:")
    choice = input("Choose output formats: ").strip()

    if not choice:
        args.selected_outputs = list(output_writers.keys())
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(output_writers)]
            args.selected_outputs = [list(output_writers.keys())[i] for i in valid_indices]
            if not args.selected_outputs:
                print("No valid choices, selecting all.")
                args.selected_outputs = list(output_writers.keys())
        except ValueError:
            print("Invalid input, selecting all.")
            args.selected_outputs = list(output_writers.keys())

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
        if hasattr(args, 'cleanup_provider') and args.cleanup_provider:
            provider_info.append(f"Cleanup={args.cleanup_provider}")
        provider_str = ", ".join(provider_info) if provider_info else "Default providers"
        print(f"\nSelected: {provider_str}, Outputs={args.selected_outputs}")
    return args