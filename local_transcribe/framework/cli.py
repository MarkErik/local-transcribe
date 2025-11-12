#!/usr/bin/env python3
# framework/cli.py - CLI argument parsing and interactive prompts

import argparse
from typing import Optional

# ---------- CLI ----------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="local-transcribe: batch transcription (dual-track or combined) – offline, Apple Silicon friendly."
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("-c", "--combined", metavar="MIXED_AUDIO", help="Process a single mixed/combined audio file.")
    mode.add_argument("-i", "--interviewer", metavar="INTERVIEWER_AUDIO", help="Interviewer track for dual-track mode.")
    p.add_argument("-p", "--participant", metavar="PARTICIPANT_AUDIO", help="Participant track for dual-track mode.")
    p.add_argument("--asr-provider", help="ASR provider to use")
    p.add_argument("--asr-model", help="ASR model to use (if provider supports multiple models)")
    p.add_argument("--diarization-provider", help="Diarization provider to use")
    p.add_argument("--num-speakers", type=int, help="Number of speakers expected in the audio (for diarization)")
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--interactive", action="store_true", help="Interactive mode: prompt for provider and output selections.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--create-plugin-template", choices=["asr", "diarization", "combined", "turn_builder", "output"], help="Create a plugin template file and exit.")

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

    # Check if combined mode and offer processing type choice
    if hasattr(args, 'combined') and args.combined:
        combined_providers = registry.list_combined_providers()
        if combined_providers:
            print("Processing Modes:")
            print("  1. Combined Provider (ASR + Diarization in one step)")
            print("  2. Separate Providers (ASR then Diarization)")
            
            while True:
                try:
                    choice = int(input("\nChoose processing mode (1 or 2): ").strip())
                    if choice == 1:
                        args.processing_mode = "combined"
                        break
                    elif choice == 2:
                        args.processing_mode = "separate"
                        break
                    else:
                        print("Invalid choice. Please enter 1 or 2.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            print("No combined providers available, using separate providers.")
            args.processing_mode = "separate"
    else:
        args.processing_mode = "separate"  # For dual-track, always separate

    if args.processing_mode == "combined":
        # Select combined provider
        combined_providers = registry.list_combined_providers()
        args.combined_provider = select_provider(combined_providers, "Combined Providers")

        # Model selection for combined provider
        combined_provider = registry.get_combined_provider(args.combined_provider)
        available_models = combined_provider.get_available_models()
        if len(available_models) > 1:
            print(f"\nAvailable models for {args.combined_provider}:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            
            while True:
                try:
                    choice = int(input(f"\nChoose model (1-{len(available_models)}): ").strip())
                    if 1 <= choice <= len(available_models):
                        args.combined_model = available_models[choice - 1]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            args.combined_model = available_models[0] if available_models else None

        # Number of speakers (for combined providers)
        if hasattr(args, 'combined') and args.combined:
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

    if args.processing_mode == "combined":
        print(f"\nSelected: Combined={args.combined_provider} ({args.combined_model}), Outputs={args.selected_outputs}")
    else:
        print(f"\nSelected: ASR={args.asr_provider}, Diarization={args.diarization_provider}, Outputs={args.selected_outputs}")
    return args