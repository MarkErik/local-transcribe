#!/usr/bin/env python3
# main.py - local-transcribe CLI runner

from __future__ import annotations
import argparse
import os
import sys
import pathlib
import warnings
from typing import Optional
from dotenv import load_dotenv

# Aggressively suppress warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Set environment variables to suppress warnings from specific libraries
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Load environment variables from .env file
load_dotenv()

# ---------- repo paths & offline env ----------
def repo_root_from_here() -> pathlib.Path:
    # Resolve repo root as the directory containing this file
    return pathlib.Path(__file__).resolve().parent

def set_offline_env(models_dir: pathlib.Path) -> None:
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(models_dir / ".xdg"))
    # Runtime must be fully offline (models must already be downloaded)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------- simple checks ----------
def ensure_models_exist(models_dir: pathlib.Path) -> None:
    if not models_dir.exists():
        print("WARNING: models/ directory not found. Models will be downloaded automatically on first run.")
        models_dir.mkdir(parents=True, exist_ok=True)
    # Best-effort check: confirm something for ASR exists in cache
    expected = models_dir / "asr"
    if not expected.exists():
        print("WARNING: ASR models not found in ./models/asr. Models will be downloaded automatically on first run.")
    # We won't strictly validate HF cache layout; downloader guarantees presence.

def ensure_file(path: str, label: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        sys.exit(f"ERROR: {label} file not found: {p}")
    return p

def ensure_outdir(path: str) -> pathlib.Path:
    out = pathlib.Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out

# ---------- import pipeline modules ----------
def import_pipeline_modules(repo_root: pathlib.Path):
    sys.path.append(str(repo_root / "local_transcribe"))
    try:
        # Core helpers (keep these as direct imports since they're utilities)
        from local_transcribe.lib.create_directories import ensure_session_dirs
        from local_transcribe.lib.audio_io import standardize_and_get_path
        from local_transcribe.lib.progress import get_progress_tracker
        from local_transcribe.lib.logging_config import configure_global_logging

        # Import core plugin system
        from local_transcribe.framework import registry
        from local_transcribe.framework.plugin_discovery import PluginLoader

        # Import providers to register plugins
        import local_transcribe.providers

        # Load external plugins
        plugin_loader = PluginLoader()
        plugin_loader.load_all_plugins()

        # Return both utilities and registry
        return {
            "ensure_session_dirs": ensure_session_dirs,
            "standardize_and_get_path": standardize_and_get_path,
            "get_progress_tracker": get_progress_tracker,
            "configure_global_logging": configure_global_logging,
            "registry": registry,
        }
    except Exception as e:
        sys.exit(f"ERROR: Failed importing pipeline modules from local_transcribe/: {e}")

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
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--interactive", action="store_true", help="Interactive mode: prompt for provider and output selections.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--create-plugin-template", choices=["asr", "diarization", "combined", "output"], help="Create a plugin template file and exit.")

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
    else:
        # Separate providers
        # ASR provider
        asr_providers = registry.list_asr_providers()
        if not asr_providers:
            print("No ASR providers available.")
            return args

        args.asr_provider = select_provider(asr_providers, "ASR Providers")

        # ASR model selection (if provider supports multiple models)
        asr_provider = registry.get_asr_provider(args.asr_provider)
        available_models = asr_provider.get_available_models()
        if len(available_models) > 1:
            print(f"\nAvailable models for {args.asr_provider}:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            
            while True:
                try:
                    choice = int(input(f"\nChoose model (1-{len(available_models)}): ").strip())
                    if 1 <= choice <= len(available_models):
                        args.asr_model = available_models[choice - 1]
                        break
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            args.asr_model = available_models[0] if available_models else None

        # Diarization provider
        diar_providers = registry.list_diarization_providers()
        args.diarization_provider = select_provider(diar_providers, "Diarization Providers")

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

def write_selected_outputs(turns, paths, selected, tracker, registry, audio_path=None):
    """Write selected outputs for merged turns."""
    output_task = tracker.add_task("Writing output files", total=100, stage="output")
    tracker.update(output_task, advance=20, description="Writing transcript files")

    srt_path = None

    if 'timestamped-txt' in selected:
        timestamped_writer = registry.get_output_writer("timestamped-txt")
        timestamped_writer.write(turns, paths["merged"] / "transcript.timestamped.txt")

    if 'plain-txt' in selected:
        plain_writer = registry.get_output_writer("plain-txt")
        plain_writer.write(turns, paths["merged"] / "transcript.txt")

    if 'csv' in selected:
        csv_writer = registry.get_output_writer("csv")
        csv_writer.write(turns, paths["merged"] / "transcript.csv")

    if 'markdown' in selected:
        markdown_writer = registry.get_output_writer("markdown")
        markdown_writer.write(turns, paths["merged"] / "transcript.md")

    tracker.update(output_task, advance=20, description="Writing subtitle files")

    if 'srt' in selected:
        srt_writer = registry.get_output_writer("srt")
        srt_path = paths["merged"] / "subtitles.srt"
        srt_writer.write(turns, srt_path)

    if 'vtt' in selected:
        vtt_writer = registry.get_output_writer("vtt")
        vtt_writer.write(turns, paths["merged"] / "subtitles.vtt")

    if srt_path and audio_path is not None:
        tracker.update(output_task, advance=30, description="Rendering video with subtitles")
        from local_transcribe.providers.writers.render_black import render_black_video
        render_black_video(srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=audio_path)
    else:
        tracker.update(output_task, advance=30, description="Skipping video rendering")

    tracker.update(output_task, advance=30, description="Finalizing outputs")
    tracker.complete_task(output_task, stage="output")

# ---------- main ----------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Early validation for required args
    if not args.list_plugins and not args.create_plugin_template:
        if not args.outdir:
            print("ERROR: -o/--outdir is required")
            return 1
        if not args.combined and not args.interviewer:
            print("ERROR: Must provide either -c/--combined or -i/--interviewer")
            return 1

    root = repo_root_from_here()
    models_dir = root / "models"
    set_offline_env(models_dir)
    api = import_pipeline_modules(root)

    # Handle plugin template creation
    if args.create_plugin_template:
        from local_transcribe.framework.plugin_discovery import create_plugin_template
        from pathlib import Path

        # Create template in current directory
        template_name = f"example_{args.create_plugin_template}_plugin.py"
        template_path = Path.cwd() / template_name
        create_plugin_template(template_path, args.create_plugin_template)
        print(f"Plugin template created: {template_path}")
        print("Edit the file and place it in your plugins directory to use it.")
        return 0

    # Handle plugin listing (doesn't require other args)
    if args.list_plugins:
        print("Available Plugins:")
        print("\nASR Providers:")
        for name, desc in api["registry"].list_asr_providers().items():
            print(f"  {name}: {desc}")
            # Show available models if provider supports multiple
            provider = api["registry"].get_asr_provider(name)
            available_models = provider.get_available_models()
            if len(available_models) > 1:
                print(f"    Available models: {', '.join(available_models)}")

        print("\nDiarization Providers:")
        for name, desc in api["registry"].list_diarization_providers().items():
            print(f"  {name}: {desc}")

        print("\nCombined Providers:")
        for name, desc in api["registry"].list_combined_providers().items():
            print(f"  {name}: {desc}")
            # Show available models if provider supports multiple
            provider = api["registry"].get_combined_provider(name)
            available_models = provider.get_available_models()
            if len(available_models) > 1:
                print(f"    Available models: {', '.join(available_models)}")

        print("\nOutput Writers:")
        for name, desc in api["registry"].list_output_writers().items():
            print(f"  {name}: {desc}")

        print("\nTo create a custom plugin template, use: --create-plugin-template [asr|diarization|combined|output]")
        return 0

    if args.interactive:
        args = interactive_prompt(args, api)

    # Set default outputs for non-interactive
    if not hasattr(args, 'selected_outputs') or not args.selected_outputs:
        if args.only_final_transcript:
            args.selected_outputs = ['timestamped-txt']
        else:
            args.selected_outputs = list(api["registry"].list_output_writers().keys())

    # Validate dual vs combined
    dual_mode = args.interviewer is not None
    combined_mode = args.combined is not None
    if dual_mode:
        if not args.participant:
            print("ERROR: Dual-track mode requires both -i/--interviewer and -p/--participant.")
            return 1
        if combined_mode:
            print("ERROR: Provide either -c/--combined OR -i/--p, not both.")
            return 1
    elif not combined_mode:
        print("ERROR: Must provide either -c/--combined or -i/--interviewer")
        return 1
    mode = "combined" if combined_mode else "dual_track"

    # Check required providers
    try:
        if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
            combined_provider = api["registry"].get_combined_provider(args.combined_provider)
        else:
            asr_provider = api["registry"].get_asr_provider(args.asr_provider)
            diarization_provider = api["registry"].get_diarization_provider(args.diarization_provider)
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list-plugins to see available options.")
        return 1

    # Set default model if not specified
    if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
        if not hasattr(args, 'combined_model') or args.combined_model is None:
            available_models = combined_provider.get_available_models()
            args.combined_model = available_models[0] if available_models else None
    else:
        if not hasattr(args, 'asr_model') or args.asr_model is None:
            available_models = asr_provider.get_available_models()
            args.asr_model = available_models[0] if available_models else None

    # Download required models for selected providers
    # Phase 2: Model Availability Check (Offline)
    if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
        required_combined_models = combined_provider.get_required_models(args.combined_model)
        print(f"[*] Checking model availability offline...")
        missing_combined_models = combined_provider.check_models_available_offline(required_combined_models, models_dir)
        all_missing = missing_combined_models
    else:
        required_asr_models = asr_provider.get_required_models(args.asr_model)
        required_diarization_models = diarization_provider.get_required_models()
        
        print(f"[*] Checking model availability offline...")
        missing_asr_models = []
        missing_diarization_models = []
        
        if required_asr_models:
            missing_asr_models = asr_provider.check_models_available_offline(required_asr_models, models_dir)
        
        if required_diarization_models:
            missing_diarization_models = diarization_provider.check_models_available_offline(required_diarization_models, models_dir)
        
        all_missing = missing_asr_models + missing_diarization_models

    # Phase 3: Conditional Download (Online Only If Needed)
    if all_missing:
        print(f"[!] Missing models detected: {', '.join(all_missing)}")
        print("[!] Switching to online mode for download...")
        
        # Check if HF_TOKEN is available
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("[!] WARNING: HF_TOKEN not found in environment variables.")
            print("[!] Please ensure your .env file contains a valid HuggingFace token.")
            print("[!] You can get a token from: https://huggingface.co/settings/tokens")
            print("[!] Alternatively, run: huggingface-cli login")
            print("[!] Make sure the token has 'read' permissions for the required models")
        
        # Explicitly set online mode
        print(f"DEBUG: Setting HF_HUB_OFFLINE to 0 (was: {os.environ.get('HF_HUB_OFFLINE')})")
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        # Force reload of huggingface_hub modules to pick up new environment
        import sys
        print(f"DEBUG: Reloading huggingface_hub modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            print(f"DEBUG: Reloaded {module_name}")
        
        # Verify the change took effect immediately
        print(f"DEBUG: Immediate verification - HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
        
        # Show current environment for debugging
        print(f"[*] Environment check:")
        print(f"    HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"    HF_TOKEN: {'***' if hf_token else 'NOT SET'}")
        print(f"    HF_HOME: {os.environ.get('HF_HOME')}")
        
        # Additional debug info
        print(f"DEBUG: All Hugging Face environment variables:")
        for key, value in os.environ.items():
            if key.startswith('HF_'):
                print(f"    {key}: {'***' if 'TOKEN' in key else value}")
        
        # Test huggingface_hub import and check its view of environment
        try:
            from huggingface_hub import HfApi
            print(f"DEBUG: HfApi().whoami() (tests token): {HfApi().whoami()}")
        except Exception as e:
            print(f"DEBUG: HfApi test failed: {e}")
        
        try:
            # Download missing models
            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                if missing_combined_models:
                    print(f"[*] Downloading combined models: {', '.join(missing_combined_models)}")
                    combined_provider.ensure_models_available(missing_combined_models, models_dir)
            else:
                if missing_asr_models:
                    print(f"[*] Downloading ASR models: {', '.join(missing_asr_models)}")
                    asr_provider.ensure_models_available(missing_asr_models, models_dir)
                
                if missing_diarization_models:
                    print(f"[*] Downloading diarization models: {', '.join(missing_diarization_models)}")
                    diarization_provider.ensure_models_available(missing_diarization_models, models_dir)
            
            # Verify downloads actually succeeded
            print("[*] Verifying downloaded models...")
            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                verified_combined = combined_provider.check_models_available_offline(required_combined_models, models_dir)
                if verified_combined:
                    print(f"[!] ERROR: Some models failed to download properly: {', '.join(verified_combined)}")
                    print("[!] Please check your internet connection and HuggingFace token.")
                    return 1
            else:
                verified_asr = asr_provider.check_models_available_offline(required_asr_models, models_dir)
                verified_diar = diarization_provider.check_models_available_offline(required_diarization_models, models_dir)
                
                if verified_asr or verified_diar:
                    print(f"[!] ERROR: Some models failed to download properly: {', '.join(verified_asr + verified_diar)}")
                    print("[!] Please check your internet connection and HuggingFace token.")
                    return 1
            
            print("[✓] All models downloaded successfully and verified.")
        except Exception as e:
            print(f"ERROR: Failed to download models: {e}")
            return 1
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("[✓] All required models are available locally.")

    # Configure logging based on verbose flag
    api["configure_global_logging"](log_level="INFO" if args.verbose else "WARNING")

    # Initialize progress tracking
    tracker = api["get_progress_tracker"]()
    tracker.start()

    try:
        # Ensure outdir & subdirs
        outdir = ensure_outdir(args.outdir)
        paths = api["ensure_session_dirs"](outdir, mode)

        print(f"[*] Mode: {mode} (combined) | Provider: {args.combined_provider} | Outputs: {', '.join(args.selected_outputs)}")

        # Run pipeline
        if combined_mode:
            mixed_path = ensure_file(args.combined, "Combined")

            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)

            # Standardize mixed audio
            tracker.update(std_task, advance=50, description="Standardizing mixed audio")
            std_mix = api["standardize_and_get_path"](mixed_path, tmpdir=temp_audio_dir)

            # Complete the standardization task
            tracker.update(std_task, advance=50, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")

            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                # Use combined provider
                turns = combined_provider.transcribe_and_diarize(
                    str(std_mix),
                    model=args.combined_model
                )
            else:
                # 2) ASR + alignment
                words = asr_provider.transcribe_with_alignment(
                    str(std_mix),
                    role=None,
                    asr_model=args.asr_model
                )

                # Save ASR results as plain text before diarization
                # TODO: Use plugin for this too
                from local_transcribe.providers.writers.txt_writer import write_asr_words
                write_asr_words(words, paths["merged"] / "asr.txt")

                # 3) Diarize → turns
                turns = diarization_provider.diarize(str.std_mix, words)

            # 4) Outputs
            write_selected_outputs(turns, paths, args.selected_outputs, tracker, api["registry"], std_mix)

            print("[✓] Combined processing complete.")

        else:
            # dual-track
            interviewer_path = ensure_file(args.interviewer, "Interviewer")
            participant_path = ensure_file(args.participant, "Participant")

            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)

            # Standardize interviewer audio
            tracker.update(std_task, advance=33, description="Standardizing interviewer audio")
            std_int = api["standardize_and_get_path"](interviewer_path, tmpdir=temp_audio_dir)

            # Standardize participant audio
            tracker.update(std_task, advance=33, description="Standardizing participant audio")
            std_part = api["standardize_and_get_path"](participant_path, tmpdir=temp_audio_dir)

            # Complete the standardization task
            tracker.update(std_task, advance=34, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")

            # 2) ASR + alignment per track
            asr_task = tracker.add_task("ASR Transcription", total=100, stage="asr_transcription")
            tracker.update(asr_task, advance=20, description="Transcribing interviewer audio")
            int_words = asr_provider.transcribe_with_alignment(
                str(std_int),
                role="Interviewer",
                asr_model=args.asr_model
            )

            # Save ASR results and build turns for interviewer
            from local_transcribe.providers.writers.txt_writer import write_asr_words
            write_asr_words(int_words, paths["speaker_interviewer"] / "asr.txt")

            # Use dual-track diarization provider for building turns
            dual_track_provider = api["registry"].get_diarization_provider("dual-track")
            int_turns = dual_track_provider.diarize(
                str(std_int),
                int_words,
                speaker_label="Interviewer"
            )

            # Save interviewer timestamped transcript
            timestamped_writer = api["registry"].get_output_writer("timestamped-txt")
            timestamped_writer.write(int_turns, paths["speaker_interviewer"] / "interviewer.timestamped.txt")
            tracker.update(asr_task, advance=30, description="Interviewer ASR and timestamped transcript complete")

            tracker.update(asr_task, advance=20, description="Transcribing participant audio")
            part_words = asr_provider.transcribe_with_alignment(
                str(std_part),
                role="Participant",
                asr_model=args.asr_model
            )

            write_asr_words(part_words, paths["speaker_participant"] / "asr.txt")

            # Build turns for participant
            part_turns = dual_track_provider.diarize(
                str(std_part),
                part_words,
                speaker_label="Participant"
            )

            # Save participant timestamped transcript
            timestamped_writer.write(part_turns, paths["speaker_participant"] / "participant.timestamped.txt")
            tracker.update(asr_task, advance=30, description="Participant ASR and timestamped transcript complete")
            tracker.complete_task(asr_task, stage="asr_transcription")

            # 3) Merge turns
            turns_task = tracker.add_task("Merging conversation turns", total=100, stage="turns")
            tracker.update(turns_task, advance=50, description="Merging conversation turns")
            # TODO: Use plugin for merging
            from local_transcribe.processing.merge import merge_turns
            merged = merge_turns(int_turns, part_turns)
            tracker.update(turns_task, advance=50, description="Turn merging complete")
            tracker.complete_task(turns_task, stage="turns")

            # 4) Write results
            write_selected_outputs(merged, paths, args.selected_outputs, tracker, api["registry"], [std_int, std_part])

            print("[✓] Dual-track processing complete.")

        # Summary
        print(f"[i] Artifacts written to: {paths['root']}")
        
        # Print performance summary
        tracker.print_summary()
        
        return 0
        
    finally:
        # Always stop progress tracking
        tracker.stop()

if __name__ == "__main__":
    raise SystemExit(main())
