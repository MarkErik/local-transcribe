#!/usr/bin/env python3
# main.py - local-transcribe CLI runner

from __future__ import annotations
import argparse
import os
import sys
import pathlib
import warnings
from typing import Optional

# Aggressively suppress warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Set environment variables to suppress warnings from specific libraries
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ---------- repo paths & offline env ----------
def repo_root_from_here() -> pathlib.Path:
    # Resolve repo root as the directory containing this file
    return pathlib.Path(__file__).resolve().parent

def set_offline_env(models_dir: pathlib.Path) -> None:
    os.environ.setdefault("HF_HOME", str(models_dir))
    # TRANSFORMERS_CACHE is deprecated, but we keep it for backward compatibility
    os.environ.setdefault("TRANSFORMERS_CACHE", str(models_dir))
    os.environ.setdefault("PYANNOTE_CACHE", str(models_dir / "diarization"))
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
    p.add_argument("--diarization-provider", help="Diarization provider to use")
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--only-final-transcript", action="store_true", help="Only create the final merged timestamped transcript (timestamped-txt), skip other outputs.")
    p.add_argument("--interactive", action="store_true", help="Interactive mode: prompt for provider and output selections.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--create-plugin-template", choices=["asr", "diarization", "output"], help="Create a plugin template file and exit.")

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

    # ASR provider
    asr_providers = registry.list_asr_providers()
    if not asr_providers:
        print("No ASR providers available.")
        return args

    args.asr_provider = select_provider(asr_providers, "ASR Providers")

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

        print("\nDiarization Providers:")
        for name, desc in api["registry"].list_diarization_providers().items():
            print(f"  {name}: {desc}")

        print("\nOutput Writers:")
        for name, desc in api["registry"].list_output_writers().items():
            print(f"  {name}: {desc}")

        print("\nTo create a custom plugin template, use: --create-plugin-template [asr|diarization|output]")
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
    if not args.asr_provider:
        available_asr = list(api["registry"].list_asr_providers().keys())
        if available_asr:
            print(f"ERROR: --asr-provider is required. Available: {', '.join(available_asr)}")
            return 1
        else:
            print("ERROR: No ASR providers available.")
            return 1
    
    if not args.diarization_provider:
        available_diar = list(api["registry"].list_diarization_providers().keys())
        if available_diar:
            print(f"ERROR: --diarization-provider is required. Available: {', '.join(available_diar)}")
            return 1
        else:
            print("ERROR: No diarization providers available.")
            return 1

    # Get plugin providers
    try:
        asr_provider = api["registry"].get_asr_provider(args.asr_provider)
        diarization_provider = api["registry"].get_diarization_provider(args.diarization_provider)
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list-plugins to see available options.")
        return 1

    # Download required models for selected ASR provider
    required_models = asr_provider.get_required_models()
    if required_models:
        print(f"[*] Checking/downloading models for {args.asr_provider}: {', '.join(required_models)}")
        from huggingface_hub import snapshot_download
        import os
        # Temporarily allow online for downloads
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            for model in required_models:
                print(f"[*] Ensuring {model} is available...")
                # Check if already downloaded
                safe_name = f"models--{model.replace('/', '--')}"
                model_path = models_dir / "asr" / "ct2" / safe_name / "snapshots"
                if model_path.exists() and any(model_path.iterdir()):
                    print(f"[✓] {model} already available locally.")
                    continue
                snapshot_download(model, cache_dir=str(models_dir / "asr" / "ct2"))
            print("[✓] All required models are ready.")
        except Exception as e:
            print(f"ERROR: Failed to download models: {e}")
            return 1
        finally:
            # Restore offline mode
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Configure logging based on verbose flag
    api["configure_global_logging"](log_level="INFO" if args.verbose else "WARNING")

    # Initialize progress tracking
    tracker = api["get_progress_tracker"]()
    tracker.start()

    try:
        # Ensure outdir & subdirs
        outdir = ensure_outdir(args.outdir)
        paths = api["ensure_session_dirs"](outdir, mode)

        print(f"[*] Mode: {mode} | ASR: {args.asr_provider} | Diarization: {args.diarization_provider} | Outputs: {', '.join(args.selected_outputs)}")

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

            # 2) ASR + alignment
            words = asr_provider.transcribe_with_alignment(
                str(std_mix),
                role=None,
                asr_model=None
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
                asr_model=None
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
                asr_model=None
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
            from local_transcribe.processing.merge import merge_turn_streams
            merged = merge_turn_streams(int_turns, part_turns)
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
