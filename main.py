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
    p.add_argument("--asr", default="large-v3", help="ASR model to use (provider-specific)")
    p.add_argument("--asr-provider", help="ASR provider to use")
    p.add_argument("--diarization-provider", help="Diarization provider to use")
    p.add_argument("-o", "--outdir", metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--write-vtt", action="store_true", help="Also write WebVTT alongside SRT.")
    p.add_argument("--render-black", action="store_true", help="Render a black MP4 with burned-in subtitles (uses SRT).")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging output.")
    p.add_argument("--list-plugins", action="store_true", help="List available plugins and exit.")
    p.add_argument("--create-plugin-template", choices=["asr", "diarization"], help="Create a plugin template file and exit.")

    args = p.parse_args(argv)

    # Validate required arguments when not listing plugins or creating templates
    if not args.list_plugins and not args.create_plugin_template:
        if not args.outdir:
            p.error("-o/--outdir is required")
        if not args.combined and not args.interviewer:
            p.error("Must provide either -c/--combined or -i/--interviewer")

    return args

# ---------- main ----------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Handle plugin template creation
    if args.create_plugin_template:
        # Need to set up paths first
        root = repo_root_from_here()
        sys.path.append(str(root / "src"))
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
        # Need to import and load plugins to list them
        root = repo_root_from_here()
        models_dir = root / "models"
        set_offline_env(models_dir)
        api = import_pipeline_modules(root)

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

        print("\nTo create a custom plugin template, use: --create-plugin-template [asr|diarization]")
        return 0

    # Validate dual vs combined (only when not listing plugins)
    dual_mode = args.interviewer is not None
    combined_mode = args.combined is not None
    if dual_mode:
        if not args.participant:
            sys.exit("ERROR: Dual-track mode requires both -i/--interviewer and -p/--participant.")
        if combined_mode:
            sys.exit("ERROR: Provide either -c/--combined OR -i/--p, not both.")
    elif not combined_mode:
        sys.exit("ERROR: Must provide either -c/--combined or -i/--interviewer.")
    mode = "combined" if combined_mode else "dual_track"

    # Resolve repo & models, enforce offline
    root = repo_root_from_here()
    models_dir = root / "models"
    set_offline_env(models_dir)
    ensure_models_exist(models_dir)

    # Import pipeline functions after sys.path setup
    api = import_pipeline_modules(root)

    # Check required providers
    if not args.asr_provider:
        available_asr = list(api["registry"].list_asr_providers().keys())
        if available_asr:
            sys.exit(f"ERROR: --asr-provider is required. Available: {', '.join(available_asr)}")
        else:
            sys.exit("ERROR: No ASR providers available.")
    
    if not args.diarization_provider:
        available_diar = list(api["registry"].list_diarization_providers().keys())
        if available_diar:
            sys.exit(f"ERROR: --diarization-provider is required. Available: {', '.join(available_diar)}")
        else:
            sys.exit("ERROR: No diarization providers available.")

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
        os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            for model in required_models:
                print(f"[*] Ensuring {model} is available...")
                snapshot_download(model, cache_dir=str(models_dir))
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

        print(f"[*] Mode: {mode} | ASR: {args.asr_provider} ({args.asr}) | Diarization: {args.diarization_provider}")

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
                asr_model=args.asr
            )

            # Save ASR results as plain text before diarization
            # TODO: Use plugin for this too
            from local_transcribe.providers.writers.txt_writer import write_asr_words
            write_asr_words(words, paths["merged"] / "asr.txt")

            # 3) Diarize → turns
            turns = diarization_provider.diarize(str(std_mix), words)

            # 4) Outputs
            output_task = tracker.add_task("Writing output files", total=100, stage="output")
            tracker.update(output_task, advance=20, description="Writing transcript files")

            # Get output writers
            timestamped_writer = api["registry"].get_output_writer("timestamped-txt")
            plain_writer = api["registry"].get_output_writer("plain-txt")
            csv_writer = api["registry"].get_output_writer("csv")
            markdown_writer = api["registry"].get_output_writer("markdown")
            srt_writer = api["registry"].get_output_writer("srt")

            timestamped_writer.write(turns, paths["merged"] / "transcript.timestamped.txt")
            plain_writer.write(turns, paths["merged"] / "transcript.txt")
            csv_writer.write(turns, paths["merged"] / "transcript.csv")
            markdown_writer.write(turns, paths["merged"] / "transcript.md")

            tracker.update(output_task, advance=20, description="Writing subtitle files")
            srt_path = paths["merged"] / "subtitles.srt"
            srt_writer.write(turns, srt_path)
            if args.write_vtt:
                vtt_writer = api["registry"].get_output_writer("vtt")
                vtt_writer.write(turns, paths["merged"] / "subtitles.vtt")

            if args.render_black:
                tracker.update(output_task, advance=30, description="Rendering video with subtitles")
                # TODO: Use plugin for video rendering
                from local_transcribe.providers.writers.render_black import render_black_video
                render_black_video(srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=std_mix)
            else:
                tracker.update(output_task, advance=30, description="Skipping video rendering")

            tracker.update(output_task, advance=30, description="Finalizing outputs")
            tracker.complete_task(output_task, stage="output")

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
                asr_model=args.asr
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
                asr_model=args.asr
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
            output_task = tracker.add_task("Writing output files", total=100, stage="output")

            # Merged outputs
            tracker.update(output_task, advance=15, description="Writing merged transcripts")
            timestamped_writer.write(merged, paths["merged"] / "transcript.timestamped.txt")
            plain_writer = api["registry"].get_output_writer("plain-txt")
            plain_writer.write(merged, paths["merged"] / "transcript.txt")
            csv_writer = api["registry"].get_output_writer("csv")
            csv_writer.write(merged, paths["merged"] / "transcript.csv")
            markdown_writer = api["registry"].get_output_writer("markdown")
            markdown_writer.write(merged, paths["merged"] / "transcript.md")

            tracker.update(output_task, advance=15, description="Writing subtitle files")
            srt_path = paths["merged"] / "subtitles.srt"
            srt_writer = api["registry"].get_output_writer("srt")
            srt_writer.write(merged, srt_path)
            if args.write_vtt:
                vtt_writer = api["registry"].get_output_writer("vtt")
                vtt_writer.write(merged, paths["merged"] / "subtitles.vtt")

            if args.render_black:
                tracker.update(output_task, advance=30, description="Rendering video with subtitles")
                # TODO: Use plugin for video rendering
                from local_transcribe.providers.writers.render_black import render_black_video
                # Pass both interviewer and participant audio for dual-track mode
                render_black_video(srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=[std_int, std_part])
            else:
                tracker.update(output_task, advance=30, description="Skipping video rendering")

            tracker.update(output_task, advance=20, description="Finalizing outputs")
            tracker.complete_task(output_task, stage="output")

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
