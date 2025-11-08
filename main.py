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
def ensure_models_exist(models_dir: pathlib.Path, asr_model: str) -> None:
    if not models_dir.exists():
        sys.exit("ERROR: models/ directory not found. Run scripts/download_models.py first.")
    asr_map = {
        "medium.en": "openai/whisper-medium.en",
        "large-v3-turbo": "openai/whisper-large-v3-turbo",
    }
    # Best-effort check: confirm something for ASR exists in cache
    expected = models_dir / "asr"
    if not expected.exists():
        sys.exit("ERROR: ASR models not found in ./models/asr. Run scripts/download_models.py first.")
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
    sys.path.append(str(repo_root / "src"))
    try:
        # Core helpers
        from core.session import ensure_session_dirs
        from core.audio_io import standardize_and_get_path
        from core.asr import transcribe_with_alignment
        from core.progress import get_progress_tracker

        # Dual-track helpers
        from dual_track.turns import build_turns
        from dual_track.merge import merge_turn_streams

        # Combined helpers
        from combined.diarize import diarize_mixed

        # Output writers
        from output.srt_vtt import write_srt, write_vtt
        from output.txt_writer import write_timestamped_txt, write_plain_txt, write_asr_words
        from output.csv_writer import write_conversation_csv
        from output.markdown_writer import write_conversation_markdown
        from output.render_black import render_black_video
    except Exception as e:
        sys.exit(f"ERROR: Failed importing pipeline modules from src/: {e}")
    return {
        "ensure_session_dirs": ensure_session_dirs,
        "standardize_and_get_path": standardize_and_get_path,
        "transcribe_with_alignment": transcribe_with_alignment,
        "build_turns": build_turns,
        "merge_turn_streams": merge_turn_streams,
        "write_srt": write_srt,
        "write_vtt": write_vtt,
        "write_timestamped_txt": write_timestamped_txt,
        "write_plain_txt": write_plain_txt,
        "write_asr_words": write_asr_words,
        "write_conversation_csv": write_conversation_csv,
        "write_conversation_markdown": write_conversation_markdown,
        "render_black_video": render_black_video,
        "diarize_mixed": diarize_mixed,
        "get_progress_tracker": get_progress_tracker,
    }

# ---------- CLI ----------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="local-transcribe: batch transcription (dual-track or combined) – offline, Apple Silicon friendly."
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("-c", "--combined", metavar="MIXED_AUDIO", help="Process a single mixed/combined audio file.")
    mode.add_argument("-i", "--interviewer", metavar="INTERVIEWER_AUDIO", help="Interviewer track for dual-track mode.")
    p.add_argument("-p", "--participant", metavar="PARTICIPANT_AUDIO", help="Participant track for dual-track mode.")
    p.add_argument("--asr", choices=("medium.en", "large-v3-turbo"), default="medium.en", help="ASR model to use.")
    p.add_argument("--outdir", required=True, metavar="OUTPUT_DIR", help="Directory to write outputs into (created if missing).")
    p.add_argument("--write-vtt", action="store_true", help="Also write WebVTT alongside SRT.")
    p.add_argument("--render-black", action="store_true", help="Render a black MP4 with burned-in subtitles (uses SRT).")
    return p.parse_args(argv)

# ---------- main ----------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Validate dual vs combined
    dual_mode = args.interviewer is not None
    combined_mode = args.combined is not None
    if dual_mode:
        if not args.participant:
            sys.exit("ERROR: Dual-track mode requires both -i/--interviewer and -p/--participant.")
        if combined_mode:
            sys.exit("ERROR: Provide either -c/--combined OR -i/-p, not both.")

    # Resolve repo & models, enforce offline
    root = repo_root_from_here()
    models_dir = root / "models"
    set_offline_env(models_dir)
    ensure_models_exist(models_dir, args.asr)

    # Import pipeline functions after sys.path setup
    api = import_pipeline_modules(root)
    
    # Configure logging to DEBUG level for detailed output
    from core.logging_config import configure_global_logging
    configure_global_logging(log_level="DEBUG")

    # Initialize progress tracking
    tracker = api["get_progress_tracker"]()
    tracker.start()

    try:
        # Ensure outdir & subdirs
        outdir = ensure_outdir(args.outdir)
        paths = api["ensure_session_dirs"](outdir)

        # Run pipeline
        if combined_mode:
            mixed_path = ensure_file(args.combined, "Combined")
            print(f"[*] Mode: combined | ASR: {args.asr}")
            
            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            tracker.update(std_task, advance=50, description="Standardizing mixed audio")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)
            std_mix = api["standardize_and_get_path"](mixed_path, tmpdir=temp_audio_dir)
            tracker.update(std_task, advance=50, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")
            
            # 2) ASR + alignment
            words = api["transcribe_with_alignment"](str(std_mix), asr_model=args.asr, role=None)
            
            # Save ASR results as plain text before diarization
            api["write_asr_words"](words, paths["merged"] / "asr.txt")
            
            # 3) Diarize → turns
            turns = api["diarize_mixed"](str(std_mix), words)
            
            # 4) Outputs
            output_task = tracker.add_task("Writing output files", total=100, stage="output")
            tracker.update(output_task, advance=20, description="Writing transcript files")
            api["write_timestamped_txt"](turns, paths["merged"] / "transcript.timestamped.txt")
            api["write_plain_txt"](turns, paths["merged"] / "transcript.txt")
            api["write_conversation_csv"](turns, paths["merged"] / "transcript.csv")
            api["write_conversation_markdown"](turns, paths["merged"] / "transcript.md")
            
            tracker.update(output_task, advance=20, description="Writing subtitle files")
            srt_path = paths["merged"] / "subtitles.srt"
            api["write_srt"](turns, srt_path)
            if args.write_vtt:
                api["write_vtt"](turns, paths["merged"] / "subtitles.vtt")
            
            if args.render_black:
                tracker.update(output_task, advance=30, description="Rendering video with subtitles")
                api["render_black_video"](srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=std_mix)
            else:
                tracker.update(output_task, advance=30, description="Skipping video rendering")
            
            tracker.update(output_task, advance=30, description="Finalizing outputs")
            tracker.complete_task(output_task, stage="output")
            
            print("[✓] Combined processing complete.")

        else:
            # dual-track
            interviewer_path = ensure_file(args.interviewer, "Interviewer")
            participant_path = ensure_file(args.participant, "Participant")
            print(f"[*] Mode: dual-track | ASR: {args.asr}")
            
            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)
            tracker.update(std_task, advance=33, description="Standardizing interviewer audio")
            std_int = api["standardize_and_get_path"](interviewer_path, tmpdir=temp_audio_dir)
            tracker.update(std_task, advance=33, description="Standardizing participant audio")
            std_part = api["standardize_and_get_path"](participant_path, tmpdir=temp_audio_dir)
            tracker.update(std_task, advance=34, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")
            
            # 2) ASR + alignment per track
            int_words = api["transcribe_with_alignment"](str(std_int), asr_model=args.asr, role="Interviewer")
            part_words = api["transcribe_with_alignment"](str(std_part), asr_model=args.asr, role="Participant")
            
            # 3) Turns per track
            turns_task = tracker.add_task("Building conversation turns", total=100, stage="turns")
            tracker.update(turns_task, advance=30, description="Building interviewer turns")
            int_turns = api["build_turns"](int_words, speaker_label="Interviewer")
            tracker.update(turns_task, advance=30, description="Building participant turns")
            part_turns = api["build_turns"](part_words, speaker_label="Participant")
            
            # 4) Merge turns
            tracker.update(turns_task, advance=20, description="Merging conversation turns")
            merged = api["merge_turn_streams"](int_turns, part_turns)
            tracker.update(turns_task, advance=20, description="Turn building complete")
            tracker.complete_task(turns_task, stage="turns")
            
            # 5) Per-speaker outputs
            output_task = tracker.add_task("Writing output files", total=100, stage="output")
            
            tracker.update(output_task, advance=10, description="Writing interviewer transcripts")
            api["write_timestamped_txt"](int_turns, paths["speaker_interviewer"] / "interviewer.timestamped.txt")
            api["write_plain_txt"](int_turns,        paths["speaker_interviewer"] / "interviewer.txt")
            
            tracker.update(output_task, advance=10, description="Writing participant transcripts")
            api["write_timestamped_txt"](part_turns, paths["speaker_participant"] / "participant.timestamped.txt")
            api["write_plain_txt"](part_turns,       paths["speaker_participant"] / "participant.txt")
            
            # 6) Merged outputs
            tracker.update(output_task, advance=15, description="Writing merged transcripts")
            api["write_timestamped_txt"](merged, paths["merged"] / "transcript.timestamped.txt")
            api["write_plain_txt"](merged,       paths["merged"] / "transcript.txt")
            api["write_conversation_csv"](merged, paths["merged"] / "transcript.csv")
            api["write_conversation_markdown"](merged, paths["merged"] / "transcript.md")
            
            tracker.update(output_task, advance=15, description="Writing subtitle files")
            srt_path = paths["merged"] / "subtitles.srt"
            api["write_srt"](merged, srt_path)
            if args.write_vtt:
                api["write_vtt"](merged, paths["merged"] / "subtitles.vtt")
            
            if args.render_black:
                tracker.update(output_task, advance=30, description="Rendering video with subtitles")
                # Pass both interviewer and participant audio for dual-track mode
                api["render_black_video"](srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=[std_int, std_part])
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
