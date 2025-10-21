#!/usr/bin/env python3
# main.py - local-transcribe CLI runner

from __future__ import annotations
import argparse
import os
import sys
import pathlib
from typing import Optional

# ---------- repo paths & offline env ----------
def repo_root_from_here() -> pathlib.Path:
    # Resolve repo root as the directory containing this file
    return pathlib.Path(__file__).resolve().parent

def set_offline_env(models_dir: pathlib.Path) -> None:
    os.environ.setdefault("HF_HOME", str(models_dir))
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
        from session import ensure_session_dirs
        from audio_io import standardize_and_get_path
        from asr import transcribe_with_alignment
        from turns import build_turns
        from merge import merge_turn_streams
        from srt_vtt import write_srt, write_vtt
        from txt_writer import write_timestamped_txt, write_plain_txt
        from render_black import render_black_video
        from diarize import diarize_mixed
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
        "render_black_video": render_black_video,
        "diarize_mixed": diarize_mixed,
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

    # Ensure outdir & subdirs
    outdir = ensure_outdir(args.outdir)
    paths = api["ensure_session_dirs"](outdir)

    # Run pipeline
    if combined_mode:
        mixed_path = ensure_file(args.combined, "Combined")
        print(f"[*] Mode: combined | ASR: {args.asr}")
        # 1) Standardize
        std_mix = api["standardize_and_get_path"](mixed_path)
        # 2) ASR + alignment
        words = api["transcribe_with_alignment"](str(std_mix), asr_model=args.asr, role=None)
        # 3) Diarize → turns
        turns = api["diarize_mixed"](str(std_mix), words)
        # 4) Outputs
        api["write_timestamped_txt"](turns, paths["merged"] / "transcript.timestamped.txt")
        api["write_plain_txt"](turns, paths["merged"] / "transcript.txt")
        srt_path = paths["merged"] / "subtitles.srt"
        api["write_srt"](turns, srt_path)
        if args.write_vtt:
            api["write_vtt"](turns, paths["merged"] / "subtitles.vtt")
        if args.render_black:
            api["render_black_video"](srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=std_mix)
        print("[✓] Combined processing complete.")

    else:
        # dual-track
        interviewer_path = ensure_file(args.interviewer, "Interviewer")
        participant_path = ensure_file(args.participant, "Participant")
        print(f"[*] Mode: dual-track | ASR: {args.asr}")
        # 1) Standardize
        std_int = api["standardize_and_get_path"](interviewer_path)
        std_part = api["standardize_and_get_path"](participant_path)
        # 2) ASR + alignment per track
        int_words = api["transcribe_with_alignment"](str(std_int), asr_model=args.asr, role="Interviewer")
        part_words = api["transcribe_with_alignment"](str(std_part), asr_model=args.asr, role="Participant")
        # 3) Turns per track
        int_turns = api["build_turns"](int_words, speaker_label="Interviewer")
        part_turns = api["build_turns"](part_words, speaker_label="Participant")
        # 4) Merge turns
        merged = api["merge_turn_streams"](int_turns, part_turns)
        # 5) Per-speaker outputs
        api["write_timestamped_txt"](int_turns, paths["speaker_interviewer"] / "interviewer.timestamped.txt")
        api["write_plain_txt"](int_turns,        paths["speaker_interviewer"] / "interviewer.txt")
        api["write_timestamped_txt"](part_turns, paths["speaker_participant"] / "participant.timestamped.txt")
        api["write_plain_txt"](part_turns,       paths["speaker_participant"] / "participant.txt")
        # 6) Merged outputs
        api["write_timestamped_txt"](merged, paths["merged"] / "transcript.timestamped.txt")
        api["write_plain_txt"](merged,       paths["merged"] / "transcript.txt")
        srt_path = paths["merged"] / "subtitles.srt"
        api["write_srt"](merged, srt_path)
        if args.write_vtt:
            api["write_vtt"](merged, paths["merged"] / "subtitles.vtt")
        if args.render_black:
            # choose which audio to mux; interviewer by default
            api["render_black_video"](srt_path, paths["merged"] / "black_subtitled.mp4", audio_path=std_int)
        print("[✓] Dual-track processing complete.")

    # Summary
    print(f"[i] Artifacts written to: {paths['root']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
