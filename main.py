#!/usr/bin/env python3
# main.py - local-transcribe CLI runner

from __future__ import annotations

# Disable tokenizers parallelism to avoid warnings when forking processes (e.g., ffmpeg)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Optional

from local_transcribe.lib.environment import repo_root_from_here, set_offline_env, ensure_models_exist, validate_system_capability
from local_transcribe.lib.system_capability_utils import set_system_capability
from local_transcribe.framework.cli import parse_args, interactive_prompt
from local_transcribe.framework.plugin_manager import import_pipeline_modules, handle_plugin_listing
from local_transcribe.framework.pipeline_runner import run_pipeline

# ---------- main ----------
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Validate system capability if provided via command line
    if hasattr(args, 'system') and args.system:
        args.system = validate_system_capability(args.system)
        set_system_capability(args.system)
    else:
        # Default to CPU if not specified (will be overridden in interactive mode)
        set_system_capability("cpu")

    # Handle show-defaults flag (doesn't require other args)
    if args.show_defaults:
        from local_transcribe.framework.cli import show_defaults
        show_defaults()
        return 0

    # Handle list-stages flag (doesn't require other args)
    if args.list_stages:
        from local_transcribe.framework.cli import list_stages
        list_stages()
        return 0

    root = repo_root_from_here()
    models_dir = root / ".models"
    set_offline_env(models_dir)
    ensure_models_exist(models_dir)
    api = import_pipeline_modules(root)

    # Handle plugin listing (doesn't require other args)
    if args.list_plugins:
        handle_plugin_listing(api)
        return 0

    # Handle pipeline re-entry from checkpoint
    if hasattr(args, 'from_diarized_json') and args.from_diarized_json:
        from local_transcribe.framework.pipeline_reentry import (
            run_pipeline_from_checkpoint,
            check_reentry_requirements
        )
        
        # Validate re-entry requirements
        requirements_met, error_msg = check_reentry_requirements(args)
        if not requirements_met:
            print(f"Error: {error_msg}")
            return 1
        
        # Run re-entry pipeline
        return run_pipeline_from_checkpoint(args, api, root)

    # Early validation for required args (full pipeline mode)
    if not args.outdir:
        print("Error: -o/--outdir is required")
        return 1
    if not hasattr(args, 'audio_files') or not args.audio_files:
        print("Error: Must provide --audio-files (-a) or --from-diarized-json")
        return 1

    if args.interactive:
        args = interactive_prompt(args, api)
        # Update system capability after interactive selection
        if hasattr(args, 'system') and args.system:
            args.system = validate_system_capability(args.system)
            set_system_capability(args.system)
            print(f"[i] System capability set to: {args.system.upper()}")

    # Run the pipeline
    return run_pipeline(args, api, root)

if __name__ == "__main__":
    raise SystemExit(main())
