#!/usr/bin/env python3
# main.py - local-transcribe CLI runner

from __future__ import annotations
import sys
from typing import Optional

from local_transcribe.lib.environment import repo_root_from_here, set_offline_env, ensure_models_exist
from local_transcribe.framework.cli import parse_args, interactive_prompt
from local_transcribe.framework.setup import import_pipeline_modules, handle_plugin_listing, handle_plugin_template_creation
from local_transcribe.framework.pipeline_runner import run_pipeline

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
    ensure_models_exist(models_dir)
    api = import_pipeline_modules(root)

    # Handle plugin template creation
    if args.create_plugin_template:
        handle_plugin_template_creation(args)
        return 0

    # Handle plugin listing (doesn't require other args)
    if args.list_plugins:
        handle_plugin_listing(api)
        return 0

    if args.interactive:
        args = interactive_prompt(args, api)

    # Run the pipeline
    return run_pipeline(args, api, root)

if __name__ == "__main__":
    raise SystemExit(main())
