#!/usr/bin/env python3
# framework/setup.py - Module importing and plugin setup

import sys
import pathlib

# ---------- import pipeline modules ----------
def import_pipeline_modules(repo_root: pathlib.Path):
    # Ensure we're in the repo root to prevent relative path issues
    import os
    os.chdir(repo_root)
    
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

def handle_plugin_listing(api):
    print("Available Plugins:")
    print("\nTranscriber Providers:")
    for name, desc in api["registry"].list_transcriber_providers().items():
        print(f"  {name}: {desc}")
        # Show available models if provider supports multiple
        provider = api["registry"].get_transcriber_provider(name)
        available_models = provider.get_available_models()
        if len(available_models) > 1:
            print(f"    Available models: {', '.join(available_models)}")
        # Show if it has built-in alignment
        if provider.has_builtin_alignment:
            print("    Has built-in alignment: Yes")
        else:
            print("    Has built-in alignment: No (requires aligner)")

    print("\nAligner Providers:")
    for name, desc in api["registry"].list_aligner_providers().items():
        print(f"  {name}: {desc}")
        # Show available models if provider supports multiple
        provider = api["registry"].get_aligner_provider(name)
        available_models = provider.get_available_models()
        if len(available_models) > 1:
            print(f"    Available models: {', '.join(available_models)}")

    print("\nDiarization Providers:")
    for name, desc in api["registry"].list_diarization_providers().items():
        print(f"  {name}: {desc}")

    print("\nTurn Builder Providers:")
    for name, desc in api["registry"].list_turn_builder_providers().items():
        print(f"  {name}: {desc}")

    print("\nUnified Providers:")
    for name, desc in api["registry"].list_unified_providers().items():
        print(f"  {name}: {desc}")
        # Show available models if provider supports multiple
        provider = api["registry"].get_unified_provider(name)
        available_models = provider.get_available_models()
        if len(available_models) > 1:
            print(f"    Available models: {', '.join(available_models)}")

    print("\nOutput Writers:")
    for name, desc in api["registry"].list_output_writers().items():
        print(f"  {name}: {desc}")

    print("\nWord Writers:")
    for name, desc in api["registry"].list_word_writers().items():
        print(f"  {name}: {desc}")