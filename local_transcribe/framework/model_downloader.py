#!/usr/bin/env python3
"""
Model downloading and availability management for local-transcribe.

This module handles checking model availability, downloading missing models,
and managing the HuggingFace environment for offline/online mode switching.
"""

import os
import sys
from typing import List, Optional
from pathlib import Path


def ensure_models_available(providers, models_dir: Path, args) -> int:
    """
    Check model availability and download missing models if needed.

    Args:
        providers: Dict containing 'asr', 'diarization', and/or 'combined' providers
        models_dir: Path to the models directory
        args: Parsed command line arguments

    Returns:
        0 on success, 1 on failure
    """
    # Phase 2: Model Availability Check (Offline)
    if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
        combined_provider = providers['combined']
        required_combined_models = combined_provider.get_required_models(args.combined_model)
        print(f"[*] Checking model availability offline...")
        missing_combined_models = combined_provider.check_models_available_offline(required_combined_models, models_dir)
        all_missing = missing_combined_models
    else:
        asr_provider = providers['asr']
        diarization_provider = providers['diarization']

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

        # Set up HuggingFace environment for downloading
        success = _setup_huggingface_for_download()
        if not success:
            return 1

        try:
            # Download missing models
            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                combined_provider = providers['combined']
                if missing_combined_models:
                    print(f"[*] Downloading combined models: {', '.join(missing_combined_models)}")
                    combined_provider.ensure_models_available(missing_combined_models, models_dir)
            else:
                asr_provider = providers['asr']
                diarization_provider = providers['diarization']

                if missing_asr_models:
                    print(f"[*] Downloading ASR models: {', '.join(missing_asr_models)}")
                    asr_provider.ensure_models_available(missing_asr_models, models_dir)

                if missing_diarization_models:
                    print(f"[*] Downloading diarization models: {', '.join(missing_diarization_models)}")
                    diarization_provider.ensure_models_available(missing_diarization_models, models_dir)

            # Verify downloads actually succeeded
            print("[*] Verifying downloaded models...")
            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                combined_provider = providers['combined']
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
            # Always restore offline mode
            os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("[✓] All required models are available locally.")

    return 0


def _setup_huggingface_for_download() -> bool:
    """
    Configure HuggingFace environment for downloading.

    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Explicitly set online mode
        print(f"DEBUG: Setting HF_HUB_OFFLINE to 0 (was: {os.environ.get('HF_HUB_OFFLINE')})")
        os.environ["HF_HUB_OFFLINE"] = "0"

        # Force reload of huggingface_hub modules to pick up new environment
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
        print(f"    HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
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

        return True
    except Exception as e:
        print(f"ERROR: Failed to setup HuggingFace environment: {e}")
        return False