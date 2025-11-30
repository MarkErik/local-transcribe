#!/usr/bin/env python3
# lib/environment.py - Environment setup and utility checks

from __future__ import annotations
import os
import sys
import pathlib
import warnings
from typing import List
from dotenv import load_dotenv

# Aggressively suppress warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Set environment variables to suppress warnings from specific libraries
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Load environment variables from .env file
load_dotenv()

# ---------- system capabilities ----------
def get_available_system_capabilities() -> List[str]:
    """Check available system capabilities for ML acceleration."""
    capabilities = ["cpu"]  # CPU is always available
    
    try:
        import torch
        if torch.cuda.is_available():
            capabilities.append("cuda")
        if torch.backends.mps.is_available():
            capabilities.append("mps")
    except ImportError:
        pass  # torch not available, only CPU
    
    return capabilities

def validate_system_capability(capability: str) -> str:
    """Validate and return a valid system capability, defaulting to CPU if invalid."""
    available = get_available_system_capabilities()
    if capability in available:
        return capability
    else:
        print(f"Warning: System capability '{capability}' is not available on this system. Available: {available}. Defaulting to 'cpu'.")
        return "cpu"
def repo_root_from_here() -> pathlib.Path:
    # Resolve repo root as the directory containing this file
    return pathlib.Path(__file__).resolve().parent.parent.parent

def set_offline_env(models_dir: pathlib.Path) -> None:
    # Don't set global HF_HOME - let plugins manage their own cache locations
    # os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(models_dir / ".xdg"))
    # Runtime must be fully offline (models must already be downloaded)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------- simple checks ----------
def ensure_models_exist(models_dir: pathlib.Path) -> None:
    if not models_dir.exists():
        print("Warning: .models/ directory not found. Models will be downloaded automatically on first run.")
        models_dir.mkdir(parents=True, exist_ok=True)
    # Check for new provider model directories
    transcriber_dir = models_dir / "transcribers"
    aligner_dir = models_dir / "aligners"
    if not (transcriber_dir.exists() or aligner_dir.exists()):
        print("Warning: Provider models not found in ./.models/. Models will be downloaded automatically on first run.")
    # We won't strictly validate HF cache layout; downloader guarantees presence.

def ensure_file(path: str, label: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        sys.exit(f"Error: {label} file not found: {p}")
    return p

def ensure_outdir(path: str) -> pathlib.Path:
    out = pathlib.Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out