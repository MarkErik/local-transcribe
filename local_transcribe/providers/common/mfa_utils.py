#!/usr/bin/env python3
"""
Shared MFA utilities for Montreal Forced Aligner command and model management.

This module consolidates duplicate MFA helper code that was previously replicated
across multiple provider files (aligners/mfa.py, transcribers/granite_mfa.py, 
transcribers/granite_vad_silero_mfa.py).

Benefits:
- Single source of truth for MFA command detection
- Consistent model downloading behavior across all MFA providers
- Reduced code duplication (~100 lines saved)
- Easier maintenance when MFA behavior needs to change
"""

import os
import pathlib
import subprocess
from typing import Optional, TYPE_CHECKING

from local_transcribe.lib.program_logger import log_progress, log_completion, log_debug

if TYPE_CHECKING:
    from logging import Logger


# Cache the MFA command path to avoid repeated file system checks
_cached_mfa_command: Optional[str] = None


def get_mfa_command(logger: Optional["Logger"] = None) -> str:
    """
    Get the MFA command, checking local environment first.
    
    Looks for MFA in the project-local .mfa_env directory first,
    then falls back to system MFA.
    
    Args:
        logger: Optional logger for debug output
        
    Returns:
        Path to MFA executable or 'mfa' for system command
    """
    global _cached_mfa_command
    
    if _cached_mfa_command is not None:
        return _cached_mfa_command
    
    # Check if MFA is available in project-local environment
    project_root = pathlib.Path(__file__).parent.parent.parent.parent
    local_mfa_env = project_root / ".mfa_env" / "bin" / "mfa"
    
    if local_mfa_env.exists():
        if logger:
            logger.info(f"[MFA] Using local MFA: {local_mfa_env}")
        _cached_mfa_command = str(local_mfa_env)
        return _cached_mfa_command
    
    # Fall back to system MFA
    if logger:
        logger.info("[MFA] Using system MFA: mfa")
    _cached_mfa_command = "mfa"
    return _cached_mfa_command


def ensure_mfa_models(mfa_models_dir: pathlib.Path, logger: Optional["Logger"] = None) -> None:
    """
    Ensure MFA acoustic model and dictionary are downloaded to project directory.
    
    Downloads the english_us_arpa acoustic model and dictionary if not present.
    
    Args:
        mfa_models_dir: Directory to store MFA models
        logger: Optional logger for output
        
    Raises:
        subprocess.CalledProcessError: If model download fails
    """
    if logger:
        logger.info(f"[MFA] Checking MFA models in {mfa_models_dir}")
    
    # Set MFA_ROOT_DIR environment variable to use project models directory
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(mfa_models_dir)

    mfa_cmd = get_mfa_command(logger)
    if logger:
        logger.info(f"[MFA] Using MFA command: {mfa_cmd}")
    
    try:
        # Check if acoustic model is already downloaded
        result = subprocess.run(
            [mfa_cmd, "model", "list", "acoustic"],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        log_debug(f"[MFA] Available acoustic models: {result.stdout.strip()}")

        if "english_us_arpa" not in result.stdout:
            log_progress(f"[MFA] Downloading MFA English acoustic model to {mfa_models_dir}...")
            subprocess.run(
                [mfa_cmd, "model", "download", "acoustic", "english_us_arpa"],
                check=True,
                env=env
            )
            log_completion("[MFA] Acoustic model downloaded successfully")
        else:
            if logger:
                logger.info("[MFA] Acoustic model english_us_arpa already available")

        # Check if dictionary is already downloaded
        result = subprocess.run(
            [mfa_cmd, "model", "list", "dictionary"],
            capture_output=True,
            text=True,
            check=True,
            env=env
        )
        log_debug(f"[MFA] Available dictionaries: {result.stdout.strip()}")

        if "english_us_arpa" not in result.stdout:
            log_progress(f"[MFA] Downloading MFA English dictionary to {mfa_models_dir}...")
            subprocess.run(
                [mfa_cmd, "model", "download", "dictionary", "english_us_arpa"],
                check=True,
                env=env
            )
            log_completion("[MFA] Dictionary downloaded successfully")
        else:
            if logger:
                logger.info("[MFA] Dictionary english_us_arpa already available")

    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"[MFA] ERROR: Failed to check/download MFA models: {e}")
            if hasattr(e, 'stdout') and e.stdout:
                logger.error(f"[MFA] stdout: {e.stdout}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"[MFA] stderr: {e.stderr}")
        raise


def get_mfa_environment(mfa_models_dir: pathlib.Path, logger: Optional["Logger"] = None) -> dict:
    """
    Get environment variables configured for MFA execution.
    
    Args:
        mfa_models_dir: Directory where MFA models are stored
        logger: Optional logger
        
    Returns:
        Dictionary of environment variables for subprocess calls
    """
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(mfa_models_dir)
    env["MFA_NO_HISTORY"] = "1"
    
    # Add MFA bin directory to PATH
    mfa_cmd = get_mfa_command(logger)
    mfa_env_bin = pathlib.Path(mfa_cmd).parent
    env["PATH"] = str(mfa_env_bin) + os.pathsep + env.get("PATH", "")
    
    return env


def get_mfa_config_path() -> pathlib.Path:
    """Get path to the MFA configuration file."""
    project_root = pathlib.Path(__file__).parent.parent.parent.parent
    return project_root / "mfa_config.yaml"


# Reset function for testing purposes
def _reset_cache() -> None:
    """Reset the cached MFA command (for testing only)."""
    global _cached_mfa_command
    _cached_mfa_command = None
