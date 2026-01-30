#!/usr/bin/env python3
"""
Shared lazy imports for provider modules.

This module provides lazy import utilities to defer heavy module loads
(like torch, transformers, etc.) until they are actually needed.
This improves CLI startup time by avoiding unnecessary imports.

Usage:
    from local_transcribe.providers.common.lazy_imports import (
        get_granite_model_manager_class,
        get_mfa_alignment_engine_class,
        get_silero_vad_provider_class,
    )
    
    # Later, when actually needed:
    GraniteModelManager = get_granite_model_manager_class()
    manager = GraniteModelManager(logger)
"""

from typing import TYPE_CHECKING, Any, Optional, Type

# Type hints for lazy-loaded modules
if TYPE_CHECKING:
    from local_transcribe.providers.common.granite_model import GraniteModelManager
    from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine


# Module-level cache for lazy-loaded classes
_granite_model_manager_class: Optional[Type["GraniteModelManager"]] = None
_mfa_alignment_engine_class: Optional[Type["MFAAlignmentEngine"]] = None
_silero_vad_provider_class: Optional[Type[Any]] = None


def get_granite_model_manager_class() -> Type["GraniteModelManager"]:
    """
    Lazily import and return the GraniteModelManager class.
    
    This defers torch import until the first time this function is called,
    improving startup time for CLI commands that don't need ML models.
    
    Returns:
        The GraniteModelManager class (not an instance)
    """
    global _granite_model_manager_class
    if _granite_model_manager_class is None:
        from local_transcribe.providers.common.granite_model import GraniteModelManager
        _granite_model_manager_class = GraniteModelManager
    return _granite_model_manager_class


def get_mfa_alignment_engine_class() -> Type["MFAAlignmentEngine"]:
    """
    Lazily import and return the MFAAlignmentEngine class.
    
    Returns:
        The MFAAlignmentEngine class (not an instance)
    """
    global _mfa_alignment_engine_class
    if _mfa_alignment_engine_class is None:
        from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine
        _mfa_alignment_engine_class = MFAAlignmentEngine
    return _mfa_alignment_engine_class


def get_silero_vad_provider_class() -> Type[Any]:
    """
    Lazily import and return the SileroVADProvider class.
    
    Returns:
        The SileroVADProvider class (not an instance)
    """
    global _silero_vad_provider_class
    if _silero_vad_provider_class is None:
        from local_transcribe.providers.vad import SileroVADProvider
        _silero_vad_provider_class = SileroVADProvider
    return _silero_vad_provider_class


def get_vad_segmenter_func():
    """
    Lazily import and return the segment_for_asr function.
    
    Returns:
        The segment_for_asr function
    """
    from local_transcribe.processing.vad.segmenter import segment_for_asr
    return segment_for_asr


# Reset functions for testing purposes
def _reset_caches():
    """Reset all cached classes (for testing only)."""
    global _granite_model_manager_class, _mfa_alignment_engine_class, _silero_vad_provider_class
    _granite_model_manager_class = None
    _mfa_alignment_engine_class = None
    _silero_vad_provider_class = None
