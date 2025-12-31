"""
Granite Model Module

A modular package for IBM Granite speech model management and transcription.

This package provides:
- GraniteModelManager: Main manager class for Granite models
- CacheManager: Cache directory management
- TranscriptionMixin: Audio transcription functionality
- ModelLoadingMixin: Model loading and downloading
- ModelValidationMixin: Model validation and availability checking

Usage:
    from local_transcribe.providers.common.granite_model import GraniteModelManager
    
    manager = GraniteModelManager(logger)
    manager._load_model("ibm-granite/granite-speech-3.3-8b")
    transcript = manager.transcribe_segment(audio_array)
"""

from local_transcribe.providers.common.granite_model.manager import GraniteModelManager
from local_transcribe.providers.common.granite_model.cache_management import CacheManager
from local_transcribe.providers.common.granite_model.transcription import TranscriptionMixin
from local_transcribe.providers.common.granite_model.model_loading import ModelLoadingMixin
from local_transcribe.providers.common.granite_model.model_validation import ModelValidationMixin

__all__ = [
    'GraniteModelManager',
    'CacheManager',
    'TranscriptionMixin',
    'ModelLoadingMixin',
    'ModelValidationMixin',
]
