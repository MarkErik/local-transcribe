#!/usr/bin/env python3
"""
Tests for lazy imports functionality.

These tests ensure that lazy imports work correctly for deferring
heavy module loads (like torch, transformers) until actually needed.
This is tested before and after consolidating lazy import patterns.
"""

import sys
import os

import pytest

# Add parent directory to path for local_transcribe imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# Test Shared Lazy Imports Module
# ==============================================================================

class TestSharedLazyImportsModule:
    """Test the shared lazy_imports module directly."""

    def test_get_granite_model_manager_class(self):
        """Test that get_granite_model_manager_class returns the correct class."""
        from local_transcribe.providers.common.lazy_imports import get_granite_model_manager_class
        
        GraniteModelManager = get_granite_model_manager_class()
        
        assert GraniteModelManager is not None
        assert hasattr(GraniteModelManager, 'MODEL_MAPPING')
        assert GraniteModelManager.__name__ == 'GraniteModelManager'

    def test_get_mfa_alignment_engine_class(self):
        """Test that get_mfa_alignment_engine_class returns the correct class."""
        from local_transcribe.providers.common.lazy_imports import get_mfa_alignment_engine_class
        
        MFAAlignmentEngine = get_mfa_alignment_engine_class()
        
        assert MFAAlignmentEngine is not None
        assert hasattr(MFAAlignmentEngine, 'parse_textgrid_to_word_dicts')
        assert MFAAlignmentEngine.__name__ == 'MFAAlignmentEngine'

    def test_get_silero_vad_provider_class(self):
        """Test that get_silero_vad_provider_class returns the correct class."""
        from local_transcribe.providers.common.lazy_imports import get_silero_vad_provider_class
        
        SileroVADProvider = get_silero_vad_provider_class()
        
        assert SileroVADProvider is not None
        assert hasattr(SileroVADProvider, 'detect_speech')
        assert SileroVADProvider.__name__ == 'SileroVADProvider'

    def test_get_vad_segmenter_func(self):
        """Test that get_vad_segmenter_func returns a callable."""
        from local_transcribe.providers.common.lazy_imports import get_vad_segmenter_func
        
        segment_for_asr = get_vad_segmenter_func()
        
        assert segment_for_asr is not None
        assert callable(segment_for_asr)

    def test_repeated_calls_return_same_instance(self):
        """Test that repeated calls return the same cached class."""
        from local_transcribe.providers.common.lazy_imports import (
            get_granite_model_manager_class,
            get_mfa_alignment_engine_class,
            get_silero_vad_provider_class,
        )
        
        # Get classes twice
        class1a = get_granite_model_manager_class()
        class1b = get_granite_model_manager_class()
        
        class2a = get_mfa_alignment_engine_class()
        class2b = get_mfa_alignment_engine_class()
        
        class3a = get_silero_vad_provider_class()
        class3b = get_silero_vad_provider_class()
        
        # Should be the exact same object (cached)
        assert class1a is class1b
        assert class2a is class2b
        assert class3a is class3b


# ==============================================================================
# Test Existing Lazy Import Pattern
# ==============================================================================

class TestGraniteModelManagerLazyImport:
    """Test lazy import of GraniteModelManager from various providers."""

    def test_granite_provider_lazy_import(self):
        """Test that granite.py lazily imports GraniteModelManager."""
        # Import the module without triggering the lazy import
        from local_transcribe.providers.transcribers import granite
        
        # The class-level variable should be None initially
        # (we can't directly test _granite_model_manager_class due to module state)
        
        # Getting the class should work
        GraniteModelManager = granite._get_granite_model_manager_class()
        
        assert GraniteModelManager is not None
        assert hasattr(GraniteModelManager, 'MODEL_MAPPING')

    def test_granite_mfa_provider_lazy_import(self):
        """Test that granite_mfa.py lazily imports GraniteModelManager."""
        from local_transcribe.providers.transcribers import granite_mfa
        
        GraniteModelManager = granite_mfa._get_granite_model_manager_class()
        
        assert GraniteModelManager is not None
        assert hasattr(GraniteModelManager, 'MODEL_MAPPING')

    def test_granite_wav2vec2_provider_lazy_import(self):
        """Test that granite_wav2vec2.py lazily imports GraniteModelManager."""
        from local_transcribe.providers.transcribers import granite_wav2vec2
        
        GraniteModelManager = granite_wav2vec2._get_granite_model_manager_class()
        
        assert GraniteModelManager is not None
        assert hasattr(GraniteModelManager, 'MODEL_MAPPING')

    def test_granite_vad_silero_mfa_provider_lazy_import(self):
        """Test that granite_vad_silero_mfa.py lazily imports GraniteModelManager."""
        from local_transcribe.providers.transcribers import granite_vad_silero_mfa
        
        GraniteModelManager = granite_vad_silero_mfa._get_granite_model_manager_class()
        
        assert GraniteModelManager is not None
        assert hasattr(GraniteModelManager, 'MODEL_MAPPING')

    def test_all_granite_providers_same_class(self):
        """Test that all providers get the same GraniteModelManager class."""
        from local_transcribe.providers.transcribers import granite
        from local_transcribe.providers.transcribers import granite_mfa
        from local_transcribe.providers.transcribers import granite_wav2vec2
        from local_transcribe.providers.transcribers import granite_vad_silero_mfa
        
        classes = [
            granite._get_granite_model_manager_class(),
            granite_mfa._get_granite_model_manager_class(),
            granite_wav2vec2._get_granite_model_manager_class(),
            granite_vad_silero_mfa._get_granite_model_manager_class(),
        ]
        
        # All should be the same class
        assert all(c is classes[0] for c in classes)


class TestMFAAlignmentEngineLazyImport:
    """Test lazy import of MFAAlignmentEngine from various providers."""

    def test_mfa_aligner_lazy_import(self):
        """Test that mfa.py (aligner) lazily imports MFAAlignmentEngine."""
        from local_transcribe.providers.aligners import mfa
        
        MFAAlignmentEngine = mfa._get_mfa_alignment_engine_class()
        
        assert MFAAlignmentEngine is not None
        assert hasattr(MFAAlignmentEngine, 'parse_textgrid_to_word_dicts')

    def test_granite_mfa_provider_mfa_lazy_import(self):
        """Test that granite_mfa.py lazily imports MFAAlignmentEngine."""
        from local_transcribe.providers.transcribers import granite_mfa
        
        MFAAlignmentEngine = granite_mfa._get_mfa_alignment_engine_class()
        
        assert MFAAlignmentEngine is not None
        assert hasattr(MFAAlignmentEngine, 'parse_textgrid_to_word_dicts')

    def test_granite_vad_silero_mfa_mfa_lazy_import(self):
        """Test that granite_vad_silero_mfa.py lazily imports MFAAlignmentEngine."""
        from local_transcribe.providers.transcribers import granite_vad_silero_mfa
        
        MFAAlignmentEngine = granite_vad_silero_mfa._get_mfa_alignment_engine_class()
        
        assert MFAAlignmentEngine is not None
        assert hasattr(MFAAlignmentEngine, 'parse_textgrid_to_word_dicts')

    def test_all_mfa_providers_same_class(self):
        """Test that all providers get the same MFAAlignmentEngine class."""
        from local_transcribe.providers.aligners import mfa
        from local_transcribe.providers.transcribers import granite_mfa
        from local_transcribe.providers.transcribers import granite_vad_silero_mfa
        
        classes = [
            mfa._get_mfa_alignment_engine_class(),
            granite_mfa._get_mfa_alignment_engine_class(),
            granite_vad_silero_mfa._get_mfa_alignment_engine_class(),
        ]
        
        # All should be the same class
        assert all(c is classes[0] for c in classes)


class TestSileroVADProviderLazyImport:
    """Test lazy import of SileroVADProvider."""

    def test_silero_vad_lazy_import(self):
        """Test that SileroVADProvider is lazily imported."""
        from local_transcribe.providers.transcribers import granite_vad_silero_mfa
        
        SileroVADProvider = granite_vad_silero_mfa._get_silero_vad_provider_class()
        
        assert SileroVADProvider is not None
        assert hasattr(SileroVADProvider, 'detect_speech')


# ==============================================================================
# Test Provider Initialization (Lazy Model Manager)
# ==============================================================================

class TestProviderLazyModelManagerInitialization:
    """Test that providers lazily initialize their model managers."""

    def test_granite_provider_lazy_model_manager(self):
        """Test GraniteTranscriberProvider lazily initializes model manager."""
        from local_transcribe.providers.transcribers.granite import GraniteTranscriberProvider
        
        provider = GraniteTranscriberProvider()
        
        # _model_manager should be None before first access
        assert provider._model_manager is None
        
        # Accessing model_manager property should initialize it
        model_manager = provider.model_manager
        
        assert model_manager is not None
        assert provider._model_manager is not None

    def test_granite_mfa_provider_lazy_model_manager(self):
        """Test GraniteMFATranscriberProvider lazily initializes model manager."""
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        
        provider = GraniteMFATranscriberProvider()
        
        assert provider._model_manager is None
        
        model_manager = provider.model_manager
        
        assert model_manager is not None
        assert provider._model_manager is not None

    def test_granite_mfa_provider_lazy_alignment_engine(self):
        """Test GraniteMFATranscriberProvider lazily initializes alignment engine."""
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        
        provider = GraniteMFATranscriberProvider()
        
        assert provider._alignment_engine is None
        
        alignment_engine = provider.word_alignment_engine
        
        assert alignment_engine is not None
        assert provider._alignment_engine is not None

    def test_mfa_aligner_provider_lazy_alignment_engine(self):
        """Test MFAAlignerProvider lazily initializes alignment engine."""
        from local_transcribe.providers.aligners.mfa import MFAAlignerProvider
        
        provider = MFAAlignerProvider()
        
        assert provider._alignment_engine is None
        
        alignment_engine = provider.word_alignment_engine
        
        assert alignment_engine is not None
        assert provider._alignment_engine is not None


# ==============================================================================
# Test Provider Properties
# ==============================================================================

class TestProviderProperties:
    """Test provider property values are consistent."""

    def test_granite_provider_properties(self):
        """Test GraniteTranscriberProvider properties."""
        from local_transcribe.providers.transcribers.granite import GraniteTranscriberProvider
        
        provider = GraniteTranscriberProvider()
        
        assert provider.name == "granite"
        assert provider.short_name == "IBM Granite"
        assert "IBM Granite" in provider.description
        assert provider.has_builtin_alignment is False

    def test_granite_mfa_provider_properties(self):
        """Test GraniteMFATranscriberProvider properties."""
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        
        provider = GraniteMFATranscriberProvider()
        
        assert provider.name == "granite_mfa"
        assert provider.short_name == "Granite + MFA"
        assert "IBM Granite" in provider.description
        assert provider.has_builtin_alignment is True

    def test_granite_wav2vec2_provider_properties(self):
        """Test GraniteWav2Vec2TranscriberProvider properties."""
        from local_transcribe.providers.transcribers.granite_wav2vec2 import GraniteWav2Vec2TranscriberProvider
        
        provider = GraniteWav2Vec2TranscriberProvider()
        
        assert provider.name == "granite_wav2vec2"
        assert provider.short_name == "Granite + Wav2Vec2"
        assert "IBM Granite" in provider.description
        assert provider.has_builtin_alignment is True

    def test_granite_vad_silero_mfa_provider_properties(self):
        """Test GraniteVADSileroMFATranscriberProvider properties."""
        from local_transcribe.providers.transcribers.granite_vad_silero_mfa import GraniteVADSileroMFATranscriberProvider
        
        provider = GraniteVADSileroMFATranscriberProvider()
        
        assert provider.name == "granite_vad_silero_mfa"
        assert provider.short_name == "Granite + VAD (Silero) + MFA"
        assert "IBM Granite" in provider.description
        assert provider.has_builtin_alignment is True

    def test_mfa_aligner_provider_properties(self):
        """Test MFAAlignerProvider properties."""
        from local_transcribe.providers.aligners.mfa import MFAAlignerProvider
        
        provider = MFAAlignerProvider()
        
        assert provider.name == "mfa"
        assert provider.short_name == "MFA"
        assert "Montreal Forced Aligner" in provider.description


# ==============================================================================
# Test Available Models
# ==============================================================================

class TestAvailableModels:
    """Test that providers return correct available models."""

    def test_granite_provider_available_models(self):
        """Test GraniteTranscriberProvider returns available models."""
        from local_transcribe.providers.transcribers.granite import GraniteTranscriberProvider
        
        provider = GraniteTranscriberProvider()
        models = provider.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "granite-8b" in models or "granite-3b" in models

    def test_granite_mfa_provider_available_models(self):
        """Test GraniteMFATranscriberProvider returns available models."""
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        
        provider = GraniteMFATranscriberProvider()
        models = provider.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0

    def test_mfa_aligner_required_models(self):
        """Test MFAAlignerProvider returns empty required models."""
        from local_transcribe.providers.aligners.mfa import MFAAlignerProvider
        
        provider = MFAAlignerProvider()
        models = provider.get_required_models()
        
        # MFA doesn't use HuggingFace models
        assert models == []


# ==============================================================================
# Run tests directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
