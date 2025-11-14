#!/usr/bin/env python3
# framework/provider_setup.py - Provider setup and configuration logic

from typing import Dict, Any, Optional, Tuple
import pathlib


class ProviderSetup:
    """Handles setup and configuration of all provider types."""
    
    def __init__(self, registry, args):
        self.registry = registry
        self.args = args
    
    def setup_providers(self, mode: str) -> Dict[str, Any]:
        """
        Setup all required providers based on processing mode and arguments.
        
        Args:
            mode: Either "combined_audio" or "split_audio"
            
        Returns:
            Dictionary containing all configured providers
        """
        providers = {}
        
        # Check if using unified processing mode
        if hasattr(self.args, 'processing_mode') and self.args.processing_mode == "unified":
            providers.update(self._setup_unified_provider())
        else:
            providers.update(self._setup_separate_processing_providers(mode))
        
        # Setup transcript cleanup provider if specified
        providers.update(self._setup_transcript_cleanup_provider())
        
        return providers
    
    def _setup_unified_provider(self) -> Dict[str, Any]:
        """Setup unified provider for unified processing mode."""
        try:
            unified_provider = self.registry.get_unified_provider(self.args.unified_provider)
            
            # Set default model if not specified
            if not hasattr(self.args, 'unified_model') or self.args.unified_model is None:
                available_models = unified_provider.get_available_models()
                self.args.unified_model = available_models[0] if available_models else None
            
            return {'unified': unified_provider}
            
        except ValueError as e:
            raise ValueError(f"Unified provider setup failed: {e}")
    
    def _setup_separate_processing_providers(self, mode: str) -> Dict[str, Any]:
        """Setup providers for separate processing mode."""
        providers = {}
        
        # Setup transcriber
        transcriber_provider = self._setup_transcriber_provider()
        providers['transcriber'] = transcriber_provider
        
        # Setup aligner if needed
        if not transcriber_provider.has_builtin_alignment:
            aligner_provider = self._setup_aligner_provider()
            providers['aligner'] = aligner_provider
        
        # Setup diarization for combined audio mode
        if mode == "combined_audio":
            diarization_provider = self._setup_diarization_provider()
            providers['diarization'] = diarization_provider
        
        # Setup turn builder
        turn_builder_provider = self._setup_turn_builder_provider(mode)
        providers['turn_builder'] = turn_builder_provider
        
        return providers
    
    def _setup_transcriber_provider(self) -> Any:
        """Setup transcriber provider with model selection."""
        try:
            transcriber_provider = self.registry.get_transcriber_provider(self.args.transcriber_provider)
            
            # Set default model if not specified
            if not hasattr(self.args, 'transcriber_model') or self.args.transcriber_model is None:
                available_models = transcriber_provider.get_available_models()
                self.args.transcriber_model = available_models[0] if available_models else None
            
            return transcriber_provider
            
        except ValueError as e:
            raise ValueError(f"Transcriber provider setup failed: {e}")
    
    def _setup_aligner_provider(self) -> Any:
        """Setup aligner provider."""
        try:
            return self.registry.get_aligner_provider(self.args.aligner_provider)
        except ValueError as e:
            raise ValueError(f"Aligner provider setup failed: {e}")
    
    def _setup_diarization_provider(self) -> Any:
        """Setup diarization provider."""
        try:
            return self.registry.get_diarization_provider(self.args.diarization_provider)
        except ValueError as e:
            raise ValueError(f"Diarization provider setup failed: {e}")
    
    def _setup_turn_builder_provider(self, mode: str) -> Any:
        """Setup turn builder provider based on mode."""
        try:
            if mode == "combined_audio":
                turn_builder_name = "multi_speaker"
            else:
                turn_builder_name = getattr(self.args, 'turn_builder_provider', "single_speaker_length_based")
            
            return self.registry.get_turn_builder_provider(turn_builder_name)
            
        except ValueError as e:
            raise ValueError(f"Turn builder provider setup failed: {e}")
    
    def _setup_transcript_cleanup_provider(self) -> Dict[str, Any]:
        """Setup transcript cleanup provider if specified."""
        providers = {}
        
        if hasattr(self.args, 'transcript_cleanup_provider') and self.args.transcript_cleanup_provider:
            try:
                transcript_cleanup_provider = self.registry.get_transcript_cleanup_provider(
                    self.args.transcript_cleanup_provider
                )
                
                # Reinitialize with custom URL if provided
                if (hasattr(self.args, 'transcript_cleanup_url') and 
                    self.args.transcript_cleanup_url and 
                    hasattr(transcript_cleanup_provider, 'url')):
                    transcript_cleanup_provider.url = self.args.transcript_cleanup_url
                
                providers['transcript_cleanup'] = transcript_cleanup_provider
                
            except ValueError as e:
                raise ValueError(f"Transcript cleanup provider setup failed: {e}")
        
        return providers
    
    def get_model_download_providers(self) -> Dict[str, Any]:
        """
        Get providers that need model downloads.
        
        Returns:
            Dictionary of providers that require models to be downloaded
        """
        providers = {}
        
        if hasattr(self.args, 'processing_mode') and self.args.processing_mode == "unified":
            # For unified mode, we only need the unified provider
            try:
                unified_provider = self.registry.get_unified_provider(self.args.unified_provider)
                providers['unified'] = unified_provider
            except ValueError:
                # If provider setup fails, return empty dict - will be handled by main pipeline
                pass
        else:
            # For separate processing mode, we need transcriber, aligner (if needed), and diarization
            try:
                transcriber_provider = self.registry.get_transcriber_provider(self.args.transcriber_provider)
                providers['transcriber'] = transcriber_provider
                
                # Add aligner if transcriber doesn't have built-in alignment
                if not transcriber_provider.has_builtin_alignment:
                    aligner_provider = self.registry.get_aligner_provider(self.args.aligner_provider)
                    providers['aligner'] = aligner_provider
                
                # Diarization is always needed for model download check
                diarization_provider = self.registry.get_diarization_provider(self.args.diarization_provider)
                providers['diarization'] = diarization_provider
                
            except ValueError:
                # If provider setup fails, return empty dict - will be handled by main pipeline
                pass
        
        return providers