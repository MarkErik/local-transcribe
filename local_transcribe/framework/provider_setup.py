#!/usr/bin/env python3
# framework/provider_setup.py - Provider setup and configuration logic

from typing import Dict, Any


class ProviderSetup:
    """Handles setup and configuration of all provider types."""
    
    def __init__(self, registry, args):
        self.registry = registry
        self.args = args
    
    def setup_providers(self, mode: str) -> Dict[str, Any]:
        """
        Setup all required providers based on processing mode and arguments.
        
        Args:
            mode: Either "combined_audio", "split_audio", or "vad_split_audio"
            
        Returns:
            Dictionary containing all configured providers
        """
        providers = {}
        
        # Check for single_speaker_audio mode
        if mode == "single_speaker_audio":
            providers.update(self._setup_single_speaker_audio_providers())
        # Check for VAD split audio mode (only needs transcriber)
        elif mode == "split_audio" and getattr(self.args, 'vad_pipeline', False):
            providers.update(self._setup_vad_pipeline_providers())
        else:
            providers.update(self._setup_separate_processing_providers(mode))
        
        # Setup transcript cleanup provider if specified
        providers.update(self._setup_transcript_cleanup_provider())
        
        return providers
    
    def _setup_vad_pipeline_providers(self) -> Dict[str, Any]:
        """Setup providers for VAD pipeline mode (only transcriber needed)."""
        providers = {}
        
        # Setup transcriber only - VAD pipeline handles everything else
        transcriber_provider = self._setup_transcriber_provider()
        providers['transcriber'] = transcriber_provider
        
        # No aligner or diarization needed - VAD handles timing
        providers['aligner'] = None
        providers['diarization'] = None
        
        return providers
    
    def _setup_single_speaker_audio_providers(self) -> Dict[str, Any]:
        """Setup providers for single_speaker_audio mode (only pure transcribers)."""
        providers = {}
        
        # Setup transcriber (only those with has_builtin_alignment = False)
        transcriber_provider = self._setup_pure_transcriber_provider()
        providers['transcriber'] = transcriber_provider
        
        return providers
    
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
    
    def _setup_pure_transcriber_provider(self) -> Any:
        """Setup pure transcriber provider (has_builtin_alignment = False)."""
        try:
            transcriber_provider = self.registry.get_transcriber_provider(self.args.transcriber_provider)
            
            # Validate that it's a pure transcriber
            if transcriber_provider.has_builtin_alignment:
                raise ValueError(f"Provider '{self.args.transcriber_provider}' has built-in alignment and is not allowed in participant-audio-only mode. Use granite or openai_whisper.")
            
            # Set default model if not specified
            if not hasattr(self.args, 'transcriber_model') or self.args.transcriber_model is None:
                available_models = transcriber_provider.get_available_models()
                self.args.transcriber_model = available_models[0] if available_models else None
            
            return transcriber_provider
            
        except ValueError as e:
            raise ValueError(f"Pure transcriber provider setup failed: {e}")
    
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
    
    def _setup_transcript_cleanup_provider(self) -> Dict[str, Any]:
        """Setup transcript cleanup provider if specified."""
        providers = {}
        
        if hasattr(self.args, 'transcript_cleanup_provider') and self.args.transcript_cleanup_provider:
            try:
                transcript_cleanup_provider = self.registry.get_transcript_cleanup_provider(
                    self.args.transcript_cleanup_provider
                )
                
                # Reinitialize with custom URL if provided
                if (hasattr(self.args, 'llm_transcript_cleanup_url') and 
                    self.args.llm_transcript_cleanup_url and 
                    hasattr(transcript_cleanup_provider, 'url')):
                    transcript_cleanup_provider.url = self.args.llm_transcript_cleanup_url
                
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
        
        # Check for single_speaker_audio mode
        if hasattr(self.args, 'single_speaker_audio') and self.args.single_speaker_audio:
            try:
                transcriber_provider = self.registry.get_transcriber_provider(self.args.transcriber_provider)
                providers['transcriber'] = transcriber_provider
            except ValueError:
                pass
            return providers
        
        # Check for VAD pipeline mode - only needs transcriber
        if getattr(self.args, 'vad_pipeline', False):
            try:
                transcriber_provider = self.registry.get_transcriber_provider(self.args.transcriber_provider)
                providers['transcriber'] = transcriber_provider
            except ValueError:
                pass
            return providers
        
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