#!/usr/bin/env python3
"""
Transcriber plugin using IBM Granite.

This provider uses GraniteModelManager for all model management and transcription,
ensuring consistent behavior across all Granite-based transcribers.
"""

from typing import List, Optional, Union, Dict, Any
import os
import pathlib
import math
import librosa
import torch
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug
from local_transcribe.providers.common.granite_model_manager import GraniteModelManager


class GraniteTranscriberProvider(TranscriberProvider):
    """Transcriber provider using IBM Granite for speech-to-text transcription.
    
    Uses GraniteModelManager for consolidated model management and transcription.
    """

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Granite Transcriber Provider")
        
        # Initialize model manager for all Granite operations
        self.model_manager = GraniteModelManager(self.logger)
        
        # Chunking configuration
        self.chunk_length_seconds = 60.0
        self.overlap_seconds = 3.0
        self.min_chunk_seconds = 6.0
        
        # Track selected model
        self.selected_model: Optional[str] = None
        self.models_dir: Optional[pathlib.Path] = None

    @property
    def device(self):
        return get_system_capability()

    @property
    def name(self) -> str:
        return "granite"

    @property
    def short_name(self) -> str:
        return "IBM Granite"

    @property
    def description(self) -> str:
        return "IBM Granite transcription (8B or 2B) for speech-to-text"

    @property
    def has_builtin_alignment(self) -> bool:
        return False

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        return self.model_manager.get_required_models(selected_model)

    def get_available_models(self) -> List[str]:
        return list(self.model_manager.MODEL_MAPPING.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        self.model_manager.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline without downloading."""
        return self.model_manager.check_models_available_offline(models, models_dir)

    def _load_model(self) -> None:
        """Load the Granite model if not already loaded."""
        if self.model_manager.model is None:
            # Set selected model in the manager
            self.model_manager.selected_model = self.selected_model
            
            # Load the model
            model_name = self.model_manager.get_required_models()[0]
            self.model_manager._load_model(model_name)

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Transcribe audio using local Granite model.
        
        Args:
            audio_path: Path to the audio file
            device: Device for processing
            **kwargs: Additional options including:
                - transcriber_model: Model to use (granite-2b or granite-8b)
                - models_dir: Directory for models
        
        Returns:
            List of dictionaries with chunk data
        """
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')
        if transcriber_model not in self.model_manager.MODEL_MAPPING:
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        self.model_manager.selected_model = transcriber_model
        
        # Set models_dir if provided in kwargs
        if 'models_dir' in kwargs:
            self.models_dir = pathlib.Path(kwargs['models_dir'])
        
        # Load the model
        self._load_model()

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Calculate audio duration in seconds
        duration = len(wav) / sr
        
        # For short audio (less than chunk_length), transcribe directly without chunking
        if duration < self.chunk_length_seconds:
            if duration < 1.0:
                raise ValueError(f"Audio duration ({duration:.1f}s) is too short for transcription. Please provide audio longer than 1 second.")
            log_progress(f"Audio duration: {duration:.1f}s - transcribing as single segment")
            text = self.model_manager.transcribe_segment(wav, int(sr))
            return [{"chunk_id": 0, "words": text.split(), "text": text}]
        
        # Calculate number of chunks accounting for overlap
        effective_chunk_length = self.chunk_length_seconds - self.overlap_seconds
        num_chunks = math.ceil(duration / effective_chunk_length) if effective_chunk_length > 0 else 1
        
        # Process in chunks for longer audio
        log_progress(f"Audio duration: {duration:.1f}s - processing in {num_chunks} chunks to manage memory")
        return self._transcribe_chunked(wav, int(sr))

    def _transcribe_chunked(self, wav, sr: int) -> List[Dict[str, Any]]:
        """Transcribe audio in chunks to manage memory for long files."""
        chunk_samples = int(self.chunk_length_seconds * sr)
        overlap_samples = int(self.overlap_seconds * sr)
        min_chunk_samples = int(self.min_chunk_seconds * sr)

        chunks = []
        total_samples = len(wav)
        total_chunks = math.ceil(total_samples / (chunk_samples - overlap_samples))
        chunk_start = 0
        chunk_num = 0
        prev_chunk_wav = None
        
        while chunk_start < total_samples:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            chunk_wav = wav[chunk_start:chunk_end]
            
            chunk_duration_sec = len(chunk_wav) / sr
            log_progress(f"Processing chunk {chunk_num} of {total_chunks} ({chunk_duration_sec:.1f}s)...")
            
            if len(chunk_wav) < min_chunk_samples:
                if prev_chunk_wav is not None:
                    # Merge with previous chunk
                    non_overlapping_part = chunk_wav[overlap_samples:]
                    merged_tensor = torch.cat([torch.from_numpy(prev_chunk_wav), torch.from_numpy(non_overlapping_part)])
                    merged_wav = merged_tensor.numpy()
                    
                    # Use consolidated transcription method
                    chunk_text = self.model_manager.transcribe_segment(merged_wav, sr)
                    existing_id = chunks[-1]["chunk_id"]
                    chunks[-1] = {"chunk_id": existing_id, "words": chunk_text.split()}
            else:
                # Normal chunk processing using consolidated transcription method
                chunk_text = self.model_manager.transcribe_segment(chunk_wav, sr)
                words = chunk_text.split()
                chunks.append({"chunk_id": chunk_num, "words": words})
            
            prev_chunk_wav = chunk_wav

            if chunk_end == total_samples:
                break
            
            chunk_start = chunk_start + chunk_samples - overlap_samples
        
        return chunks

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Not implemented for pure transcribers - use with an aligner.
        This method raises NotImplementedError.
        """
        raise NotImplementedError("Pure transcribers require an aligner. Use transcribe() + align_transcript() instead.")

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.model_manager.ensure_models_available(models, models_dir)


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()