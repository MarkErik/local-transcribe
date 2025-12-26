#!/usr/bin/env python3
"""
Transcriber plugin using IBM Granite.
"""

from typing import List, Optional, Union, Dict, Any
import os
import pathlib
import re
import sys
import torch
import math
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download, snapshot_download
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug
from local_transcribe.providers.common.granite_model_manager import GraniteModelManager


class GraniteTranscriberProvider(TranscriberProvider):
    """Transcriber provider using IBM Granite for speech-to-text transcription."""

    def __init__(self):
        # Device is determined dynamically
        self.model_mapping = {
            "granite-2b": "ibm-granite/granite-speech-3.3-2b",
            "granite-8b": "ibm-granite/granite-speech-3.3-8b"
        }
        self.selected_model = None  # Will be set during transcription
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.models_dir = None  # Will be set when needed
        self.chunk_length_seconds = 60.0  # Configurable chunk length in seconds
        self.overlap_seconds = 3.0  # Configurable overlap between chunks in seconds
        self.min_chunk_seconds = 6.0  # Configurable minimum chunk length in seconds
        self.logger = get_logger()
        
        # Remote transcription settings
        self.use_remote_granite: bool = False
        self.remote_granite_url: Optional[str] = None
        self._model_manager: Optional[GraniteModelManager] = None
    
    def _get_model_manager(self) -> GraniteModelManager:
        """Get or create the GraniteModelManager instance."""
        if self._model_manager is None:
            self._model_manager = GraniteModelManager(self.logger, self.models_dir)
        return self._model_manager

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
        return "IBM Granite transcription (2B/8B) for speech-to-text"

    @property
    def has_builtin_alignment(self) -> bool:
        return False

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model and selected_model in self.model_mapping:
            return [self.model_mapping[selected_model]]
        # Default to 8b model if no selection
        return [self.model_mapping["granite-8b"]]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        # Use the GraniteModelManager for consistent model management
        model_manager = GraniteModelManager(self.logger, models_dir)
        model_manager.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline without downloading."""
        # Use the GraniteModelManager for consistent model management
        model_manager = GraniteModelManager(self.logger, models_dir)
        return model_manager.check_models_available_offline(models, models_dir)

    def _load_model(self):
        """Load the Granite model if not already loaded."""
        if self.model is None:
            # Use the GraniteModelManager for consistent model loading
            model_manager = GraniteModelManager(self.logger, self.models_dir)
            
            # Get the model name
            if self.selected_model and self.selected_model in self.model_mapping:
                model_name = self.model_mapping[self.selected_model]
            else:
                model_name = self.model_mapping["granite-8b"]
            
            # Load the model using the model manager
            model_manager._load_model(model_name)
            
            # Copy the loaded components from the model manager
            self.processor = model_manager.processor
            self.model = model_manager.model
            self.tokenizer = model_manager.tokenizer

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Transcribe audio using Granite model.
        
        Supports both local and remote Granite transcription. Configure remote
        transcription by passing:
            use_remote_granite=True
            remote_granite_url="http://server:7070"
        
        Args:
            audio_path: Path to the audio file
            device: Device for processing
            **kwargs: Additional options including:
                - transcriber_model: Model to use (granite-2b or granite-8b)
                - models_dir: Directory for models
                - use_remote_granite: Use remote Granite server (default: False)
                - remote_granite_url: URL of remote Granite server
        
        Returns:
            List of dictionaries with chunk data
        """
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')  # Default to 8b
        if transcriber_model not in self.model_mapping:
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        
        # Set models_dir if provided in kwargs
        if 'models_dir' in kwargs:
            self.models_dir = kwargs['models_dir']
        
        # Configure remote Granite if requested
        self.use_remote_granite = kwargs.get('use_remote_granite', False)
        self.remote_granite_url = kwargs.get('remote_granite_url')
        
        model_manager = self._get_model_manager()
        
        if self.use_remote_granite:
            log_progress(f"Remote Granite transcription enabled: {self.remote_granite_url or 'default URL'}")
            model_manager.configure_remote(
                use_remote=True,
                remote_url=self.remote_granite_url
            )
            
            # Check if remote is available
            if not model_manager.is_remote_available():
                self.logger.warning("Remote Granite server not available, falling back to local")
                self.use_remote_granite = False
                model_manager.configure_remote(use_remote=False)
        else:
            model_manager.configure_remote(use_remote=False)
        
        # Only load local model if not using remote
        if not self.use_remote_granite:
            self._load_model()
        else:
            log_progress("Using remote Granite server - skipping local model load")

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Calculate audio duration in seconds
        duration = len(wav) / sr
        
        # Validate audio length - must be at least as long as chunk length
        if duration < self.chunk_length_seconds:
            raise ValueError(f"Audio duration ({duration:.1f}s) is shorter than the minimum required chunk length ({self.chunk_length_seconds}s). Please provide audio longer than {self.chunk_length_seconds}s for transcription.")
        
        # Calculate number of chunks accounting for overlap
        effective_chunk_length = self.chunk_length_seconds - self.overlap_seconds
        num_chunks = math.ceil(duration / effective_chunk_length) if effective_chunk_length > 0 else 1
        
        # Always process in chunks and return chunked output
        log_progress(f"Audio duration: {duration:.1f}s - processing in {num_chunks} chunks to manage memory")
        return self._transcribe_chunked(wav, sr, **kwargs)

    def _transcribe_single_chunk(self, wav, sample_rate: int = 16000, **kwargs) -> str:
        """Transcribe a single audio chunk (local or remote)."""
        segment_duration = len(wav) / sample_rate
        
        # Check if we should use remote transcription
        model_manager = self._get_model_manager()
        if self.use_remote_granite and model_manager.should_use_remote():
            log_progress(f"Transcribing {segment_duration:.1f}s chunk with remote Granite server")
            try:
                text = model_manager.transcribe_remote(
                    audio=wav,
                    sample_rate=sample_rate,
                    segment_duration=segment_duration,
                    include_disfluencies=True
                )
                # Apply local cleaning
                cleaned_text = self._clean_transcription_output(text)
                return cleaned_text
            except Exception as e:
                self.logger.warning(f"Remote transcription failed, falling back to local: {e}")
                # Load local model if not already loaded
                if self.model is None:
                    self._load_model()
        
        # Local transcription
        try:
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)

            # Create text prompt
            chat = [
                {
                    "role": "system",
                    "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
                },
                {
                    "role": "user",
                    "content": "<|audio|>can you transcribe the speech into a written format?",
                }
            ]

            text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            model_inputs = self.processor(
                text,
                wav_tensor,
                device=self.device,
                return_tensors="pt",
            ).to(self.device)

            # Generate transcription with no_grad to prevent gradient accumulation
            with torch.no_grad():
                model_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    repetition_penalty=3.0,
                    length_penalty=1.0,
                    temperature=1.0,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Extract generated text
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

            output_text = self.tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )

            # Post-process the output to remove dialogue markers and quotation marks
            cleaned_text = self._clean_transcription_output(output_text[0].strip())

            return cleaned_text
            
        finally:
            # Explicitly clean up tensors and memory
            if 'wav_tensor' in locals():
                del wav_tensor
            if 'model_inputs' in locals():
                # Delete individual tensors from model_inputs dict
                for key in list(model_inputs.keys()):
                    del model_inputs[key]
                del model_inputs
            if 'model_outputs' in locals():
                del model_outputs
            if 'new_tokens' in locals():
                del new_tokens
            
            # Force garbage collection and empty cache
            import gc
            gc.collect()
            
            clear_device_cache()


    def _transcribe_chunked(self, wav, sr, **kwargs) -> List[Dict[str, Any]]:
        """Transcribe audio in chunks to manage memory for long files."""

        chunk_samples = int(self.chunk_length_seconds * sr)
        overlap_samples = int(self.overlap_seconds * sr)  # Configurable overlap to avoid cutting words
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
                    # Only append the non-overlapping part of the new chunk to avoid duplication
                    non_overlapping_part = chunk_wav[overlap_samples:]
                    merged_tensor = torch.cat([torch.from_numpy(prev_chunk_wav), torch.from_numpy(non_overlapping_part)])
                    merged_wav = merged_tensor.numpy()
                    
                    chunk_text = self._transcribe_single_chunk(merged_wav, **kwargs)
                    # Update the last chunk in results
                    existing_id = chunks[-1]["chunk_id"]
                    chunks[-1] = {"chunk_id": existing_id, "words": chunk_text.split()}
            else:
                # Normal chunk processing
                chunk_text = self._transcribe_single_chunk(chunk_wav, **kwargs)
                words = chunk_text.split()
                chunks.append({"chunk_id": chunk_num, "words": words})
            
            prev_chunk_wav = chunk_wav

            # Optimization: If this chunk reached the end of the file, we are done.
            # Any subsequent chunk would just be a subset of this one (due to overlap).
            if chunk_end == total_samples:
                break
            
            
            # Move to next chunk: advance by chunk size minus overlap
            chunk_start = chunk_start + chunk_samples - overlap_samples
        
        return chunks
            
    def _clean_transcription_output(self, text: str) -> str:
        """
        Clean the transcription output for a single chunk by removing dialogue markers and quotation marks.
        
        Args:
            text: Raw transcription output from the model for a single chunk
            
        Returns:
            Cleaned transcription text for a single chunk
        """
        # Count labels before removal for debug logging
        user_count = len(re.findall(r'\bUser:\s*', text, flags=re.IGNORECASE))
        assistant_count = len(re.findall(r'\bAI Assistant:\s*', text, flags=re.IGNORECASE))
        assistant_short_count = len(re.findall(r'\bAssistant:\s*', text, flags=re.IGNORECASE))
        total_removed = user_count + assistant_count + assistant_short_count
        
        # Remove "User:" and "AI Assistant:" labels (case insensitive)
        text = re.sub(r'\bUser:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAI Assistant:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAssistant:\s*', '', text, flags=re.IGNORECASE)
        
        # Remove all types of quotation marks using Unicode escape sequences
        text = text.replace('"', '')  # Straight double quote (ASCII 34)
        text = text.replace('\u201C', '')  # Curly double quote left (Unicode 8220)
        text = text.replace('\u201D', '')  # Curly double quote right (Unicode 8221)
        
        # Clean up extra whitespace that might result from removals
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Log count if any labels were removed
        if total_removed > 0:
            log_debug(f"Removed {total_removed} labels from chunk transcript.")
        
        return text

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
        # Use the GraniteModelManager for consistent model management
        model_manager = GraniteModelManager(self.logger, models_dir)
        model_manager.ensure_models_available(models, models_dir)


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()