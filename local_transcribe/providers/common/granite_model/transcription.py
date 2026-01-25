# Granite Model Transcription
# Handles audio transcription using Granite models

import gc
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from local_transcribe.lib.system_capability_utils import clear_device_cache
from local_transcribe.lib.program_logger import log_progress, log_debug

if TYPE_CHECKING:
    import torch


class TranscriptionMixin:
    """Mixin class providing transcription functionality for GraniteModelManager.
    
    This mixin provides all transcription-related methods including:
    - Audio segment transcription
    - Output cleaning and prompt fragment stripping
    - Generation parameter calculation
    """
    
    # Type hints for attributes provided by the host class
    model: Optional[Any]
    processor: Optional[Any]
    tokenizer: Optional[Any]
    device: str
    
    # Single source of truth for the transcription prompt
    # This is used both when sending to the model and when cleaning the output
    _TRANSCRIPTION_PROMPT = (
        "can you transcribe the speech into a written format? "
        "make sure to include disfluencies and repeated words."
    )
    
    def _clean_transcription_output(self, text: str) -> str:
        """Clean the transcription output by removing artifacts from model generation.
        
        The Granite model sometimes includes unwanted content in its output:
        - Dialogue markers like "User:" or "Assistant:"
        - Quotation marks around the transcription
        - Echoed prompt fragments
        
        This method removes all such artifacts to produce clean transcription text.
        
        Args:
            text: Raw transcription output from the model
            
        Returns:
            Cleaned transcription text
        """
        if not text:
            return text
        
        # Step 1: Remove dialogue markers
        user_count = len(re.findall(r'\bUser:\s*', text, flags=re.IGNORECASE))
        assistant_count = len(re.findall(r'\bAI Assistant:\s*', text, flags=re.IGNORECASE))
        assistant_short_count = len(re.findall(r'\bAssistant:\s*', text, flags=re.IGNORECASE))
        total_removed = user_count + assistant_count + assistant_short_count
        
        text = re.sub(r'\bUser:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAI Assistant:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAssistant:\s*', '', text, flags=re.IGNORECASE)
        
        if total_removed > 0:
            log_debug(f"Removed {total_removed} dialogue labels from transcript.")
        
        # Step 2: Remove quotation marks (straight and curly)
        text = text.replace('"', '')
        text = text.replace('\u201C', '')  # Left double quotation mark
        text = text.replace('\u201D', '')  # Right double quotation mark
        
        # Step 3: Strip echoed prompt fragments
        # Build fragments from the canonical prompt to ensure they stay in sync
        # Split by sentence boundaries so we catch partial echoes (e.g., just the second sentence)
        prompt_fragments = []
        for sentence in re.split(r'[.?!]\s*', self._TRANSCRIPTION_PROMPT.lower()):
            sentence = sentence.strip()
            if sentence:
                prompt_fragments.append(sentence)
        
        for fragment in prompt_fragments:
            text = re.sub(re.escape(fragment), '', text, flags=re.IGNORECASE)
        
        # Step 4: Final cleanup - normalize whitespace and trim
        text = text.rstrip(" .,\n\t")
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_generation_params(self, segment_duration: float, input_length: int) -> Dict[str, Any]:
        """Calculate generation parameters based on segment duration.
        
        Args:
            segment_duration: Duration of audio segment in seconds
            input_length: Length of input token IDs for RepetitionPenaltyLogitsProcessor
            
        Returns:
            Dictionary with max_new_tokens and optional logits_processor
        """
        from transformers import RepetitionPenaltyLogitsProcessor
        
        if segment_duration < 8.0:
            # For short segments: reduce max_new_tokens, exclude logits_processor
            return {
                "max_new_tokens": 128,
                "logits_processor": None
            }
        elif segment_duration < 20.0:
            # For medium segments
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                penalty=3.0,
                prompt_ignore_length=input_length,
            )
            return {
                "max_new_tokens": 256,
                "logits_processor": [repetition_penalty_processor]
            }
        else:
            # For long segments (20+ seconds)
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                penalty=3.0,
                prompt_ignore_length=input_length,
            )
            return {
                "max_new_tokens": 512,
                "logits_processor": [repetition_penalty_processor]
            }
    
    def transcribe_segment(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000,
        **kwargs
    ) -> str:
        """
        Transcribe a single audio segment using local Granite model.
        
        This is the main transcription method that should be used by all Granite-based
        transcriber providers. It handles:
        - Duration-based parameter calculation
        - Memory cleanup
        - Output cleaning and prompt fragment stripping
        
        Args:
            audio: Audio samples as numpy array (mono, float32, 16kHz)
            sample_rate: Sample rate (should be 16000)
            **kwargs: Additional options (unused, for compatibility)
        
        Returns:
            Cleaned transcription text
        """
        return self._transcribe_segment_local(audio, sample_rate)
    
    def _transcribe_segment_local(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe a single audio segment using local Granite model.
        
        Args:
            audio: Audio samples as numpy array (mono, float32, 16kHz)
            sample_rate: Sample rate (should be 16000)
        
        Returns:
            Cleaned transcription text
        """
        log_progress("Transcribing audio segment with Granite")
        
        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Granite model not loaded. Call _load_model() first.")
        if self.processor is None:
            raise RuntimeError("Granite processor not loaded. Call _load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Granite tokenizer not loaded. Call _load_model() first.")
        
        try:
            # Lazy import of torch
            import torch
            
            wav_tensor = torch.from_numpy(audio).unsqueeze(0)
            segment_duration = len(audio) / sample_rate
            
            # Build chat prompt
            chat = [
                {
                    "role": "system",
                    "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
                },
                {
                    "role": "user",
                    "content": f"<|audio|>{self._TRANSCRIPTION_PROMPT}",
                }
            ]
            
            text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            
            model_inputs = self.processor.__call__(
                text,
                wav_tensor,
                device=self.device,
                return_tensors="pt",
            ).to(self.device)
            
            # Calculate generation parameters based on segment duration
            gen_params = self._calculate_generation_params(
                segment_duration,
                model_inputs["input_ids"].shape[-1]
            )
            
            with torch.no_grad():
                model_outputs = self.model.generate.__call__(
                    **model_inputs,
                    max_new_tokens=gen_params["max_new_tokens"],
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    length_penalty=1.0,
                    temperature=1.0,
                    early_stopping=True,
                    logits_processor=gen_params["logits_processor"],
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
            
            output_text = self.tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )
            
            return self._clean_transcription_output(output_text[0].strip())
            
        finally:
            # Explicit memory cleanup
            if 'wav_tensor' in locals():
                del wav_tensor
            if 'model_inputs' in locals():
                for key in list(model_inputs.keys()):
                    del model_inputs[key]
                del model_inputs
            if 'model_outputs' in locals():
                del model_outputs
            if 'new_tokens' in locals():
                del new_tokens
            
            gc.collect()
            clear_device_cache()
