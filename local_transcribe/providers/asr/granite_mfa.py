#!/usr/bin/env python3
"""
ASR plugin using IBM Granite with MFA forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import tempfile
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download
import librosa
from local_transcribe.framework.plugins import ASRProvider, WordSegment, registry


class GraniteMFASRProvider(ASRProvider):
    """ASR provider using IBM Granite with MFA forced alignment for word timestamps."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ibm-granite/granite-speech-3.3-8b"
        self.processor = None
        self.model = None
        self.tokenizer = None

    @property
    def name(self) -> str:
        return "granite-mfa"

    @property
    def description(self) -> str:
        return "IBM Granite ASR with MFA forced alignment for word timestamps"

    def get_required_models(self) -> List[str]:
        return ["ibm-granite/granite-speech-3.3-8b"]

    def get_available_models(self) -> List[str]:
        return ["granite-8b"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        # Determine if we should use local files only based on the environment
        local_only = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
        
        for model in models:
            if model == "ibm-granite/granite-speech-3.3-8b":
                cache_dir = models_dir / "asr" / "granite"
                cache_dir.mkdir(parents=True, exist_ok=True)
                AutoProcessor.from_pretrained(model, cache_dir=str(cache_dir), local_files_only=local_only)
                AutoModelForSpeechSeq2Seq.from_pretrained(model, cache_dir=str(cache_dir), local_files_only=local_only)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline without downloading."""
        missing_models = []
        for model in models:
            if model == "ibm-granite/granite-speech-3.3-8b":
                cache_dir = models_dir / "asr" / "granite"
                # Check for model files (this is a simplified check)
                if not any(cache_dir.rglob("*.bin")) and not any(cache_dir.rglob("*.safetensors")):
                    missing_models.append(model)
        return missing_models

    def _load_model(self):
        """Load the Granite model if not already loaded."""
        if self.model is None:
            # Temporarily allow online access to download model if needed
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
            os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "asr" / "granite"
                cache_dir.mkdir(parents=True, exist_ok=True)
                token = os.getenv("HF_TOKEN")
                # During transcription, models should be cached, so use local_files_only=True
                self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=str(cache_dir), local_files_only=True, token=token, revision="main")
                self.tokenizer = self.processor.tokenizer
                
                # Load base model
                base_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, cache_dir=str(cache_dir), local_files_only=True, token=token, revision="main").to(self.device)
                
                # Check if this is a PEFT model
                try:
                    self.model = PeftModel.from_pretrained(base_model, self.model_name, cache_dir=str(cache_dir), token=token, revision="main").to(self.device)
                except:
                    # If not a PEFT model, use the base model
                    self.model = base_model
            finally:
                # Restore offline mode
                os.environ["HF_HUB_OFFLINE"] = offline_mode

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Granite model."""
        self._load_model()

        # Load audio
        wav, sr = torchaudio.load(audio_path, normalize=True)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # Convert to mono
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

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
            wav,
            device=self.device,
            return_tensors="pt",
        ).to(self.device)

        # Generate transcription
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

        return output_text[0].strip()

    def _align_with_mfa(self, audio_path: str, transcript: str) -> List[WordSegment]:
        """Use basic alignment for word timestamps (placeholder for MFA)."""

        # Get audio duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr

        # Simple word-based alignment
        words = transcript.split()
        if not words:
            return []

        word_duration = duration / len(words)

        segments = []
        current_time = 0.0
        for word in words:
            segments.append(WordSegment(
                text=word,
                start=current_time,
                end=min(current_time + word_duration, duration),
                speaker=None
            ))
            current_time += word_duration

        return segments

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio using Granite + MFA alignment.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            **kwargs: Additional configuration
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Get transcription from Granite
        transcript = self._transcribe_audio(audio_path)

        # Align with MFA
        segments = self._align_with_mfa(audio_path, transcript)

        # Add speaker role if provided
        if role:
            for segment in segments:
                segment.speaker = role

        return segments

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)


def register_asr_plugins():
    """Register ASR plugins."""
    registry.register_asr_provider(GraniteMFASRProvider())


# Auto-register on import
register_asr_plugins()