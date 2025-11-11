#!/usr/bin/env python3
"""
ASR plugin using IBM Granite with Wav2Vec2 forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, Wav2Vec2Processor, Wav2Vec2ForCTC
from peft import PeftModel
import librosa
from local_transcribe.framework.plugins import ASRProvider, WordSegment, registry


class GraniteWav2Vec2ASRProvider(ASRProvider):
    """ASR provider using IBM Granite with Wav2Vec2 forced alignment for word timestamps."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.granite_model_name = "ibm-granite/granite-speech-3.3-8b"
        self.wav2vec2_model_name = "facebook/wav2vec2-base-960h"  # English model
        
        # Granite components
        self.granite_processor = None
        self.granite_model = None
        self.tokenizer = None
        
        # Wav2Vec2 components for alignment
        self.wav2vec2_processor = None
        self.wav2vec2_model = None

    @property
    def name(self) -> str:
        return "granite-wav2vec2"

    @property
    def description(self) -> str:
        return "IBM Granite ASR with Wav2Vec2 forced alignment for accurate word timestamps"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        return [
            "ibm-granite/granite-speech-3.3-8b",
            "facebook/wav2vec2-base-960h"
        ]

    def get_available_models(self) -> List[str]:
        return ["granite-wav2vec2"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite and Wav2Vec2 models to cache."""
        import os
        import sys
        
        # DEBUG: Log environment state before download attempt
        print(f"DEBUG: HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"DEBUG: HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"DEBUG: HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
        
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        # DEBUG: Confirm environment variable was set
        print(f"DEBUG: HF_HUB_OFFLINE after setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        
        # Force reload of huggingface_hub modules to pick up new environment
        print(f"DEBUG: Reloading huggingface_hub modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            print(f"DEBUG: Reloaded {module_name}")
        
        # Also reload transformers modules
        print(f"DEBUG: Reloading transformers modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            print(f"DEBUG: Reloaded {module_name}")
        
        from huggingface_hub import snapshot_download
        
        try:
            cache_dir_granite = models_dir / "asr" / "granite"
            cache_dir_granite.mkdir(parents=True, exist_ok=True)
            cache_dir_wav2vec2 = models_dir / "asr" / "wav2vec2"
            cache_dir_wav2vec2.mkdir(parents=True, exist_ok=True)
            for model in models:
                if model == "ibm-granite/granite-speech-3.3-8b":
                    # Use snapshot_download to download the entire repo
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, cache_dir=str(cache_dir_granite), token=token)
                    print(f"[✓] {model} downloaded successfully.")
                elif model == "facebook/wav2vec2-base-960h":
                    # Use snapshot_download to download the entire repo
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, cache_dir=str(cache_dir_wav2vec2), token=token)
                    print(f"[✓] {model} downloaded successfully.")
        except Exception as e:
            print(f"DEBUG: Download failed with error: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            
            # Additional debug: Check environment at time of error
            print(f"DEBUG: At error time - HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
            print(f"DEBUG: At error time - HF_HOME: {os.environ.get('HF_HOME')}")
            
            raise Exception(f"Failed to download {model}: {e}")
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite and Wav2Vec2 models are available offline without downloading."""
        missing_models = []
        for model in models:
            if model == "ibm-granite/granite-speech-3.3-8b":
                cache_dir = models_dir / "asr" / "granite"
                # Check for model files (this is a simplified check)
                if not any(cache_dir.rglob("*.bin")) and not any(cache_dir.rglob("*.safetensors")):
                    missing_models.append(model)
            elif model == "facebook/wav2vec2-base-960h":
                cache_dir = models_dir / "asr" / "wav2vec2"
                # Check for model files
                if not any(cache_dir.rglob("*.bin")) and not any(cache_dir.rglob("*.safetensors")):
                    missing_models.append(model)
        return missing_models

    def _load_granite_model(self):
        """Load the Granite model if not already loaded."""
        if self.granite_model is None:
            # Temporarily allow online access to download model if needed
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
            os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "asr" / "granite"
                cache_dir.mkdir(parents=True, exist_ok=True)
                token = os.getenv("HF_TOKEN")
                self.granite_processor = AutoProcessor.from_pretrained(self.granite_model_name, cache_dir=str(cache_dir), local_files_only=True, token=token, revision="main")
                self.tokenizer = self.granite_processor.tokenizer
                
                # Load base model
                base_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.granite_model_name, cache_dir=str(cache_dir), local_files_only=True, token=token, revision="main").to(self.device)
                
                # Check if this is a PEFT model
                try:
                    self.granite_model = PeftModel.from_pretrained(base_model, self.granite_model_name, cache_dir=str(cache_dir), token=token, revision="main").to(self.device)
                except:
                    # If not a PEFT model, use the base model
                    self.granite_model = base_model
            finally:
                # Restore offline mode
                os.environ["HF_HUB_OFFLINE"] = offline_mode

    def _load_wav2vec2_model(self):
        """Load the Wav2Vec2 model for alignment if not already loaded."""
        if self.wav2vec2_model is None:
            # Temporarily allow online access to download model if needed
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
            os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "asr" / "wav2vec2"
                cache_dir.mkdir(parents=True, exist_ok=True)
                token = os.getenv("HF_TOKEN")
                self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_name, cache_dir=str(cache_dir), local_files_only=True, token=token)
                self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_model_name, cache_dir=str(cache_dir), local_files_only=True, token=token).to(self.device)
            finally:
                # Restore offline mode
                os.environ["HF_HUB_OFFLINE"] = offline_mode

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Granite model."""
        self._load_granite_model()

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
        model_inputs = self.granite_processor(
            text,
            wav,
            device=self.device,
            return_tensors="pt",
        ).to(self.device)

        # Generate transcription
        model_outputs = self.granite_model.generate(
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

    def _align_with_wav2vec2(self, audio_path: str, transcript: str) -> List[WordSegment]:
        """Use Wav2Vec2 for forced alignment of transcript to audio."""
        self._load_wav2vec2_model()

        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Process audio for Wav2Vec2
        inputs = self.wav2vec2_processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.wav2vec2_model(**inputs).logits

        # Get predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = predicted_ids[0].cpu().numpy()

        # Decode to text (this gives us character-level alignment)
        pred_str = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
        
        # For word-level alignment, we need to map the transcript words to time positions
        # This is a simplified approach - in practice, you'd want more sophisticated alignment
        
        # Get emission timestamps (time positions for each character)
        emissions = logits[0].cpu().numpy()
        
        # Use CTC alignment to get character timestamps
        char_timestamps = self._get_char_timestamps(emissions, predicted_tokens, sr)
        
        # Convert character timestamps to word timestamps
        return self._chars_to_words(transcript, pred_str, char_timestamps)

    def _get_char_timestamps(self, emissions: np.ndarray, predicted_tokens: np.ndarray, sample_rate: int) -> List[tuple]:
        """Extract character timestamps from CTC emissions."""
        # This is a simplified CTC alignment - in production, you'd use a more robust method
        char_timestamps = []
        
        # Remove consecutive duplicates and blanks (CTC behavior)
        prev_token = None
        for i, token in enumerate(predicted_tokens):
            if token != self.wav2vec2_processor.tokenizer.pad_token_id and token != prev_token:
                if token != self.wav2vec2_processor.tokenizer.unk_token_id:
                    char = self.wav2vec2_processor.tokenizer.decode(token)
                    # Estimate timestamp based on position in emissions
                    timestamp = (i / len(predicted_tokens)) * (len(emissions) / sample_rate)
                    char_timestamps.append((char, timestamp))
            prev_token = token
        
        return char_timestamps

    def _chars_to_words(self, transcript: str, predicted_text: str, char_timestamps: List[tuple]) -> List[WordSegment]:
        """Convert character timestamps to word timestamps by aligning with transcript."""
        words = transcript.split()
        if not words:
            return []

        # Simple alignment: distribute timestamps evenly across words
        # This is a fallback - proper alignment would use dynamic programming
        total_chars = sum(len(word) for word in words)
        if total_chars == 0:
            return []

        segments = []
        char_idx = 0
        
        for word in words:
            word_start_char = char_idx
            word_end_char = char_idx + len(word)
            
            # Find corresponding timestamps
            start_time = 0.0
            end_time = 0.0
            
            # Find characters that match this word
            word_chars_found = 0
            for char, timestamp in char_timestamps:
                if word_chars_found < len(word) and char.lower() in word.lower():
                    if word_chars_found == 0:
                        start_time = timestamp
                    word_chars_found += 1
                elif word_chars_found >= len(word):
                    end_time = timestamp
                    break
            
            if end_time == 0.0:
                end_time = start_time + 0.1  # Fallback duration
            
            segments.append(WordSegment(
                text=word,
                start=start_time,
                end=end_time,
                speaker=None
            ))
            
            char_idx = word_end_char

        return segments

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio using Granite + Wav2Vec2 alignment.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            **kwargs: Additional configuration
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Get transcription from Granite
        transcript = self._transcribe_audio(audio_path)

        # Align with Wav2Vec2
        segments = self._align_with_wav2vec2(audio_path, transcript)

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
    registry.register_asr_provider(GraniteWav2Vec2ASRProvider())


# Auto-register on import
register_asr_plugins()