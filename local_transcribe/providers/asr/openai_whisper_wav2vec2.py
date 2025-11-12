#!/usr/bin/env python3
"""
ASR plugin using OpenAI Whisper with Wav2Vec2 forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from local_transcribe.framework.plugins import ASRProvider, WordSegment, registry


class OpenAIWhisperASRProvider(ASRProvider):
    """ASR provider using OpenAI Whisper with Wav2Vec2 forced alignment for word timestamps."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Model mapping: user-friendly name -> OpenAI Whisper model name
        self.model_mapping = {
            "tiny.en": "tiny.en",
            "base.en": "base.en",
            "small.en": "small.en"
        }
        self.selected_model = None  # Will be set during transcription
        self.wav2vec2_model_name = "facebook/wav2vec2-base-960h"  # English model

        # Whisper components
        self.whisper_model = None

        # Wav2Vec2 components for alignment
        self.wav2vec2_processor = None
        self.wav2vec2_model = None

    @property
    def name(self) -> str:
        return "openai-whisper"

    @property
    def description(self) -> str:
        return "OpenAI Whisper ASR (tiny.en/base.en/small.en) with Wav2Vec2 forced alignment for accurate word timestamps"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model and selected_model in self.model_mapping:
            return [
                self.model_mapping[selected_model],  # Whisper model name
                self.wav2vec2_model_name  # Wav2Vec2 model for forced alignment
            ]
        # Default to base.en model
        return [
            self.model_mapping["base.en"],
            self.wav2vec2_model_name
        ]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Whisper and Wav2Vec2 models to cache."""
        import os
        import sys

        # DEBUG: Log environment state before download attempt
        print(f"DEBUG: HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"DEBUG: HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"DEBUG: HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
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
            cache_dir_whisper = models_dir / "asr" / "whisper"
            cache_dir_whisper.mkdir(parents=True, exist_ok=True)
            cache_dir_wav2vec2 = models_dir / "asr" / "wav2vec2"
            cache_dir_wav2vec2.mkdir(parents=True, exist_ok=True)

            for model in models:
                if model in self.model_mapping.values():  # It's a Whisper model
                    try:
                        import whisper
                        # Load and immediately discard to download/cache the model
                        temp_model = whisper.load_model(model, download_root=str(cache_dir_whisper))
                        del temp_model  # Free memory immediately
                        print(f"[✓] Whisper {model} downloaded successfully.")
                    except ImportError:
                        print(f"Warning: openai-whisper not available, skipping {model}")
                elif model == "facebook/wav2vec2-base-960h":
                    os.environ["HF_HOME"] = str(cache_dir_wav2vec2)
                    # Use snapshot_download to download the entire repo
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, token=token)
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
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Whisper and Wav2Vec2 models are available offline without downloading."""
        missing_models = []
        hub_dir = models_dir / "hub"
        whisper_cache_dir = models_dir / "asr" / "whisper"

        for model in models:
            if model in self.model_mapping.values():  # It's a Whisper model
                # Check if Whisper model exists in cache
                # Whisper stores models as .pt files with the model name
                model_file = whisper_cache_dir / f"{model}.pt"
                if not model_file.exists():
                    missing_models.append(model)
            elif "/" in model:  # It's a HuggingFace model like Wav2Vec2
                org, repo = model.split("/")
                model_dir_name = f"models--{org}--{repo.replace('/', '--')}"
                model_cache_dir = hub_dir / model_dir_name

                # Check if the model directory exists and contains model files
                has_model_files = (
                    model_cache_dir.exists() and (
                        any(model_cache_dir.rglob("*.bin")) or
                        any(model_cache_dir.rglob("*.safetensors")) or
                        any(model_cache_dir.rglob("*.pt")) or
                        any(model_cache_dir.rglob("*.pth"))
                    )
                )

                if not has_model_files:
                    missing_models.append(model)

        return missing_models

    def _load_whisper_model(self):
        """Load the Whisper model if not already loaded."""
        if self.whisper_model is None:
            try:
                import whisper
                # Get the actual model name from selected model
                model_name = self.model_mapping.get(self.selected_model, "base.en")
                
                # Set up cache directory consistent with other providers
                models_root = pathlib.Path(os.getenv("HF_HOME", "./models")).resolve()
                whisper_cache_dir = models_root / "asr" / "whisper"
                whisper_cache_dir.mkdir(parents=True, exist_ok=True)
                
                self.whisper_model = whisper.load_model(model_name, download_root=str(whisper_cache_dir))
            except ImportError:
                raise ImportError("openai-whisper package is required. Install with: pip install openai-whisper")

    def _load_wav2vec2_model(self):
        """Load the Wav2Vec2 model for alignment if not already loaded."""
        if self.wav2vec2_model is None:
            cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "asr" / "wav2vec2"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Set HF_HOME to our cache directory
            original_hf_home = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = str(cache_dir)

            try:
                token = os.getenv("HF_TOKEN")
                self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_name, local_files_only=True, token=token)
                self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_model_name, local_files_only=True, token=token).to(self.device)
            finally:
                if original_hf_home is not None:
                    os.environ["HF_HOME"] = original_hf_home
                else:
                    os.environ.pop("HF_HOME", None)

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper model."""
        self._load_whisper_model()

        # Transcribe with Whisper
        result = self.whisper_model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()

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
        Transcribe audio using OpenAI Whisper + Wav2Vec2 alignment.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            **kwargs: Additional configuration, should include 'asr_model'
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Set selected model from kwargs
        self.selected_model = kwargs.get('asr_model', 'base.en')

        # Get transcription from Whisper
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
    registry.register_asr_provider(OpenAIWhisperASRProvider())


# Auto-register on import
register_asr_plugins()