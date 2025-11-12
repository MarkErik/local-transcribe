#!/usr/bin/env python3
"""
Aligner plugin using Wav2Vec2 forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from local_transcribe.framework.plugins import AlignerProvider, WordSegment, registry


class Wav2Vec2AlignerProvider(AlignerProvider):
    """Aligner provider using Wav2Vec2 for forced alignment of transcripts to audio."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wav2vec2_model_name = "facebook/wav2vec2-base-960h"  # English model

        # Wav2Vec2 components for alignment
        self.wav2vec2_processor = None
        self.wav2vec2_model = None

    @property
    def name(self) -> str:
        return "wav2vec2"

    @property
    def description(self) -> str:
        return "Wav2Vec2 forced alignment for accurate word-level timestamps"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        return [self.wav2vec2_model_name]

    def get_available_models(self) -> List[str]:
        return ["base-960h"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Wav2Vec2 models to cache."""
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
            cache_dir_wav2vec2 = models_dir / "aligners" / "wav2vec2"
            cache_dir_wav2vec2.mkdir(parents=True, exist_ok=True)

            for model in models:
                if model == "facebook/wav2vec2-base-960h":
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
        """Check which Wav2Vec2 models are available offline without downloading."""
        missing_models = []
        hub_dir = models_dir / "hub"

        for model in models:
            if "/" in model:  # It's a HuggingFace model like Wav2Vec2
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

    def _load_wav2vec2_model(self):
        """Load the Wav2Vec2 model for alignment if not already loaded."""
        if self.wav2vec2_model is None:
            cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "aligners" / "wav2vec2"
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
                    if char:  # Skip empty characters
                        # Calculate timestamp (simplified - assumes uniform time steps)
                        time_step = (i / len(predicted_tokens)) * (len(emissions) / sample_rate * 1000)  # in ms
                        char_timestamps.append((char, time_step))
                prev_token = token

        return char_timestamps

    def _chars_to_words(self, transcript: str, pred_str: str, char_timestamps: List[tuple]) -> List[WordSegment]:
        """Convert character timestamps to word timestamps by aligning with transcript."""
        # Simple alignment: distribute timestamps evenly across words
        # This is a fallback - proper alignment would use dynamic programming

        words = transcript.split()
        if not words:
            return []

        segments = []
        char_idx = 0

        for word in words:
            word_start_char = char_idx
            word_end_char = word_start_char + len(word)

            # Find corresponding character timestamps
            start_time = None
            end_time = None

            for i, (char, time) in enumerate(char_timestamps):
                if i >= word_start_char and start_time is None:
                    start_time = time
                if i >= word_end_char - 1:
                    end_time = time
                    break

            # Fallback if timestamps not found
            if start_time is None:
                start_time = 0
            if end_time is None or end_time <= start_time:
                # Estimate duration based on word length
                word_duration = len(word) * 50  # Rough estimate: 50ms per character
                end_time = start_time + word_duration

            segments.append(WordSegment(
                text=word,
                start=start_time / 1000,  # Convert ms to seconds
                end=end_time / 1000,
                speaker=None
            ))

            char_idx = word_end_char

        return segments

    def align_transcript(
        self,
        audio_path: str,
        transcript: str,
        **kwargs
    ) -> List[WordSegment]:
        """Align transcript text to audio using Wav2Vec2."""
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

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)


def register_aligner_plugins():
    """Register aligner plugins."""
    registry.register_aligner_provider(Wav2Vec2AlignerProvider())


# Auto-register on import
register_aligner_plugins()