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
from local_transcribe.framework.plugin_interfaces import AlignerProvider, WordSegment, registry


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
    def short_name(self) -> str:
        return "Wav2Vec2"

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
            # Use XDG_CACHE_HOME as the base (which is set to models/.xdg)
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                models_root = pathlib.Path(xdg_cache_home)
            else:
                # Fallback to standard HuggingFace cache location
                models_root = pathlib.Path.home() / ".cache" / "huggingface"
            
            # Create the standard HuggingFace hub directory structure
            cache_dir = models_root / "huggingface" / "hub"
            cache_dir.mkdir(parents=True, exist_ok=True)

            for model in models:
                if model == "facebook/wav2vec2-base-960h":
                    # Use snapshot_download with cache_dir parameter pointing to the standard location
                    snapshot_download(model, cache_dir=cache_dir, token=os.getenv("HF_TOKEN"))
                    print(f"[✓] {model} downloaded successfully.")
                else:
                    print(f"Warning: Unknown model {model}, skipping download")
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
        """Check which Wav2Vec2 models are available offline without downloading."""
        missing_models = []
        
        # Use XDG_CACHE_HOME as the base (which is set to models/.xdg)
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            models_root = pathlib.Path(xdg_cache_home)
        else:
            # Fallback to standard HuggingFace cache location
            models_root = pathlib.Path.home() / ".cache" / "huggingface"
        
        # Models are stored in standard HuggingFace hub structure
        hub_dir = models_root / "huggingface" / "hub"

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
            # Use XDG_CACHE_HOME as the base (which is set to models/.xdg)
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                models_root = pathlib.Path(xdg_cache_home)
            else:
                # Fallback to standard HuggingFace cache location
                models_root = pathlib.Path.home() / ".cache" / "huggingface"
            
            # The models are stored in the standard HuggingFace hub structure
            # We don't need to set HF_HOME, just let transformers find the models in the standard location

            try:
                token = os.getenv("HF_TOKEN")
                # Models should be cached by preload_models, so use local_files_only=True
                # Let transformers find models in the standard HuggingFace cache location
                self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_name, local_files_only=True, token=token)
                self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_model_name, local_files_only=True, token=token).to(self.device)
            except Exception as e:
                print(f"DEBUG: Failed to load Wav2Vec2 model {self.wav2vec2_model_name}")
                print(f"DEBUG: Cache directory exists: {(models_root / 'huggingface' / 'hub').exists()}")
                if (models_root / 'huggingface' / 'hub').exists():
                    print(f"DEBUG: Cache directory contents: {list((models_root / 'huggingface' / 'hub').iterdir())}")
                raise e

    def _get_char_timestamps(self, emissions: np.ndarray, predicted_tokens: np.ndarray, sample_rate: int) -> List[tuple]:
        """Extract character timestamps from CTC emissions using improved alignment."""
        char_timestamps = []
        
        # Get the probability distribution over time for each token
        time_steps = emissions.shape[0]  # Number of time steps
        
        # Remove consecutive duplicates and blanks (CTC behavior)
        prev_token = None
        token_indices = []
        
        for i, token in enumerate(predicted_tokens):
            if (token != self.wav2vec2_processor.tokenizer.pad_token_id and
                token != prev_token and
                token != self.wav2vec2_processor.tokenizer.unk_token_id):
                
                char = self.wav2vec2_processor.tokenizer.decode(token)
                if char:  # Skip empty characters
                    token_indices.append((i, token, char))
                prev_token = token
        
        if not token_indices:
            return char_timestamps
        
        # For each token, find the most likely time position
        for token_idx, token, char in token_indices:
            # Get the probability of this token across all time steps
            token_probs = emissions[token_idx, :]
            
            # Find the peak probability (most likely time position)
            max_prob_time = np.argmax(token_probs)
            
            # Convert to milliseconds, accounting for model downsampling
            inputs_to_logits_ratio = self.wav2vec2_model.config.inputs_to_logits_ratio
            time_ms = max_prob_time * (inputs_to_logits_ratio / sample_rate) * 1000
            
            char_timestamps.append((char, time_ms))
        
        return char_timestamps

    def _chars_to_words(self, transcript: str, pred_str: str, char_timestamps: List[tuple]) -> List[WordSegment]:
        """Convert character timestamps to word timestamps using improved alignment."""
        words = transcript.split()
        if not words:
            return []

        segments = []
        
        # Create a mapping from character position to timestamp
        char_to_time = {}
        for i, (char, time) in enumerate(char_timestamps):
            char_to_time[i] = time
        
        # If we have no character timestamps, fall back to simple word timing
        if not char_to_time:
            return self._fallback_word_timing(words, len(char_timestamps))
        
        total_chars = sum(len(word) for word in words)
        char_idx = 0

        for word in words:
            word_len = len(word)
            word_start_char = char_idx
            word_end_char = char_idx + word_len

            # Find timestamps for the first and last characters of this word
            start_time = None
            end_time = None
            
            # Look for timestamps within the word's character range
            word_timestamps = []
            for char_pos in range(word_start_char, min(word_end_char, len(char_timestamps))):
                if char_pos in char_to_time:
                    word_timestamps.append(char_to_time[char_pos])
            
            if word_timestamps:
                # Use the earliest and latest timestamps in this word
                start_time = min(word_timestamps)
                end_time = max(word_timestamps)
            else:
                # If no direct timestamps, interpolate from surrounding characters
                start_time, end_time = self._interpolate_word_timing(
                    word_start_char, word_end_char, char_to_time, total_chars
                )

            # Ensure reasonable timing
            if start_time is None:
                start_time = 0
            if end_time is None or end_time <= start_time:
                # Estimate duration based on word length and speech rate
                word_duration = max(len(word) * 75, 200)  # At least 200ms for short words
                end_time = start_time + word_duration

            segments.append(WordSegment(
                text=word,
                start=start_time / 1000,  # Convert ms to seconds
                end=end_time / 1000,
                speaker=None
            ))

            char_idx = word_end_char

        return segments
    
    def _fallback_word_timing(self, words: List[str], total_time_steps: int) -> List[WordSegment]:
        """Fallback method for word timing when no character timestamps are available."""
        segments = []
        
        # Distribute time evenly across words
        total_duration = (total_time_steps / 16000) * 1000  # Convert to milliseconds
        
        for i, word in enumerate(words):
            if i == 0:
                start_time = 0
            else:
                start_time = (i / len(words)) * total_duration
            
            if i == len(words) - 1:
                end_time = total_duration
            else:
                end_time = ((i + 1) / len(words)) * total_duration
            
            segments.append(WordSegment(
                text=word,
                start=start_time / 1000,  # Convert ms to seconds
                end=end_time / 1000,
                speaker=None
            ))
        
        return segments
    
    def _interpolate_word_timing(self, word_start: int, word_end: int, char_to_time: dict, total_chars: int) -> tuple:
        """Interpolate word timing when no direct timestamps are available."""
        # Find the closest timestamps before and after the word
        before_timestamp = None
        after_timestamp = None
        
        # Look for timestamp before the word
        for char_pos in range(word_start - 1, -1, -1):
            if char_pos in char_to_time:
                before_timestamp = char_to_time[char_pos]
                break
        
        # Look for timestamp after the word
        for char_pos in range(word_end, len(char_to_time)):
            if char_pos in char_to_time:
                after_timestamp = char_to_time[char_pos]
                break
        
        if before_timestamp is not None and after_timestamp is not None:
            # Interpolate between the two timestamps
            word_duration = (after_timestamp - before_timestamp) * ((word_end - word_start) / (total_chars))
            return before_timestamp, before_timestamp + word_duration
        elif before_timestamp is not None:
            # Extrapolate forward from the before timestamp
            word_duration = max(200, (word_end - word_start) * 75)  # Minimum 200ms
            return before_timestamp, before_timestamp + word_duration
        elif after_timestamp is not None:
            # Extrapolate backward from the after timestamp
            word_duration = max(200, (word_end - word_start) * 75)
            return after_timestamp - word_duration, after_timestamp
        else:
            # No surrounding timestamps found
            return 0, max(200, (word_end - word_start) * 75)

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