#!/usr/bin/env python3
"""
Aligner plugin using Wav2Vec2 forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import torch
import torchaudio
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

    def _get_token_timestamps(self, emissions: torch.Tensor, transcript: str) -> List[tuple]:
        """Extract token timestamps using proper CTC forced alignment."""
        # Use torchaudio's forced alignment which properly handles CTC
        # emissions shape: [batch=1, time, vocab]
        
        # Get the tokenizer's vocabulary
        dictionary = {c: i for i, c in enumerate(self.wav2vec2_processor.tokenizer.get_vocab().keys())}
        
        # Prepare the transcript tokens
        transcript_normalized = transcript.upper()  # Wav2Vec2 uses uppercase
        tokens = []
        for char in transcript_normalized:
            if char == ' ':
                tokens.append('|')  # Wav2Vec2 uses | for space
            elif char in dictionary:
                tokens.append(char)
        
        if not tokens:
            return []
        
        # Convert to token IDs
        token_ids = [dictionary.get(t, dictionary.get('[UNK]', 0)) for t in tokens]
        
        try:
            # Use torchaudio's forced_align for proper CTC alignment
            # This gives us the time positions where each token appears
            token_spans = torchaudio.functional.forced_align(
                emissions.log_softmax(dim=-1),  # Shape: [batch, time, vocab]
                torch.tensor([token_ids]),  # Shape: [batch, target_length]
                blank=self.wav2vec2_processor.tokenizer.pad_token_id
            )
            
            # Extract timestamps from alignment
            # token_spans is a list of (token_idx, start_frame, end_frame) tuples
            token_timestamps = []
            
            for i, (token_id, start_frame, end_frame) in enumerate(token_spans[0]):
                if i < len(tokens):
                    char = tokens[i]
                    # Convert frames to time (emissions are downsampled by ~320x from 16kHz)
                    # Wav2Vec2 base has stride of 20ms per frame
                    start_time = start_frame * 0.02  # 20ms per frame
                    end_time = end_frame * 0.02
                    token_timestamps.append((char, start_time * 1000, end_time * 1000))  # Convert to ms
            
            return token_timestamps
            
        except Exception as e:
            # Fallback to simple frame-based alignment if forced_align fails
            print(f"Warning: Forced alignment failed ({e}), using fallback method")
            return self._fallback_token_alignment(emissions, tokens)

    def _fallback_token_alignment(self, emissions: torch.Tensor, tokens: List[str]) -> List[tuple]:
        """Fallback token alignment using simple peak detection."""
        token_timestamps = []
        emissions_np = emissions[0].cpu().numpy()  # [time, vocab]
        
        # Get the vocabulary
        dictionary = {c: i for i, c in enumerate(self.wav2vec2_processor.tokenizer.get_vocab().keys())}
        
        # For each token, find where it's most likely in the emission sequence
        time_per_token = emissions_np.shape[0] / max(len(tokens), 1)
        
        for i, token in enumerate(tokens):
            token_id = dictionary.get(token, dictionary.get('[UNK]', 0))
            
            # Search in a window around expected position
            expected_time = int(i * time_per_token)
            window_start = max(0, expected_time - int(time_per_token))
            window_end = min(emissions_np.shape[0], expected_time + int(time_per_token * 2))
            
            # Find peak probability for this token in the window
            window_probs = emissions_np[window_start:window_end, token_id]
            if len(window_probs) > 0:
                peak_pos = window_start + np.argmax(window_probs)
                start_time = peak_pos * 0.02 * 1000  # 20ms per frame, convert to ms
                end_time = (peak_pos + 1) * 0.02 * 1000
                token_timestamps.append((token, start_time, end_time))
        
        return token_timestamps

    def _chars_to_words(self, transcript: str, token_timestamps: List[tuple], speaker: Optional[str] = None) -> List[WordSegment]:
        """Convert token timestamps to word timestamps."""
        words = transcript.split()
        if not words:
            return []
        
        if not token_timestamps:
            return self._fallback_word_timing(words, 0, speaker)
        
        segments = []
        token_idx = 0
        
        # Process each word
        for word in words:
            word_upper = word.upper()
            word_tokens = []
            
            # Collect tokens for this word
            for char in word_upper:
                if token_idx < len(token_timestamps):
                    token, start_ms, end_ms = token_timestamps[token_idx]
                    # Handle space token (|)
                    if token == '|':
                        token_idx += 1
                        if token_idx < len(token_timestamps):
                            token, start_ms, end_ms = token_timestamps[token_idx]
                    
                    # Match character (case-insensitive)
                    if token.upper() == char.upper() or (token == '|' and char == ' '):
                        word_tokens.append((start_ms, end_ms))
                        token_idx += 1
                    else:
                        # Try to skip ahead to find matching token
                        found = False
                        for skip in range(1, min(5, len(token_timestamps) - token_idx)):
                            next_token = token_timestamps[token_idx + skip][0]
                            if next_token.upper() == char.upper():
                                token_idx += skip
                                word_tokens.append((token_timestamps[token_idx][1], token_timestamps[token_idx][2]))
                                token_idx += 1
                                found = True
                                break
                        if not found:
                            token_idx += 1
            
            # Skip space token if present
            if token_idx < len(token_timestamps) and token_timestamps[token_idx][0] == '|':
                token_idx += 1
            
            # Determine word boundaries from tokens
            if word_tokens:
                start_time = min(t[0] for t in word_tokens)
                end_time = max(t[1] for t in word_tokens)
            else:
                # Estimate timing if no tokens matched
                if segments:
                    start_time = segments[-1].end * 1000
                else:
                    start_time = 0
                # Estimate ~150ms per character
                end_time = start_time + len(word) * 150
            
            segments.append(WordSegment(
                text=word,
                start=start_time / 1000,  # Convert to seconds
                end=end_time / 1000,
                speaker=speaker
            ))
        
        return segments
    
    def _fallback_word_timing(self, words: List[str], total_time_steps: int, speaker: Optional[str] = None) -> List[WordSegment]:
        """Fallback method for word timing when no character timestamps are available."""
        segments = []
        
        # Distribute time evenly across words based on word length
        total_chars = sum(len(w) for w in words)
        current_time = 0
        
        for word in words:
            # Allocate time proportional to word length
            word_duration = (len(word) / max(total_chars, 1)) * total_time_steps
            # Minimum 100ms per word
            word_duration = max(word_duration, 100)
            
            segments.append(WordSegment(
                text=word,
                start=current_time / 1000,  # Convert ms to seconds
                end=(current_time + word_duration) / 1000,
                speaker=speaker
            ))
            
            current_time += word_duration
        
        return segments
    
    def align_transcript(
        self,
        audio_path: str,
        transcript: str,
        **kwargs
    ) -> List[WordSegment]:
        """Align transcript text to audio using Wav2Vec2 forced alignment."""
        # Extract speaker from kwargs (passed from split_audio mode)
        speaker = kwargs.get('role') or kwargs.get('speaker')
        
        self._load_wav2vec2_model()

        # Load audio
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Get audio duration for fallback
        audio_duration_ms = len(speech) / sr * 1000

        # Process audio for Wav2Vec2
        inputs = self.wav2vec2_processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.wav2vec2_model(**inputs).logits

        # Get token timestamps using proper CTC forced alignment
        token_timestamps = self._get_token_timestamps(logits, transcript)

        # Convert token timestamps to word timestamps
        if token_timestamps:
            return self._chars_to_words(transcript, token_timestamps, speaker=speaker)
        else:
            # Fallback to simple timing
            words = transcript.split()
            return self._fallback_word_timing(words, audio_duration_ms, speaker=speaker)

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)


def register_aligner_plugins():
    """Register aligner plugins."""
    registry.register_aligner_provider(Wav2Vec2AlignerProvider())


# Auto-register on import
register_aligner_plugins()