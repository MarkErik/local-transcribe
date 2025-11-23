#!/usr/bin/env python3
"""
Aligner plugin using Wav2Vec2 forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import warnings
import torch
import torchaudio
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from local_transcribe.framework.plugin_interfaces import AlignerProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion


class Wav2Vec2AlignerProvider(AlignerProvider):
    """Aligner provider using Wav2Vec2 for forced alignment of transcripts to audio."""

    def __init__(self):
        # Device is determined dynamically
        self.wav2vec2_model_name = "facebook/wav2vec2-base-960h"  # English model

        # Wav2Vec2 components for alignment
        self.wav2vec2_processor = None
        self.wav2vec2_model = None
        self.logger = get_logger()

    @property
    def device(self):
        return get_system_capability()

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
        self.logger.debug(f"HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        self.logger.debug(f"HF_HOME: {os.environ.get('HF_HOME')}")
        self.logger.debug(f"HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"

        # DEBUG: Confirm environment variable was set
        self.logger.debug(f"HF_HUB_OFFLINE after setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")

        # Force reload of huggingface_hub modules to pick up new environment
        self.logger.debug("Reloading huggingface_hub modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            self.logger.debug(f"Reloaded {module_name}")

        # Also reload transformers modules
        self.logger.debug("Reloading transformers modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            self.logger.debug(f"Reloaded {module_name}")

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
                    log_completion(f"{model} downloaded successfully")
                else:
                    self.logger.warning(f"Unknown model {model}, skipping download")
        except Exception as e:
            self.logger.debug(f"Download failed with error: {e}")
            self.logger.debug(f"Error type: {type(e)}")

            # Additional debug: Check environment at time of error
            self.logger.debug(f"At error time - HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
            self.logger.debug(f"At error time - HF_HOME: {os.environ.get('HF_HOME')}")

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
                
                # Suppress the masked_spec_embed warning - it's only used during training, not inference
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*masked_spec_embed.*")
                    self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(self.wav2vec2_model_name, local_files_only=True, token=token)
                    self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(self.wav2vec2_model_name, local_files_only=True, token=token).to(self.device)
            except Exception as e:
                self.logger.debug(f"Failed to load Wav2Vec2 model {self.wav2vec2_model_name}")
                self.logger.debug(f"Cache directory exists: {(models_root / 'huggingface' / 'hub').exists()}")
                if (models_root / 'huggingface' / 'hub').exists():
                    self.logger.debug(f"Cache directory contents: {list((models_root / 'huggingface' / 'hub').iterdir())}")
                raise e

    def _get_token_timestamps(self, emissions: torch.Tensor, transcript: str) -> List[tuple]:
        """Extract token timestamps using CTC alignment with frame-level paths."""
        # emissions shape: [batch=1, time, vocab]
        
        # Get the tokenizer's vocabulary
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        dictionary = {c: i for i, c in enumerate(vocab.keys())}
        id_to_char = {i: c for c, i in vocab.items()}
        
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
            # Use torchaudio's forced_align to get frame-level alignment
            # Note: This function returns (aligned_labels, scores) as tensors, not token spans
            log_probs = emissions.log_softmax(dim=-1).cpu()  # Move to CPU for forced_align compatibility
            
            # Suppress deprecation warning for forced_align
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*forced_align.*deprecated.*")
                aligned_labels, scores = torchaudio.functional.forced_align(
                    log_probs,  # Shape: [batch, time, vocab]
                    torch.tensor([token_ids]),  # Shape: [batch, target_length]
                    blank=self.wav2vec2_processor.tokenizer.pad_token_id
                )
            
            # aligned_labels shape: [batch, time] - contains token_id for each frame
            # Now we need to extract token boundaries from the frame-level predictions
            token_timestamps = self._extract_token_boundaries(
                aligned_labels[0],  # Remove batch dimension: [time]
                tokens,
                token_ids,
                self.wav2vec2_processor.tokenizer.pad_token_id
            )
            
            return token_timestamps
            
        except Exception as e:
            # Fallback to simple frame-based alignment if forced_align fails
            self.logger.warning(f"Forced alignment failed ({e}), using fallback method")
            return self._fallback_token_alignment(emissions, tokens)

    def _extract_token_boundaries(self, aligned_labels: torch.Tensor, tokens: List[str], 
                                   token_ids: List[int], blank_id: int) -> List[tuple]:
        """Extract token boundaries from frame-level aligned labels.
        
        Args:
            aligned_labels: Tensor of shape [time] with token_id for each frame
            tokens: List of characters/tokens in the transcript
            token_ids: List of token IDs corresponding to tokens
            blank_id: The ID of the blank token in CTC
            
        Returns:
            List of (char, start_time_ms, end_time_ms) tuples
        """
        token_timestamps = []
        aligned_labels_list = aligned_labels.tolist()
        
        # Map token_ids to their positions in the transcript
        token_id_to_char = {tid: tokens[i] for i, tid in enumerate(token_ids)}
        
        # Track which transcript token we're currently looking for
        current_token_idx = 0
        frame_start = None
        
        for frame_idx, label_id in enumerate(aligned_labels_list):
            # Skip blank frames
            if label_id == blank_id:
                # If we were tracking a token, finalize it
                if frame_start is not None and current_token_idx < len(token_ids):
                    # Convert frames to time (20ms per frame for Wav2Vec2 base)
                    start_time = frame_start * 0.02 * 1000  # Convert to ms
                    end_time = frame_idx * 0.02 * 1000  # End at current frame
                    token_timestamps.append((tokens[current_token_idx], start_time, end_time))
                    
                    current_token_idx += 1
                    frame_start = None
                continue
            
            # Check if this label matches the current expected token
            if current_token_idx < len(token_ids) and label_id == token_ids[current_token_idx]:
                # Start or continue tracking this token
                if frame_start is None:
                    frame_start = frame_idx
            else:
                # Label doesn't match - could be repeated token or mismatch
                # If we were tracking a token, finalize it
                if frame_start is not None and current_token_idx < len(token_ids):
                    start_time = frame_start * 0.02 * 1000
                    end_time = frame_idx * 0.02 * 1000
                    token_timestamps.append((tokens[current_token_idx], start_time, end_time))
                    
                    current_token_idx += 1
                    frame_start = None
                    
                    # Check if this new label matches the next token
                    if current_token_idx < len(token_ids) and label_id == token_ids[current_token_idx]:
                        frame_start = frame_idx
        
        # Finalize the last token if still tracking
        if frame_start is not None and current_token_idx < len(token_ids):
            start_time = frame_start * 0.02 * 1000
            end_time = len(aligned_labels_list) * 0.02 * 1000
            token_timestamps.append((tokens[current_token_idx], start_time, end_time))
        
        return token_timestamps

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
        """Convert character-level token timestamps to word timestamps."""
        words = transcript.split()
        if not words:
            return []
        
        if not token_timestamps:
            return self._fallback_word_timing(words, 0, speaker)
        
        segments = []
        token_idx = 0
        
        # Process each word
        for word in words:
            # Strip all punctuation for alignment, but keep the original word for output
            # Only match alphanumeric characters and apostrophes/hyphens that are internal to words
            word_chars = [c for c in word.upper() if c.isalnum()]
            word_token_times = []
            
            # Collect all character tokens for this word
            chars_matched = 0
            while token_idx < len(token_timestamps) and chars_matched < len(word_chars):
                token_char, start_ms, end_ms = token_timestamps[token_idx]
                
                # Skip space tokens between words
                if token_char == '|':
                    token_idx += 1
                    continue
                
                # Check if this token matches the expected character
                if chars_matched < len(word_chars) and token_char.upper() == word_chars[chars_matched].upper():
                    word_token_times.append((start_ms, end_ms))
                    chars_matched += 1
                    token_idx += 1
                else:
                    # Token doesn't match - try to skip ahead to find it
                    found = False
                    for skip in range(1, min(10, len(token_timestamps) - token_idx)):
                        if token_timestamps[token_idx + skip][0] == '|':
                            continue
                        if token_timestamps[token_idx + skip][0].upper() == word_chars[chars_matched].upper():
                            # Found the matching character ahead
                            token_idx += skip
                            found = True
                            break
                    
                    if not found:
                        # Give up on this character, move to next
                        chars_matched += 1
                        token_idx += 1
            
            # Determine word boundaries from collected character tokens
            if word_token_times:
                # Word starts at the earliest character start time
                start_time = min(t[0] for t in word_token_times)
                # Word ends at the latest character end time
                end_time = max(t[1] for t in word_token_times)
            else:
                # Fallback: estimate timing if no tokens matched
                if segments:
                    start_time = segments[-1].end * 1000
                else:
                    start_time = 0
                # Estimate ~150ms per character as minimum
                end_time = start_time + max(len(word) * 150, 200)
            
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
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """Align transcript text to audio using Wav2Vec2 forced alignment.
        
        Args:
            audio_path: Path to audio file
            transcript: Transcript text
            device: Device to use (cuda/mps/cpu). If None, uses global config.
            **kwargs: Additional options including 'role' or 'speaker'
        """
        # Extract speaker from kwargs (passed from split_audio mode)
        speaker = kwargs.get('role') or kwargs.get('speaker')
        
        self._load_wav2vec2_model()

        try:
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
                result = self._chars_to_words(transcript, token_timestamps, speaker=speaker)
            else:
                # Fallback to simple timing
                words = transcript.split()
                result = self._fallback_word_timing(words, audio_duration_ms, speaker=speaker)
            
            return result
            
        finally:
            # Clean up tensors and memory
            if 'speech' in locals():
                del speech
            if 'inputs' in locals():
                # Delete individual tensors from inputs dict
                for key in list(inputs.keys()):
                    del inputs[key]
                del inputs
            if 'logits' in locals():
                del logits
            
            import gc
            gc.collect()
            
            clear_device_cache()

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)


def register_aligner_plugins():
    """Register aligner plugins."""
    registry.register_aligner_provider(Wav2Vec2AlignerProvider())


# Auto-register on import
register_aligner_plugins()