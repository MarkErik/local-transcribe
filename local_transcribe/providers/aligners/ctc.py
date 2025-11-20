#!/usr/bin/env python3
"""
Aligner plugin using CTC-based forced alignment.
Inspired by CTC forced alignment techniques for efficient word-level timestamps.
"""

from typing import List, Optional
import os
import pathlib
import torch
import torchaudio
import librosa
from transformers import AutoModelForCTC, AutoTokenizer
from uroman import Uroman
from local_transcribe.framework.plugin_interfaces import AlignerProvider, WordSegment, registry
from local_transcribe.lib.config import get_system_capability, clear_device_cache


# Initialize uroman for romanization
uroman_instance = Uroman()


class CTCAlignerProvider(AlignerProvider):
    """Aligner provider using CTC-based forced alignment for accurate word-level timestamps."""

    def __init__(self):
        # Device is determined dynamically
        self.model_name = "facebook/mms-300m"  # Multilingual CTC model
        self.model = None

    @property
    def device(self):
        return get_system_capability()
        self.tokenizer = None
        self.sample_rate = 16000

    @property
    def name(self) -> str:
        return "ctc"

    @property
    def short_name(self) -> str:
        return "CTC"

    @property
    def description(self) -> str:
        return "CTC-based forced alignment for multilingual word-level timestamps"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model:
            return [selected_model]
        return [self.model_name]

    def get_available_models(self) -> List[str]:
        return ["mms-300m", "mms-1b", "wav2vec2-base"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload CTC models to cache."""
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
        print(f"DEBUG: HF_HOME after setting: {os.environ.get('HF_HOME')}")

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

        try:
            for model in models:
                if model == "facebook/mms-300m":
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

                    # Use snapshot_download with cache_dir parameter pointing to the standard location
                    from huggingface_hub import snapshot_download
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
        """Check which CTC models are available offline without downloading."""
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
            if "/" in model:  # It's a HuggingFace model like MMS
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

    def _load_model(self, model_name: str):
        """Load the CTC model and tokenizer."""
        if self.model is None or model_name != getattr(self, '_current_model', None):
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
                self.model = AutoModelForCTC.from_pretrained(model_name, local_files_only=True, token=token).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, token=token)
                self._current_model = model_name
            except Exception as e:
                print(f"DEBUG: Failed to load CTC model {model_name}")
                print(f"DEBUG: Cache directory exists: {(models_root / 'huggingface' / 'hub').exists()}")
                if (models_root / 'huggingface' / 'hub').exists():
                    print(f"DEBUG: Cache directory contents: {list((models_root / 'huggingface' / 'hub').iterdir())}")
                raise e

    def _normalize_text(self, text: str, romanize: bool = False, language: str = "eng") -> str:
        """Basic text normalization with optional romanization."""
        # Simple normalization: lowercase, remove extra spaces
        text = text.lower().strip()
        # Remove punctuation except apostrophes
        import re
        text = re.sub(r'[^\w\s\']', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        if romanize and language != "eng":
            try:
                text = uroman_instance.romanize_string(text, lcode=language)
            except Exception:
                # Fallback to original text if romanization fails
                pass
        
        return text.strip()

    def _preprocess_text(self, text: str, romanize: bool = False, language: str = "eng") -> tuple:
        """Preprocess text into characters/tokens for alignment with optional romanization."""
        normalized = self._normalize_text(text, romanize, language)
        words = normalized.split()
        
        # Create character sequence with word boundaries marked
        # We need both the character sequence for alignment and word boundaries for segmentation
        chars = []
        word_boundaries = []  # List of (word_text, start_char_idx, end_char_idx)
        char_idx = 0
        
        for word in words:
            word_start = char_idx
            for char in word:
                chars.append(char)
                char_idx += 1
            word_end = char_idx
            word_boundaries.append((word, word_start, word_end))
            
            # Add space between words (except after last word)
            if word != words[-1]:
                chars.append(' ')
                char_idx += 1
        
        return chars, word_boundaries

    def _generate_emissions(self, audio_path: str, window_length: int = 30, context_length: int = 2, batch_size: int = 4) -> tuple:
        """Generate CTC emissions from audio with chunked processing."""
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
        waveform = torch.from_numpy(waveform).float().to(self.device)

        if sr != self.sample_rate:
            waveform = torch.from_numpy(librosa.resample(waveform.cpu().numpy(), orig_sr=sr, target_sr=self.sample_rate)).to(self.device)

        # Process in chunks to handle long audio
        window_samples = window_length * self.sample_rate
        context_samples = context_length * self.sample_rate

        if waveform.size(0) < window_samples:
            # Short audio, process directly
            with torch.no_grad():
                emissions = self.model(waveform.unsqueeze(0)).logits
            stride = float(waveform.size(0) * 1000 / emissions.size(1) / self.sample_rate)
            return emissions.squeeze(0), stride

        # Chunked processing
        emissions_list = []
        stride = None

        for start in range(0, waveform.size(0), window_samples):
            end = min(start + window_samples + 2 * context_samples, waveform.size(0))
            chunk_start = max(0, start - context_samples)
            chunk = waveform[chunk_start:end]

            with torch.no_grad():
                chunk_emissions = self.model(chunk.unsqueeze(0)).logits.squeeze(0)

            # Remove context from emissions if added
            if chunk_start > 0:
                # Calculate how many emission frames to remove from start
                context_frames = int(context_samples * chunk_emissions.size(0) / (chunk.size(0)))
                chunk_emissions = chunk_emissions[context_frames:]

            if end < waveform.size(0):
                # Remove context from end for overlapping chunks
                context_frames_end = int(context_samples * chunk_emissions.size(0) / (chunk.size(0)))
                chunk_emissions = chunk_emissions[:-context_frames_end]

            emissions_list.append(chunk_emissions)

            if stride is None:
                stride = float(chunk.size(0) * 1000 / chunk_emissions.size(0) / self.sample_rate)
            
            # Clean up chunk tensors immediately
            del chunk
            del chunk_emissions

        # Concatenate emissions
        emissions = torch.cat(emissions_list, dim=0)
        
        # Clean up emissions list
        del emissions_list
        
        # Clear cache after processing
        import gc
        gc.collect()
        clear_device_cache()
        
        return emissions, stride

    def _forced_align(self, emissions: torch.Tensor, tokens: List[str], blank_id: int = 0) -> List[tuple]:
        """Perform forced alignment using torchaudio's CTC forced alignment."""
        # Convert tokens to vocabulary indices
        vocab = self.tokenizer.get_vocab()
        token_indices = []
        
        for token in tokens:
            if token in vocab:
                token_indices.append(vocab[token])
            else:
                # Try lowercase/uppercase variants
                if token.lower() in vocab:
                    token_indices.append(vocab[token.lower()])
                elif token.upper() in vocab:
                    token_indices.append(vocab[token.upper()])
                else:
                    # Skip unknown tokens or use blank
                    unk_id = vocab.get('<unk>', vocab.get('[UNK]', vocab.get('|', blank_id)))
                    if unk_id != blank_id:
                        token_indices.append(unk_id)

        if not token_indices:
            return []

        try:
            # Use torchaudio's forced_align for proper CTC alignment
            log_probs = torch.log_softmax(emissions, dim=-1)
            
            # Ensure emissions has batch dimension: [batch, time, vocab]
            if log_probs.dim() == 2:
                log_probs = log_probs.unsqueeze(0)
            
            token_spans = torchaudio.functional.forced_align(
                log_probs,
                torch.tensor([token_indices]),
                blank=blank_id
            )
            
            # Convert to list of (token_id, frame) tuples
            alignment = []
            for token_id, start_frame, end_frame in token_spans[0]:
                # Use the middle of the span as the representative frame
                mid_frame = (start_frame + end_frame) // 2
                alignment.append((token_id.item(), mid_frame.item()))
            
            return alignment
            
        except Exception as e:
            print(f"Warning: Forced alignment failed ({e}), using fallback")
            return self._fallback_align(emissions, token_indices, blank_id)

    def _fallback_align(self, emissions: torch.Tensor, token_indices: List[int], blank_id: int) -> List[tuple]:
        """Fallback alignment using simple frame-based distribution."""
        alignment = []
        
        if emissions.dim() == 3:
            emissions = emissions.squeeze(0)
        
        T = emissions.shape[0]
        frames_per_token = T / max(len(token_indices), 1)
        
        for i, token_id in enumerate(token_indices):
            frame = int(i * frames_per_token + frames_per_token / 2)
            frame = min(frame, T - 1)
            alignment.append((token_id, frame))
        
        return alignment

    def _alignment_to_segments(self, alignment: List[tuple], word_boundaries: List[tuple], stride: float, speaker: Optional[str] = None) -> List[WordSegment]:
        """Convert character-level alignment to word segments using word boundaries."""
        if not alignment or not word_boundaries:
            return []
        
        # Create a mapping from character index to frame
        char_to_frame = {}
        for i, (token_id, frame) in enumerate(alignment):
            char_to_frame[i] = frame
        
        segments = []
        
        # For each word, find the frames corresponding to its characters
        for word_text, start_char, end_char in word_boundaries:
            # Collect frames for characters in this word
            word_frames = []
            for char_idx in range(start_char, end_char):
                if char_idx in char_to_frame:
                    word_frames.append(char_to_frame[char_idx])
            
            if word_frames:
                start_frame = min(word_frames)
                end_frame = max(word_frames)
            else:
                # Fallback: estimate based on position
                if segments:
                    # Continue from last segment
                    start_frame = int(segments[-1].end * 1000 / stride)
                else:
                    start_frame = 0
                # Estimate duration based on word length (~100ms per character)
                end_frame = start_frame + len(word_text) * 5
            
            start_time = start_frame * stride / 1000  # Convert to seconds
            end_time = end_frame * stride / 1000
            
            # Ensure end time is after start time
            if end_time <= start_time:
                end_time = start_time + 0.1  # Minimum 100ms
            
            segments.append(WordSegment(
                text=word_text,
                start=start_time,
                end=end_time,
                speaker=speaker
            ))
        
        return segments

    def align_transcript(
        self,
        audio_path: str,
        transcript: str,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """Align transcript to audio using CTC forced alignment.
        
        Args:
            audio_path: Path to audio file
            transcript: Transcript text
            device: Device to use (cuda/mps/cpu). If None, uses global config.
            **kwargs: Additional options including 'role' or 'speaker'
        """
        # Extract speaker from kwargs (passed from split_audio mode)
        speaker = kwargs.get('role') or kwargs.get('speaker')
        
        model_name = kwargs.get('model', self.model_name)
        romanize = kwargs.get('romanize', False)
        language = kwargs.get('language', 'eng')
        
        self._load_model(model_name)

        # Preprocess text into characters and word boundaries
        chars, word_boundaries = self._preprocess_text(transcript, romanize, language)

        # Generate emissions
        emissions, stride = self._generate_emissions(audio_path)

        # Get blank token id
        blank_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Perform forced alignment on characters
        alignment = self._forced_align(emissions, chars, blank_id)

        # Convert character alignment to word segments
        segments = self._alignment_to_segments(alignment, word_boundaries, stride, speaker=speaker)

        return segments

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available."""
        self.preload_models(models, models_dir)


def register_aligner_plugins():
    """Register aligner plugins."""
    registry.register_aligner_provider(CTCAlignerProvider())


# Auto-register on import
register_aligner_plugins()
