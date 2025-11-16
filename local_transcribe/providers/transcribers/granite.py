#!/usr/bin/env python3
"""
Transcriber plugin using IBM Granite.
"""

from typing import List, Optional
import os
import pathlib
import re
import torch
import librosa
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download, snapshot_download
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.config import get_system_capability, clear_device_cache


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
        import os
        import sys

        # Define cache directory first
        cache_dir = models_dir / "transcribers" / "granite"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # DEBUG: Log environment state before download attempt
        print(f"DEBUG: HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"DEBUG: HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"DEBUG: HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
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

        from huggingface_hub import snapshot_download

        try:
            for model in models:
                if model in self.model_mapping.values():  # Check if it's a valid Granite model
                    # Set HF_HOME to cache directory for download
                    os.environ["HF_HOME"] = str(cache_dir)
                    # Use snapshot_download without cache_dir parameter (uses HF_HOME)
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, token=token)
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
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline without downloading."""
        missing_models = []
        for model in models:
            if model in self.model_mapping.values():  # Check if it's a valid Granite model
                # Use XDG_CACHE_HOME as the base (which is set to models/.xdg)
                xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
                if xdg_cache_home:
                    models_root = pathlib.Path(xdg_cache_home)
                else:
                    # Fallback to standard HuggingFace cache location
                    models_root = pathlib.Path.home() / ".cache" / "huggingface"
                
                # Models are stored in standard HuggingFace hub structure
                hub_dir = models_root / "huggingface" / "hub"
                
                # Convert model name to HuggingFace cache directory format
                hf_model_name = model.replace("/", "--")
                model_dir = hub_dir / f"models--{hf_model_name}"
                
                # Check for model files (this is a simplified check)
                if not model_dir.exists() or not any(model_dir.rglob("*.bin")) and not any(model_dir.rglob("*.safetensors")):
                    missing_models.append(model)
        return missing_models

    def _load_model(self):
        """Load the Granite model if not already loaded."""
        if self.model is None:
            model_name = self.model_mapping.get(self.selected_model, self.model_mapping["granite-8b"])
            
            # Use XDG_CACHE_HOME as the base (which is set to models/.xdg), falling back to standard HuggingFace cache
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                models_root = pathlib.Path(xdg_cache_home)
            else:
                # Fallback to standard HuggingFace cache location
                models_root = pathlib.Path.home() / ".cache" / "huggingface"
            
            # The models are stored in the standard HuggingFace hub structure under xdg_cache_home/huggingface/hub
            # We don't need to set HF_HOME, just let transformers find the models in the standard location
            cache_dir = models_root / "huggingface" / "hub"

            try:
                token = os.getenv("HF_TOKEN")
                
                # Models should be cached by preload_models, so use local_files_only=True
                # Let transformers find models in the standard HuggingFace cache location
                self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, token=token)
                self.tokenizer = self.processor.tokenizer

                # Load model - PEFT adapter loads automatically from model config
                # The model has embedded PEFT config, no need to manually call PeftModel.from_pretrained
                print(f"[i] Loading Granite model on device: {self.device}")
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name, 
                    local_files_only=True, 
                    token=token
                ).to(self.device)
                print(f"[✓] Granite model loaded successfully")
            except Exception as e:
                print(f"DEBUG: Failed to load model {model_name}")
                print(f"DEBUG: Cache directory exists: {cache_dir.exists()}")
                if cache_dir.exists():
                    print(f"DEBUG: Cache directory contents: {list(cache_dir.iterdir())}")
                raise e

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> str:
        """Transcribe audio using Granite model."""
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')  # Default to 8b
        if transcriber_model not in self.model_mapping:
            print(f"Warning: Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        self._load_model()

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Calculate audio duration in seconds
        duration = len(wav) / sr
        
        # Check if user wants to disable chunking for quality
        disable_chunking = kwargs.get('disable_chunking', False)
        
        # For audio longer than 1.5 minutes, process in chunks to avoid memory issues
        max_chunk_duration = 90.0  # seconds (1.5 minutes)
        
        if duration > max_chunk_duration and not disable_chunking:
            if kwargs.get('verbose', False):
                print(f"[i] Audio duration: {duration:.1f}s - processing in chunks to manage memory")
                print(f"[i] Note: Chunking may slightly reduce transcription quality at segment boundaries")
                print(f"[i] Use disable_chunking=True for maximum quality (higher memory usage)")
            return self._transcribe_chunked(wav, sr, **kwargs)
        else:
            if duration > max_chunk_duration and disable_chunking and kwargs.get('verbose', False):
                print(f"[i] Audio duration: {duration:.1f}s - processing without chunking (higher memory usage)")
            return self._transcribe_single(wav, **kwargs)

    def _transcribe_single(self, wav, **kwargs) -> str:
        """Transcribe a single audio segment."""
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
            cleaned_text = self._clean_transcription_output(output_text[0].strip(), verbose=kwargs.get('verbose', False))

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

    def _fuzzy_word_match(self, word1: str, word2: str, threshold: float = 0.8) -> bool:
        """Check if two words are similar enough using fuzzy matching."""
        # Handle exact matches first
        if word1 == word2:
            return True
        
        # Handle filler words - ignore them in matching BUT preserve them in final transcript
        filler_words = {'like', 'um', 'uh', 'ah', 'er', 'you know', 'i mean'}
        if word1.lower() in filler_words or word2.lower() in filler_words:
            return True
        
        # Use sequence similarity for partial word matches
        similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        return similarity >= threshold

    def _find_fuzzy_overlap(self, prev_words: List[str], curr_words: List[str], verbose: bool = False) -> int:
        """Find overlapping words using fuzzy matching."""
        max_possible = min(len(prev_words), len(curr_words))
        best_overlap = 0
        
        # Try exact matching first
        for i in range(1, max_possible + 1):
            if prev_words[-i:] == curr_words[:i]:
                best_overlap = i
        
        if best_overlap > 0:
            return best_overlap
        
        # If no exact match, try fuzzy matching
        for overlap_length in range(max_possible, 0, -1):
            match_count = 0
            required_matches = max(1, int(overlap_length * 0.7))  # Require 70% of words to match
            
            for j in range(overlap_length):
                if self._fuzzy_word_match(prev_words[-(overlap_length-j)], curr_words[j]):
                    match_count += 1
            
            if match_count >= required_matches:
                if verbose:
                    print(f"[i] Fuzzy match found: {match_count}/{overlap_length} words matched")
                return overlap_length
        
        return 0

    def _transcribe_chunked(self, wav, sr, **kwargs) -> str:
        """Transcribe audio in chunks to manage memory for long files."""
        chunk_duration = 30.0  # seconds (0.5 minutes)
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(3.0 * sr)  # 3 second overlap to avoid cutting words
        
        chunks = []
        total_samples = len(wav)
        start = 0
        chunk_num = 0
        prev_chunk_text = ""
        
        verbose = kwargs.get('verbose', False)
        
        while start < total_samples:
            chunk_num += 1
            end = min(start + chunk_samples, total_samples)
            chunk_wav = wav[start:end]
            
            if verbose:
                chunk_duration_sec = len(chunk_wav) / sr
                print(f"[i] Processing chunk {chunk_num} ({chunk_duration_sec:.1f}s)...")
            
            # Transcribe chunk
            chunk_text = self._transcribe_single(chunk_wav, **kwargs)
            original_chunk_text = chunk_text  # Save original before trimming
            
            # Remove duplicate text from overlap region
            if chunk_num > 1 and prev_chunk_text:
                # Get last ~15 words from previous chunk (3 seconds at ~2-3 words/sec = 6-9 words + buffer)
                prev_words = prev_chunk_text.split()[-15:]
                curr_words = chunk_text.split()
                
                if verbose:
                    print(f"[i] Chunk {chunk_num-1} end (last 15 words): {' '.join(prev_words)}")
                    print(f"[i] Chunk {chunk_num} start (first 15 words): {' '.join(curr_words[:15])}")
                
                # Try exact matching first, then fuzzy matching
                overlap_length = self._find_fuzzy_overlap(prev_words, curr_words, verbose)
                
                # Remove the overlapping portion from current chunk
                if overlap_length > 0:
                    removed_text = " ".join(curr_words[:overlap_length])
                    chunk_text = " ".join(curr_words[overlap_length:])
                    if verbose:
                        print(f"[i] Found {overlap_length} overlapping words: '{removed_text}'")
                        print(f"[i] Removed {overlap_length} overlapping words at chunk boundary")
                else:
                    if verbose:
                        print(f"[i] No overlapping words found between chunks")
            
            chunks.append(chunk_text)
            # Store the ORIGINAL untrimmed chunk text for next comparison
            prev_chunk_text = original_chunk_text
            
            # Move to next chunk: advance by chunk size minus overlap
            # This ensures we have 2 seconds of overlap, not 28 seconds
            start = start + chunk_samples - overlap_samples
            if start >= total_samples:
                break
        
        # Combine chunks with spaces
        full_transcript = " ".join(chunks)
        
        if verbose:
            print(f"[✓] Combined {chunk_num} chunks into final transcript")
        
        return full_transcript
            
    def _clean_transcription_output(self, text: str, verbose: bool = False) -> str:
        """
        Clean the transcription output by removing dialogue markers and quotation marks.
        
        Args:
            text: Raw transcription output from the model
            verbose: If True, print how many labels were removed
            
        Returns:
            Cleaned transcription text
        """
        # Patterns for labels to remove
        label_patterns = [
            r'\bUser:\s*',
            r'\bAI Assistant:\s*',
            r'\bAssistant:\s*'
        ]
        
        # Count labels before removal if verbose
        if verbose:
            total_removed = sum(len(re.findall(pat, text, flags=re.IGNORECASE)) for pat in label_patterns)
        
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
        
        # Print count if verbose
        if verbose and total_removed > 0:
            print(f"Removed {total_removed} labels from transcript.")
        
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
        self.preload_models(models, models_dir)


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()