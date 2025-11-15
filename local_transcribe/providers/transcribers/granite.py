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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download, snapshot_download
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry


class GraniteTranscriberProvider(TranscriberProvider):
    """Transcriber provider using IBM Granite for speech-to-text transcription."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Model mapping: user-friendly name -> HuggingFace model name
        self.model_mapping = {
            "granite-2b": "ibm-granite/granite-speech-3.3-2b",
            "granite-8b": "ibm-granite/granite-speech-3.3-8b"
        }
        self.selected_model = None  # Will be set during transcription
        self.processor = None
        self.model = None
        self.tokenizer = None

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

                # Load base model
                base_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, local_files_only=True, token=token).to(self.device)

                # Check if this is a PEFT model
                try:
                    self.model = PeftModel.from_pretrained(base_model, model_name, token=token).to(self.device)
                except:
                    # If not a PEFT model, use the base model
                    self.model = base_model
            except Exception as e:
                print(f"DEBUG: Failed to load model {model_name}")
                print(f"DEBUG: Cache directory exists: {cache_dir.exists()}")
                if cache_dir.exists():
                    print(f"DEBUG: Cache directory contents: {list(cache_dir.iterdir())}")
                raise e

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """Transcribe audio using Granite model."""
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')  # Default to 8b
        if transcriber_model not in self.model_mapping:
            print(f"Warning: Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        self._load_model()

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        wav = torch.from_numpy(wav).unsqueeze(0)

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

        # Post-process the output to remove dialogue markers and quotation marks
        cleaned_text = self._clean_transcription_output(output_text[0].strip(), verbose=kwargs.get('verbose', False))

        return cleaned_text

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