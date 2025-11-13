#!/usr/bin/env python3
"""
Transcriber plugin using IBM Granite.
"""

from typing import List, Optional
import os
import pathlib
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download, snapshot_download
from local_transcribe.framework.plugins import TranscriberProvider, WordSegment, registry


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

        # DEBUG: Log environment state before download attempt
        print(f"DEBUG: HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"DEBUG: HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"DEBUG: HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["HF_HOME"] = str(cache_dir)

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

        # Now import transformers classes after environment change
        # from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        from huggingface_hub import snapshot_download

        try:
            for model in models:
                if model in self.model_mapping.values():  # Check if it's a valid Granite model
                    cache_dir = models_dir / "transcribers" / "granite"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    # Use snapshot_download to download the entire repo
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
                cache_dir = models_dir / "transcribers" / "granite"
                # Check for model files (this is a simplified check)
                hub_dir = cache_dir / "hub"
                if not any(hub_dir.rglob("*.bin")) and not any(hub_dir.rglob("*.safetensors")):
                    missing_models.append(model)
        return missing_models

    def _load_model(self):
        """Load the Granite model if not already loaded."""
        if self.model is None:
            # Get the actual model name from selected model
            model_name = self.model_mapping.get(self.selected_model, self.model_mapping["granite-8b"])

            cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "transcribers" / "granite"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Set HF_HOME to our cache directory so loading finds the files
            original_hf_home = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = str(cache_dir)

            try:
                token = os.getenv("HF_TOKEN")
                # Models should be cached by preload_models, so use local_files_only=True
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
            finally:
                # Restore original HF_HOME
                if original_hf_home is not None:
                    os.environ["HF_HOME"] = original_hf_home
                else:
                    os.environ.pop("HF_HOME", None)

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """Transcribe audio using Granite model."""
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')  # Default to 8b
        if transcriber_model not in self.model_mapping:
            print(f"Warning: Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
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