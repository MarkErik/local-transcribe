#!/usr/bin/env python3
"""
Transcriber plugin using OpenAI Whisper.
"""

from typing import List, Optional
import os
import pathlib
from local_transcribe.framework.plugins import TranscriberProvider, WordSegment, registry


class OpenAIWhisperTranscriberProvider(TranscriberProvider):
    """Transcriber provider using OpenAI Whisper for speech-to-text transcription."""

    def __init__(self):
        # Model mapping: user-friendly name -> OpenAI Whisper model name
        self.model_mapping = {
            "tiny.en": "tiny.en",
            "base.en": "base.en",
            "small.en": "small.en"
        }
        self.selected_model = None  # Will be set during transcription
        self.whisper_model = None

    @property
    def name(self) -> str:
        return "openai_whisper"

    @property
    def description(self) -> str:
        return "OpenAI Whisper transcription (tiny.en/base.en/small.en) for speech-to-text"

    @property
    def has_builtin_alignment(self) -> bool:
        return False

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model and selected_model in self.model_mapping:
            return [self.model_mapping[selected_model]]
        # Default to base.en model
        return [self.model_mapping["base.en"]]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Whisper models to cache."""
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

        try:
            cache_dir_whisper = models_dir / "transcribers" / "openai_whisper"
            cache_dir_whisper.mkdir(parents=True, exist_ok=True)

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
        """Check which Whisper models are available offline without downloading."""
        missing_models = []
        whisper_cache_dir = models_dir / "transcribers" / "openai_whisper"

        for model in models:
            if model in self.model_mapping.values():  # It's a Whisper model
                # Check if Whisper model exists in cache
                # Whisper stores models as .pt files with the model name
                model_file = whisper_cache_dir / f"{model}.pt"
                if not model_file.exists():
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
                whisper_cache_dir = models_root / "transcribers" / "openai_whisper"
                whisper_cache_dir.mkdir(parents=True, exist_ok=True)

                self.whisper_model = whisper.load_model(model_name, download_root=str(whisper_cache_dir))
            except ImportError:
                raise ImportError("openai-whisper package is required. Install with: pip install openai-whisper")

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """Transcribe audio using Whisper model."""
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Set selected model from kwargs
        self.selected_model = kwargs.get('transcriber_model', 'base.en')
        self._load_whisper_model()

        # Transcribe with Whisper
        result = self.whisper_model.transcribe(audio_path, language="en", fp16=False)
        return result["text"].strip()

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
    registry.register_transcriber_provider(OpenAIWhisperTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()