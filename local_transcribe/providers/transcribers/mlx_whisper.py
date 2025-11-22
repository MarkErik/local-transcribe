#!/usr/bin/env python3
"""
Transcriber plugin using MLX Whisper for Apple Silicon.
"""

from typing import List, Optional
import os
import pathlib
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_output import get_logger, log_completion


class MLXWhisperTranscriberProvider(TranscriberProvider):
    """Transcriber provider using MLX Whisper for speech-to-text transcription on Apple Silicon."""

    def __init__(self):
        # Model mapping: user-friendly name -> MLX Whisper model repo
        self.model_mapping = {
            "tiny": "mlx-community/whisper-tiny",
            "tiny.en": "mlx-community/whisper-tiny.en",
            "base": "mlx-community/whisper-base",
            "base.en": "mlx-community/whisper-base.en",
            "small": "mlx-community/whisper-small",
            "small.en": "mlx-community/whisper-small.en",
            "medium": "mlx-community/whisper-medium",
            "medium.en": "mlx-community/whisper-medium.en",
            "large": "mlx-community/whisper-large",
            "large-v2": "mlx-community/whisper-large-v2",
            "large-v3": "mlx-community/whisper-large-v3",
            "turbo": "mlx-community/whisper-turbo",
        }
        self.logger = get_logger()
        self.selected_model = None  # Will be set during transcription

    @property
    def name(self) -> str:
        return "mlx_whisper"

    @property
    def short_name(self) -> str:
        return "MLX Whisper"

    @property
    def description(self) -> str:
        return "MLX Whisper transcription for Apple Silicon (optimized for M1/M2/M3 chips)"

    @property
    def has_builtin_alignment(self) -> bool:
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model and selected_model in self.model_mapping:
            return [self.model_mapping[selected_model]]
        # Default to base model
        return [self.model_mapping["base"]]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload MLX Whisper models to cache."""
        try:
            import mlx_whisper
            for model in models:
                if model in self.model_mapping.values():  # It's an MLX Whisper model
                    try:
                        # Load and immediately discard to download/cache the model
                        # Use a dummy audio file path that doesn't exist to trigger download without transcribing
                        temp_result = mlx_whisper.transcribe("/dev/null", path_or_hf_repo=model, verbose=False)
                        log_completion(f"MLX Whisper {model} downloaded successfully")
                    except Exception as e:
                        self.logger.warning(f"Failed to preload {model}: {e}")
        except ImportError:
            self.logger.warning("mlx-whisper not available, skipping preload")

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which MLX Whisper models are available offline without downloading."""
        missing = []
        try:
            from huggingface_hub import snapshot_download
            for model in models:
                try:
                    # Try to download with local_files_only=True to check if available
                    snapshot_download(model, local_files_only=True)
                except Exception:
                    missing.append(model)
        except ImportError:
            # If huggingface_hub not available, assume all missing
            missing = models
        return missing

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> str:
        """Transcribe audio using MLX Whisper model."""
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            import mlx_whisper
        except ImportError:
            raise ImportError("mlx-whisper package is required. Install with: pip install mlx-whisper")

        # Set selected model from kwargs
        self.selected_model = kwargs.get('transcriber_model', 'base')
        model_repo = self.model_mapping.get(self.selected_model, self.model_mapping["base"])

        # Transcribe with MLX Whisper
        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_repo, verbose=False)
        return result["text"].strip()

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio with word-level timestamps using MLX Whisper.
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            import mlx_whisper
        except ImportError:
            raise ImportError("mlx-whisper package is required. Install with: pip install mlx-whisper")

        # Set selected model from kwargs
        self.selected_model = kwargs.get('transcriber_model', 'base')
        model_repo = self.model_mapping.get(self.selected_model, self.model_mapping["base"])

        # Transcribe with word timestamps
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            word_timestamps=True,
            verbose=False
        )

        # Convert to WordSegment format
        word_segments = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                word_segments.append(WordSegment(
                    text=word_info["word"].strip(),
                    start=word_info["start"],
                    end=word_info["end"],
                    speaker=role  # Use the provided role if any
                ))

        return word_segments

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by downloading them."""
        try:
            from huggingface_hub import snapshot_download
            for model in models:
                snapshot_download(model)
                log_completion(f"MLX Whisper {model} downloaded successfully")
        except ImportError:
            raise ImportError("huggingface_hub package is required for downloading models.")


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(MLXWhisperTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()