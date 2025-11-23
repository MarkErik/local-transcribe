#!/usr/bin/env python3
"""
Transcriber plugin using Faster-Whisper with built-in alignment.
"""

from typing import List, Optional
import os
import pathlib
from faster_whisper import WhisperModel as FWModel
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug


# CT2 (faster-whisper) repos to search locally under ./models/transcribers/ct2/...
_CT2_REPO_CHOICES: dict[str, list[str]] = {
    "medium.en": [
        "Systran/faster-whisper-medium.en",
    ],
    "large-v3": [
        "Systran/faster-whisper-large-v3",
    ],
}


def _ctranslate_device() -> str:
    """
    Device for CTranslate2 (faster-whisper). CTranslate2 supports CUDA and CPU,
    but does NOT support MPS, so we fall back to CPU when MPS is selected.
    """
    from local_transcribe.lib.system_capability_utils import get_system_capability
    import torch
    
    selected_device = get_system_capability()
    
    # CTranslate2 supports CUDA
    if selected_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    
    # CTranslate2 does NOT support MPS, fall back to CPU
    # CPU is also the default for any other case
    return "cpu"


def _latest_snapshot_dir_any(cache_root: pathlib.Path, repo_ids: list[str]) -> pathlib.Path:
    """
    Given cache_root=./models/transcribers/ct2 and a list of repo_ids, return the newest
    model directory that exists locally:
      ./models/transcribers/ct2/models--ORG--REPO/snapshots/<rev>/
    """
    for repo_id in repo_ids:
        safe = f"models--{repo_id.replace('/', '--')}"
        base = cache_root / safe / "snapshots"
        if not base.exists():
            continue
        snaps = [p for p in base.iterdir() if p.is_dir()]
        if snaps:
            snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return snaps[0]
    raise FileNotFoundError(
        f"No model directory found under {cache_root} for any of: {repo_ids}. "
        "Models will be downloaded automatically on first run."
    )


class FasterWhisperTranscriberProvider(TranscriberProvider):
    """Transcriber provider using Faster-Whisper with built-in word-level alignment."""

    def __init__(self):
        self.logger = get_logger()

    @property
    def name(self) -> str:
        return "faster_whisper"

    @property
    def short_name(self) -> str:
        return "Faster-Whisper"

    @property
    def description(self) -> str:
        return "Faster-Whisper transcription with built-in word-level timestamps"

    @property
    def has_builtin_alignment(self) -> bool:
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        if selected_model is None:
            # Return default model
            return ["Systran/faster-whisper-large-v3"]
        elif selected_model in _CT2_REPO_CHOICES:
            return _CT2_REPO_CHOICES[selected_model]
        else:
            raise ValueError(f"Unknown transcriber model: {selected_model}")

    def get_available_models(self) -> List[str]:
        return list(_CT2_REPO_CHOICES.keys())

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure CT2 models are available locally."""
        import os
        import sys

        # DEBUG: Log environment state before download attempt
        log_debug(f"HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        log_debug(f"HF_HOME: {os.environ.get('HF_HOME')}")
        log_debug(f"HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")

        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"

        # DEBUG: Confirm environment variable was set
        log_debug(f"HF_HUB_OFFLINE after setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")

        # Force reload of huggingface_hub modules to pick up new environment
        log_debug("Reloading huggingface_hub modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            log_debug(f"Reloaded {module_name}")

        # Now import snapshot_download after environment change
        from huggingface_hub import snapshot_download

        try:
            ct2_cache = models_dir / "transcribers" / "ct2"

            # DEBUG: Check cache directory structure
            log_debug(f"Cache directory: {ct2_cache}")
            log_debug(f"Cache directory exists: {ct2_cache.exists()}")
            if ct2_cache.exists():
                log_debug(f"Cache directory contents: {list(ct2_cache.iterdir())}")

            for model in models:
                # Use cache_dir to create standard HF cache structure
                log_debug(f"Attempting to download {model} to cache_dir: {ct2_cache}")

                # Additional debug: Check if huggingface_hub sees the offline mode correctly
                from huggingface_hub import HfFolder
                log_debug(f"HfFolder.get_token(): {'***' if HfFolder.get_token() else 'NOT SET'}")

                snapshot_download(model, cache_dir=str(ct2_cache))
                log_completion(f"{model} downloaded successfully")

                # DEBUG: Check what was actually created
                if ct2_cache.exists():
                    log_debug(f"After download, cache directory contents: {list(ct2_cache.iterdir())}")
        except Exception as e:
            log_debug(f"Download failed with error: {e}")
            log_debug(f"Error type: {type(e)}")

            # Additional debug: Check environment at time of error
            log_debug(f"At error time - HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
            log_debug(f"At error time - HF_HOME: {os.environ.get('HF_HOME')}")

            raise Exception(f"Failed to download {model}: {e}")
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which CT2 models are available offline without downloading."""
        missing_models = []
        ct2_cache = models_dir / "transcribers" / "ct2"
        for model in models:
            safe = f"models--{model.replace('/', '--')}"
            base = ct2_cache / safe / "snapshots"
            if not (base.exists() and any(p.is_dir() for p in base.iterdir() if p.is_dir())):
                missing_models.append(model)
        return missing_models

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs) -> str:
        """Transcribe audio and return text only."""
        transcriber_model = kwargs.get('transcriber_model', 'large-v3')

        if transcriber_model not in _CT2_REPO_CHOICES:
            raise ValueError(f"Unknown transcriber model: {transcriber_model}")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Resolve local CT2 model snapshot directory
        models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / "models"))).resolve()
        ct2_cache = models_root / "transcribers" / "ct2"

        local_model_dir = _latest_snapshot_dir_any(ct2_cache, _CT2_REPO_CHOICES[transcriber_model])

        # Load CT2 model
        fw = FWModel(
            str(local_model_dir),
            device=_ctranslate_device(),
            compute_type="int8"
        )

        # Transcribe without word timestamps for text-only
        segments, info = fw.transcribe(
            audio_path,
            language="en",
            vad_filter=False,
            word_timestamps=False,  # No word timestamps for text-only
        )

        # Combine all segment text
        full_text = " ".join(segment.text for segment in segments)
        return full_text.strip()

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio using faster-whisper with word timestamps.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            device: Device to use (cuda/mps/cpu). If None, uses global config.
            **kwargs: Should include 'transcriber_model' key
        """
        transcriber_model = kwargs.get('transcriber_model', 'large-v3')

        if transcriber_model not in _CT2_REPO_CHOICES:
            raise ValueError(f"Unknown transcriber model: {transcriber_model}")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Resolve local CT2 model snapshot directory
        models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / "models"))).resolve()
        ct2_cache = models_root / "transcribers" / "ct2"

        local_model_dir = _latest_snapshot_dir_any(ct2_cache, _CT2_REPO_CHOICES[transcriber_model])

        # Load CT2 model
        fw = FWModel(
            str(local_model_dir),
            device=_ctranslate_device(),
            compute_type="int8"
        )

        # Transcribe with word timestamps
        segments, info = fw.transcribe(
            audio_path,
            language="en",
            vad_filter=False,
            word_timestamps=True,
            beam_size=5,
        )

        # Process segments into WordSegments
        words = []
        for segment in segments:
            for word in segment.words:
                words.append(
                    WordSegment(
                        text=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        speaker=role
                    )
                )

        return words

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload models (alias for ensure_models_available)."""
        self.ensure_models_available(models, models_dir)


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(FasterWhisperTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()