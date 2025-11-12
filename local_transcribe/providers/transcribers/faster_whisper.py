#!/usr/bin/env python3
"""
Transcriber plugin using Faster-Whisper with built-in alignment.
"""

from typing import List, Optional
import os
import pathlib
from faster_whisper import WhisperModel as FWModel
from local_transcribe.framework.plugins import TranscriberProvider, WordSegment, registry


# CT2 (faster-whisper) repos to search locally under ./models/transcribers/ct2/...
_CT2_REPO_CHOICES: dict[str, list[str]] = {
    "medium.en": [
        "Systran/faster-whisper-medium.en",
    ],
    "large-v3": [
        "Systran/faster-whisper-large-v3",
    ],
}


def _asr_device() -> str:
    """
    Device for CTranslate2 (faster-whisper). CTranslate2 does NOT support 'mps',
    so we force CPU on Apple Silicon.
    """
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

    @property
    def name(self) -> str:
        return "faster_whisper"

    @property
    def description(self) -> str:
        return "Faster-Whisper ASR with built-in word-level timestamps"

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
            raise ValueError(f"Unknown ASR model: {selected_model}")

    def get_available_models(self) -> List[str]:
        return list(_CT2_REPO_CHOICES.keys())

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure CT2 models are available locally."""
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

        # Now import snapshot_download after environment change
        from huggingface_hub import snapshot_download

        try:
            ct2_cache = models_dir / "transcribers" / "ct2"

            # DEBUG: Check cache directory structure
            print(f"DEBUG: Cache directory: {ct2_cache}")
            print(f"DEBUG: Cache directory exists: {ct2_cache.exists()}")
            if ct2_cache.exists():
                print(f"DEBUG: Cache directory contents: {list(ct2_cache.iterdir())}")

            for model in models:
                # Use cache_dir to create standard HF cache structure
                print(f"DEBUG: Attempting to download {model} to cache_dir: {ct2_cache}")

                # Additional debug: Check if huggingface_hub sees the offline mode correctly
                from huggingface_hub import HfFolder
                print(f"DEBUG: HfFolder.get_token(): {'***' if HfFolder.get_token() else 'NOT SET'}")

                snapshot_download(model, cache_dir=str(ct2_cache))
                print(f"[✓] {model} downloaded successfully.")

                # DEBUG: Check what was actually created
                if ct2_cache.exists():
                    print(f"DEBUG: After download, cache directory contents: {list(ct2_cache.iterdir())}")
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
        """Check which CT2 models are available offline without downloading."""
        missing_models = []
        ct2_cache = models_dir / "transcribers" / "ct2"
        for model in models:
            safe = f"models--{model.replace('/', '--')}"
            base = ct2_cache / safe / "snapshots"
            if not (base.exists() and any(p.is_dir() for p in base.iterdir() if p.is_dir())):
                missing_models.append(model)
        return missing_models

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """Transcribe audio and return text only."""
        asr_model = kwargs.get('asr_model', 'large-v3')

        if asr_model not in _CT2_REPO_CHOICES:
            raise ValueError(f"Unknown ASR model: {asr_model}")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Resolve local CT2 model snapshot directory
        models_root = pathlib.Path(os.getenv("HF_HOME", "./models")).resolve()
        ct2_cache = models_root / "transcribers" / "ct2"

        local_model_dir = _latest_snapshot_dir_any(ct2_cache, _CT2_REPO_CHOICES[asr_model])

        # Load CT2 model
        fw = FWModel(
            str(local_model_dir),
            device=_asr_device(),
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
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio using faster-whisper with word timestamps.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            **kwargs: Should include 'asr_model' key
        """
        asr_model = kwargs.get('asr_model', 'large-v3')

        if asr_model not in _CT2_REPO_CHOICES:
            raise ValueError(f"Unknown ASR model: {asr_model}")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Resolve local CT2 model snapshot directory
        models_root = pathlib.Path(os.getenv("HF_HOME", "./models")).resolve()
        ct2_cache = models_root / "transcribers" / "ct2"

        local_model_dir = _latest_snapshot_dir_any(ct2_cache, _CT2_REPO_CHOICES[asr_model])

        # Load CT2 model
        fw = FWModel(
            str(local_model_dir),
            device=_asr_device(),
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