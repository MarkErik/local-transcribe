#!/usr/bin/env python3
"""
ASR plugin implementations.
"""

from typing import List, Optional
import os
import pathlib
from faster_whisper import WhisperModel as FWModel
from local_transcribe.framework.plugins import ASRProvider, WordSegment, registry


# CT2 (faster-whisper) repos to search locally under ./models/asr/ct2/...
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
    Given cache_root=./models/asr/ct2 and a list of repo_ids, return the newest
    snapshot directory that exists locally:
      ./models/asr/ct2/models--ORG--REPO/snapshots/<rev>/
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
        f"No CT2 snapshot found under {cache_root} for any of: {repo_ids}. "
        "Models will be downloaded automatically on first run."
    )


class WhisperASRProvider(ASRProvider):
    """ASR provider using Whisper-based transcription with faster-whisper."""

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def description(self) -> str:
        return "Faster-Whisper ASR with word timestamps"

    def get_required_models(self) -> List[str]:
        return ["Systran/faster-whisper-medium.en", "Systran/faster-whisper-large-v3"]

    def get_available_models(self) -> List[str]:
        return list(_CT2_REPO_CHOICES.keys())

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
        ct2_cache = models_root / "asr" / "ct2"

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


def register_asr_plugins():
    """Register ASR plugins."""
    registry.register_asr_provider(WhisperASRProvider())


# Auto-register on import
register_asr_plugins()