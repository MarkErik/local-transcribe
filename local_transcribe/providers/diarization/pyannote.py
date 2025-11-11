#!/usr/bin/env python3
"""
PyAnnote diarization plugin implementation.
"""

from typing import List
import os
import pathlib
import warnings
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from local_transcribe.framework.plugins import DiarizationProvider, WordSegment, Turn, registry


class PyAnnoteDiarizationProvider(DiarizationProvider):
    """Diarization provider using pyannote.audio models."""

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def description(self) -> str:
        return "Speaker diarization using pyannote.audio models"

    def get_required_models(self) -> List[str]:
        return ["pyannote/speaker-diarization"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload pyannote models to cache."""
        import os
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            for model in models:
                if model == "pyannote/speaker-diarization":
                    # Preload by creating the pipeline briefly
                    from pyannote.audio import Pipeline
                    cache_dir = models_dir / "diarization"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    # This will download and cache the model
                    Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", cache_dir=str(cache_dir))
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)

    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Perform speaker diarization using pyannote.

        Args:
            audio_path: Path to audio file
            words: Word segments from ASR
            **kwargs: Provider-specific options
        """
        turns_dicts = self._diarize_mixed(audio_path, words)

        # Convert to Turn
        turns = [
            Turn(
                speaker=t['speaker'],
                start=t['start'],
                end=t['end'],
                text=t['text']
            )
            for t in turns_dicts
        ]
        return turns

    def _load_waveform_mono_32f(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """
        Load audio as float32 and return (waveform [1, T], sample_rate).
        """
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)

        if data.size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        # Ensure mono
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1).astype("float32", copy=False)

        # Convert to torch and add channel dim -> [1, T]
        waveform = torch.from_numpy(data).unsqueeze(0)
        return waveform, sr

    def _diarize_mixed(self, audio_path: str, words: List[WordSegment]) -> List[dict]:
        """
        Diarize a mixed/combined track and assign speakers to words by majority overlap,
        then build readable turns per speaker.
        """
        # Suppress warnings
        warnings.filterwarnings("ignore")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        if not words:
            raise ValueError("No words provided for diarization")

        # Ensure pyannote/huggingface hub will read token from env
        token = os.getenv("HF_TOKEN", "")
        if token:
            os.environ.setdefault("HF_TOKEN", token)

        cache_dir = os.getenv("PYANNOTE_CACHE", "./models/diarization")

        # Load pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            cache_dir=cache_dir,
        )
        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Load waveform
        waveform, sample_rate = self._load_waveform_mono_32f(audio_path)

        # Run diarization
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # Assign speakers to words by majority overlap
        word_speakers = []
        for word in words:
            # Find overlapping diarization segments
            overlaps = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(word.start, segment.start)
                overlap_end = min(word.end, segment.end)
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append((overlap_duration, speaker))

            # Assign speaker with majority overlap
            if overlaps:
                overlaps.sort(reverse=True)
                speaker = overlaps[0][1]
            else:
                speaker = "Unknown"

            word_speakers.append({
                'text': word.text,
                'start': word.start,
                'end': word.end,
                'speaker': speaker
            })

        # Build turns per speaker
        turns = self._build_turns_from_words(word_speakers)

        return turns

    def _build_turns_from_words(self, words: List[dict], max_gap_s: float = 0.8, max_chars: int = 120) -> List[dict]:
        """
        Group words into turns by speaker.
        """
        turns = []
        current_turn = None

        for word in words:
            if current_turn is None or current_turn['speaker'] != word['speaker'] or \
               word['start'] - current_turn['end'] > max_gap_s or \
               len(current_turn['text']) + len(word['text']) + 1 > max_chars:
                if current_turn:
                    turns.append(current_turn)
                current_turn = {
                    'speaker': word['speaker'],
                    'start': word['start'],
                    'end': word['end'],
                    'text': word['text']
                }
            else:
                current_turn['end'] = word['end']
                current_turn['text'] += ' ' + word['text']

        if current_turn:
            turns.append(current_turn)

        return turns


def register_diarization_plugins():
    """Register diarization plugins."""
    registry.register_diarization_provider(PyAnnoteDiarizationProvider())


# Auto-register on import
register_diarization_plugins()