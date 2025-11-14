#!/usr/bin/env python3
"""
PyAnnote diarization plugin implementation.
"""

from typing import List, Optional
import os
import pathlib
import warnings
import torch
import soundfile as sf
from pyannote.audio import Pipeline
from local_transcribe.framework.plugin_interfaces import DiarizationProvider, WordSegment, Turn, registry


class PyAnnoteDiarizationProvider(DiarizationProvider):
    """Diarization provider using pyannote.audio models."""

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def short_name(self) -> str:
        return "PyAnnote"

    @property
    def description(self) -> str:
        return "Speaker diarization using pyannote.audio models"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        return ["pyannote/speaker-diarization"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload pyannote models to cache."""
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

        try:
            for model in models:
                if model == "pyannote/speaker-diarization":
                    # Preload by creating the pipeline briefly
                    from pyannote.audio import Pipeline
                    
                    # Define cache directory first (consistent with other providers) - use diarizers not transcribers
                    cache_dir = models_dir / "diarizers" / "pyannote"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Get token from environment
                    token = os.getenv("HF_TOKEN", "")
                    
                    print(f"DEBUG: Attempting to download pyannote model to cache_dir: {cache_dir}")
                    print(f"DEBUG: Using HF_TOKEN: {'***' if token else 'NOT SET'}")
                    
                    # This will download and cache the model
                    Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-community-1",
                        cache_dir=str(cache_dir),
                        token=token if token else None
                    )
                    print(f"[✓] {model} downloaded successfully.")
                    
                    # DEBUG: Check what was actually created
                    if cache_dir.exists():
                        print(f"DEBUG: After download, cache directory contents: {list(cache_dir.iterdir())}")
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

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which pyannote models are available offline without downloading."""
        missing_models = []
        
        # Use the same cache directory structure as download - diarizers not transcribers
        cache_dir = models_dir / "diarizers" / "pyannote"
        
        for model in models:
            if model == "pyannote/speaker-diarization":
                # Convert model name to HuggingFace cache directory format
                hf_model_name = "pyannote--speaker-diarization-community-1".replace("/", "--")
                model_dir = cache_dir / f"models--{hf_model_name}"
                
                # Check for snapshots directory (this is the standard HF structure)
                snapshots_dir = model_dir / "snapshots"
                if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
                    missing_models.append(model)
            else:
                missing_models.append(model)  # Assume unknown models are missing
                
        return missing_models

    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        num_speakers: int,
        **kwargs
    ) -> List[WordSegment]:
        """
        Perform speaker diarization using pyannote and assign speakers to words.

        Args:
            audio_path: Path to audio file
            words: Word segments from transcription
            num_speakers: Number of speakers expected in the audio
            **kwargs: Provider-specific options
        """
        turns_dicts = self._diarize_mixed(audio_path, words, num_speakers)

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

    def _assign_speakers_to_words(self, audio_path: str, words: List[WordSegment], num_speakers: int) -> List[WordSegment]:
        """
        Diarize audio and assign speakers to words by majority overlap.
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

        # Use the same cache directory structure as download - diarizers not transcribers
        cache_dir = models_dir / "diarizers" / "pyannote"
        
        # Convert model name to HuggingFace cache directory format
        hf_model_name = "pyannote--speaker-diarization-community-1".replace("/", "--")
        model_dir = cache_dir / f"models--{hf_model_name}"
        
        # Use the actual model directory for loading
        if not model_dir.exists():
            raise FileNotFoundError(f"PyAnnote model not found at {model_dir}. Please run with --download-models first.")
        
        print(f"DEBUG: Loading pyannote model from: {model_dir}")

        # Get token from environment
        token = os.getenv("HF_TOKEN", "")
        
        print(f"DEBUG: Loading pyannote model with token: {'***' if token else 'NOT SET'}")
        
        # Load pipeline using local files only
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            cache_dir=str(cache_dir),  # Pass the correct cache directory
            local_files_only=True,
            token=token if token else None
        )
        # Move to GPU if available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Load waveform
        waveform, sample_rate = self._load_waveform_mono_32f(audio_path)

        # Run diarization
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate, "num_speakers": num_speakers})

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

            word_speakers.append(WordSegment(text=word.text, start=word.start, end=word.end, speaker=speaker))

        return word_speakers

def register_diarization_plugins():
    """Register diarization plugins."""
    registry.register_diarization_provider(PyAnnoteDiarizationProvider())


# Auto-register on import
register_diarization_plugins()