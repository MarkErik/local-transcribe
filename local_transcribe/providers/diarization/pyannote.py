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
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug


class ModelNotFoundError(FileNotFoundError):
    """Raised when a required model is not found in the cache."""
    pass


class ModelDownloadError(Exception):
    """Raised when model download fails."""
    pass


class AudioLoadError(Exception):
    """Raised when audio file cannot be loaded."""
    pass


class PyAnnoteDiarizationProvider(DiarizationProvider):
    """Diarization provider using pyannote.audio models."""

    # Default model used by this provider
    DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

    def _get_cache_dir(self, models_dir: pathlib.Path) -> pathlib.Path:
        """Get the cache directory for pyannote models."""
        return models_dir / "diarization" / "pyannote"

    def _model_name_to_hf_format(self, model: str) -> str:
        """Convert model name to HuggingFace cache directory format."""
        return model.replace("/", "--")

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def short_name(self) -> str:
        return "PyAnnote"

    @property
    def description(self) -> str:
        return "Speaker diarization using pyannote.audio models"

    def __init__(self):
        self.logger = get_logger()

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        # Return the actual HuggingFace model ID that needs to be downloaded
        return [self.DEFAULT_MODEL]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload pyannote models to cache."""
        import sys
        
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"

        # Force reload of huggingface_hub modules to pick up new environment
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]

        # Get token from environment once
        token = os.getenv("HF_TOKEN", "")

        try:
            from huggingface_hub import snapshot_download
            
            for model in models:
                if model == self.DEFAULT_MODEL:
                    cache_dir = self._get_cache_dir(models_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    snapshot_download(model, cache_dir=str(cache_dir), token=token if token else None)
                    log_completion(f"{model} downloaded successfully.")
                else:
                    self.logger.warning(f"Unknown model {model}, skipping download")
        except Exception as e:
            raise ModelDownloadError(f"Failed to download {model}: {e}") from e
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which pyannote models are available offline without downloading."""
        missing_models = []
        cache_dir = self._get_cache_dir(models_dir)
        
        for model in models:
            if model == self.DEFAULT_MODEL:
                hf_model_name = self._model_name_to_hf_format(model)
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
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Perform speaker diarization using pyannote and assign speakers to words.

        Args:
            audio_path: Path to audio file
            words: Word segments from transcription
            num_speakers: Number of speakers expected in the audio
            device: Device to use (cuda/mps/cpu). If None, uses global config.
            **kwargs: Provider-specific options. Must include 'models_dir' (pathlib.Path).
        
        Raises:
            ValueError: If models_dir is not provided or invalid
        """
        models_dir = kwargs.get('models_dir')
        if models_dir is None:
            raise ValueError("models_dir is required for PyAnnote diarization")
        if not isinstance(models_dir, pathlib.Path):
            models_dir = pathlib.Path(models_dir)
        
        return self._assign_speakers_to_words(audio_path, words, num_speakers, models_dir)
    

    def _load_waveform_mono_32f(self, audio_path: str) -> tuple[torch.Tensor, int]:
        """
        Load audio as float32 and return (waveform [1, T], sample_rate).
        
        Raises:
            AudioLoadError: If audio file cannot be read or is invalid
        """
        try:
            data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except sf.SoundFileError as e:
            raise AudioLoadError(f"Cannot read audio file '{audio_path}': {e}") from e
        except Exception as e:
            raise AudioLoadError(f"Unexpected error loading audio file '{audio_path}': {e}") from e

        if data.size == 0:
            raise AudioLoadError(f"Audio file is empty: {audio_path}")

        # Ensure mono
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1).astype("float32", copy=False)

        # Convert to torch and add channel dim -> [1, T]
        waveform = torch.from_numpy(data).unsqueeze(0)
        return waveform, sr

    def _assign_speakers_to_words(self, audio_path: str, words: List[WordSegment], num_speakers: int, models_dir: pathlib.Path) -> List[WordSegment]:
        """
        Diarize audio and assign speakers to words by majority overlap.
        
        Raises:
            ValueError: If audio_path doesn't exist or words is empty
            ModelNotFoundError: If the model is not found in cache
            AudioLoadError: If audio cannot be loaded
        """
        import gc
        
        # Suppress warnings
        warnings.filterwarnings("ignore")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        if not words:
            raise ValueError("No words provided for diarization")

        # Get token from environment once
        token = os.getenv("HF_TOKEN", "")

        # Find model snapshot directory
        cache_dir = self._get_cache_dir(models_dir)
        hf_model_name = self._model_name_to_hf_format(self.DEFAULT_MODEL)
        model_dir = cache_dir / f"models--{hf_model_name}"
        
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
            raise ModelNotFoundError(f"PyAnnote model not found at {model_dir}. Please run with --download-models first.")
        
        snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if not snapshot_dirs:
            raise ModelNotFoundError(f"No snapshots found in {snapshots_dir}")
        
        latest_snapshot_dir = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)

        # Initialize variables for cleanup tracking
        pipeline = None
        waveform = None
        diarization = None
        
        try:
            # Load pipeline from the specific snapshot directory
            pipeline = Pipeline.from_pretrained(
                str(latest_snapshot_dir),
                token=token if token else None
            )
            
            # Move to GPU if available and supported
            device = get_system_capability()
            if device != "cpu":
                try:
                    if device == "cuda" and torch.cuda.is_available():
                        pipeline.to(torch.device("cuda"))
                        log_progress("PyAnnote using CUDA device")
                    elif device == "mps" and torch.backends.mps.is_available():
                        pipeline.to(torch.device("mps"))
                        log_progress("PyAnnote using MPS device")
                    else:
                        self.logger.warning(f"Warning: {device} not available, using CPU for PyAnnote")
                except Exception as e:
                    self.logger.warning(f"Warning: Could not move PyAnnote pipeline to {device}, using CPU: {e}")

            # Load waveform
            waveform, sample_rate = self._load_waveform_mono_32f(audio_path)

            # Run diarization with no_grad to prevent gradient accumulation
            with torch.no_grad():
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate, "num_speakers": num_speakers})

            # Assign speakers to words by majority overlap
            word_speakers = []
            for word in words:
                # Find overlapping diarization segments
                overlaps = []
                for segment, speaker in diarization.speaker_diarization:
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
            
        finally:
            # Clean up tensors and memory
            if waveform is not None:
                del waveform
            if pipeline is not None:
                del pipeline
            if diarization is not None:
                del diarization
            
            # Force garbage collection and empty cache
            gc.collect()
            
            # Clear GPU cache
            clear_device_cache()

def register_diarization_plugins():
    """Register diarization plugins."""
    registry.register_diarization_provider(PyAnnoteDiarizationProvider())


# Auto-register on import
register_diarization_plugins()