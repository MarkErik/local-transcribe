#!/usr/bin/env python3
"""
Unified ASR + Diarization plugin using WhisperX.
"""

from typing import List, Optional
import os
import pathlib
import torch
import whisperx
from local_transcribe.framework.plugin_interfaces import UnifiedProvider, Turn, registry
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger


class WhisperXUnifiedProvider(UnifiedProvider):
    """Unified provider using WhisperX for ASR with word-level timestamps and speaker diarization."""

    def __init__(self):
        # Device is determined dynamically
        self.model_mapping = {
            "large-v2": "large-v2",
            "large-v3": "large-v3",
            "medium": "medium",
            "small": "small",
            "base": "base",
        }
        self.selected_model = None  # Will be set during transcription
        self.logger = get_logger()

    @property
    def device(self):
        return get_system_capability()

    @property
    def name(self) -> str:
        return "whisperx"

    @property
    def short_name(self) -> str:
        return "WhisperX"

    @property
    def description(self) -> str:
        return "WhisperX: Fast ASR with word-level timestamps and integrated speaker diarization"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        models = []
        if selected_model and selected_model in self.model_mapping:
            models.append(self.model_mapping[selected_model])  # Whisper model
            models.append("WAV2VEC2_ASR_LARGE_LV60K_960H")  # Default alignment model
            models.append("pyannote/speaker-diarization")  # Diarization model
        else:
            # Default models
            models = ["large-v2", "WAV2VEC2_ASR_LARGE_LV60K_960H", "pyannote/speaker-diarization"]
        return models

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload WhisperX models to cache."""
        for model in models:
            if model in self.model_mapping.values():
                # Preload Whisper model
                whisperx.load_model(model, device=self.device, download_root=str(models_dir / "asr"))
            elif model.startswith("WAV2VEC2"):
                # Preload alignment model
                whisperx.load_align_model("en", device=self.device, download_root=str(models_dir / "asr"))
            elif "speaker-diarization" in model:
                # Preload diarization model (requires HF token)
                from pyannote.audio import Pipeline
                cache_dir = models_dir / "diarization"
                cache_dir.mkdir(parents=True, exist_ok=True)
                Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", cache_dir=str(cache_dir))

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline."""
        # For simplicity, assume all are missing if not preloaded
        # In practice, you'd check for model files
        return models

    def transcribe_and_diarize(
        self,
        audio_path: str,
        num_speakers: int,
        device: Optional[str] = None,
        **kwargs
    ) -> List[Turn]:
        """
        Transcribe audio with WhisperX and perform speaker diarization.

        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers expected in the audio
            device: Device to use (cuda/mps/cpu). If None, uses global config.
            **kwargs: Should include 'model' key for Whisper model

        Returns:
            List of Turn objects with speaker assignments and text
        """
        model_name = kwargs.get('model', 'large-v2')
        if model_name not in self.model_mapping:
            raise ValueError(f"Unknown model: {model_name}")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Load audio
        audio = whisperx.load_audio(audio_path)

        try:
            # 1. Transcribe with Whisper
            # Use appropriate compute type based on device
            compute_type = "float16" if self.device in ["cuda", "mps"] else "int8"
            model = whisperx.load_model(model_name, device=self.device, compute_type=compute_type)
            with torch.no_grad():
                result = model.transcribe(audio, batch_size=16)
            self.logger.debug(f"Transcription: {result['segments'][:2]}...")
            
            # Clean up transcription model
            del model
            clear_device_cache()

            # 2. Align output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            with torch.no_grad():
                result = whisperx.align(result["segments"], model_a, metadata, audio, device=self.device, return_char_alignments=False)
            self.logger.debug(f"Alignment: {result['segments'][:2]}...")
            
            # Clean up alignment model
            del model_a
            del metadata
            clear_device_cache()

            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=self.device)
            with torch.no_grad():
                diarize_segments = diarize_model(audio, num_speakers=num_speakers)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            self.logger.debug(f"Diarization: {result['segments'][:2]}...")
            
            # Clean up diarization model
            del diarize_model
            del diarize_segments
            clear_device_cache()

            # Convert to Turns
            turns = []
            current_speaker = None
            current_text = []
            current_start = None
            current_end = None

            for segment in result["segments"]:
                for word in segment.get("words", []):
                    speaker = word.get("speaker")
                    if speaker != current_speaker:
                        if current_speaker is not None:
                            turns.append(Turn(
                                speaker=current_speaker,
                                start=current_start,
                            end=current_end,
                            text=" ".join(current_text)
                        ))
                    current_speaker = speaker
                    current_text = []
                    current_start = word["start"]
                current_text.append(word["word"])
                current_end = word["end"]

            # Add final turn
            if current_speaker is not None:
                turns.append(Turn(
                    speaker=current_speaker,
                    start=current_start,
                    end=current_end,
                    text=" ".join(current_text)
                ))

            return turns
            
        finally:
            # Final cleanup
            if 'audio' in locals():
                del audio
            
            import gc
            gc.collect()
            
            clear_device_cache()


def register_unified_plugins():
    """Register unified plugins."""
    registry.register_unified_provider(WhisperXUnifiedProvider())


# Auto-register on import
register_unified_plugins()