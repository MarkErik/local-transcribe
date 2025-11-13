#!/usr/bin/env python3
"""
Plugin system for local-transcribe components.

This module defines the abstract base classes and plugin registry for extensible
transcription, alignment, diarization, and output writer components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass
import pathlib


@dataclass
class WordSegment:
    """Represents a transcribed word with timing and speaker information."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class Turn:
    """Represents a conversation turn with speaker and timing."""
    speaker: str
    start: float
    end: float
    text: str


class TranscriberProvider(ABC):
    """Abstract base class for transcription providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this transcriber provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this provider."""
        pass

    @property
    @abstractmethod
    def has_builtin_alignment(self) -> bool:
        """Return True if this transcriber provides built-in word-level alignment."""
        pass

    @abstractmethod
    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return a list of model identifiers required by this provider (e.g., Hugging Face repo IDs).

        Args:
            selected_model: The selected model name, if any. If None, return default models.
        """
        pass

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload the specified models to cache. Default implementation does nothing."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure the specified models are available, downloading if necessary. Default implementation does nothing."""
        pass

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline without downloading. Returns list of missing model identifiers."""
        # Default implementation assumes all models are missing if not overridden
        return models

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return a list of available model names for this provider."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        **kwargs
    ) -> str:
        """
        Transcribe audio file and return transcript text.

        Args:
            audio_path: Path to the audio file
            **kwargs: Provider-specific configuration options

        Returns:
            Transcript text as a string
        """
        pass

    @abstractmethod
    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio file and return word-level segments with timestamps.

        Args:
            audio_path: Path to the audio file
            role: Optional speaker role (e.g., "Interviewer", "Participant")
            **kwargs: Provider-specific configuration options

        Returns:
            List of WordSegment objects with text, timing, and speaker info
        """
        pass


class AlignerProvider(ABC):
    """Abstract base class for alignment providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this aligner provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this provider."""
        pass

    @abstractmethod
    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return a list of model identifiers required by this provider (e.g., Hugging Face repo IDs).

        Args:
            selected_model: The selected model name, if any. If None, return default models.
        """
        pass

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload the specified models to cache. Default implementation does nothing."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure the specified models are available, downloading if necessary. Default implementation does nothing."""
        pass

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline without downloading. Returns list of missing model identifiers."""
        # Default implementation assumes all models are missing if not overridden
        return models

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return a list of available model names for this provider."""
        pass

    @abstractmethod
    def align_transcript(
        self,
        audio_path: str,
        transcript: str,
        **kwargs
    ) -> List[WordSegment]:
        """
        Align transcript text to audio and return word-level segments with timestamps.

        Args:
            audio_path: Path to the audio file
            transcript: Transcript text to align
            **kwargs: Provider-specific configuration options

        Returns:
            List of WordSegment objects with text, timing, and speaker info
        """
        pass


class DiarizationProvider(ABC):
    """Abstract base class for speaker diarization providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this diarization provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this provider."""
        pass

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return a list of model identifiers required by this provider (e.g., Hugging Face repo IDs). Default empty."""
        return []

    def preload_models(self, models: List[str]) -> None:
        """Preload the specified models to cache. Default implementation does nothing."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure the specified models are available, downloading if necessary. Default implementation does nothing."""
        pass

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline without downloading. Returns list of missing model identifiers."""
        # Default implementation assumes all models are missing if not overridden
        return models

    @abstractmethod
    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        num_speakers: int,
        **kwargs
    ) -> List[WordSegment]:
        """
        Perform speaker diarization on audio and assign speakers to words.

        Args:
            audio_path: Path to the audio file
            words: Word segments from ASR (speaker=None)
            num_speakers: Number of speakers expected in the audio
            **kwargs: Provider-specific configuration options

        Returns:
            List of WordSegment objects with speakers assigned
        """
        pass


class UnifiedProvider(ABC):
    """Abstract base class for unified ASR + diarization providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this unified provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this provider."""
        pass

    @abstractmethod
    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return a list of model identifiers required by this provider (e.g., Hugging Face repo IDs).

        Args:
            selected_model: The selected model name, if any. If None, return default models.
        """
        pass

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload the specified models to cache. Default implementation does nothing."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure the specified models are available, downloading if necessary. Default implementation does nothing."""
        pass

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline without downloading. Returns list of missing model identifiers."""
        # Default implementation assumes all models are missing if not overridden
        return models

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return a list of available model names for this provider."""
        pass

    @abstractmethod
    def transcribe_and_diarize(
        self,
        audio_path: str,
        num_speakers: int,
        **kwargs
    ) -> List[Turn]:
        """
        Transcribe audio file with word-level timestamps and perform speaker diarization.

        Args:
            audio_path: Path to the audio file
            num_speakers: Number of speakers expected in the audio            
            **kwargs: Provider-specific configuration options

        Returns:
            List of Turn objects with speaker assignments, timing, and text
        """
        pass


class TurnBuilderProvider(ABC):
    """Abstract base class for turn building providers (grouping words into turns)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this turn builder provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this provider."""
        pass

    @abstractmethod
    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build conversation turns from word segments with speakers.

        Args:
            words: Word segments with speaker assignments
            **kwargs: Provider-specific configuration options

        Returns:
            List of Turn objects grouped by speaker and timing
        """
        pass


class WordWriter(ABC):
    """Abstract base class for word output writers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this word writer."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this writer."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of file extensions this writer supports (e.g., ['.txt'])."""
        pass

    @abstractmethod
    def write(
        self,
        words: List[WordSegment],
        output_path: str,
        **kwargs
    ) -> None:
        """
        Write ASR word segments to output file.

        Args:
            words: List of word segments from ASR (without speaker assignments)
            output_path: Path where to write the output file
            **kwargs: Writer-specific configuration options
        """
        pass


class OutputWriter(ABC):
    """Abstract base class for output format writers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this output writer."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this writer."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of file extensions this writer supports (e.g., ['.txt', '.srt'])."""
        pass

    @abstractmethod
    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """
        Write conversation turns to output file.

        Args:
            turns: List of conversation turns to write
            output_path: Path where to write the output file
            **kwargs: Writer-specific configuration options
        """
        pass


class PluginRegistry:
    """Registry for managing and discovering plugins."""

    def __init__(self):
        self._transcriber_providers: Dict[str, TranscriberProvider] = {}
        self._aligner_providers: Dict[str, AlignerProvider] = {}
        self._diarization_providers: Dict[str, DiarizationProvider] = {}
        self._unified_providers: Dict[str, UnifiedProvider] = {}
        self._turn_builder_providers: Dict[str, TurnBuilderProvider] = {}
        self._word_writers: Dict[str, WordWriter] = {}
        self._output_writers: Dict[str, OutputWriter] = {}

    def register_transcriber_provider(self, provider: TranscriberProvider) -> None:
        """Register a transcriber provider."""
        self._transcriber_providers[provider.name] = provider

    def register_aligner_provider(self, provider: AlignerProvider) -> None:
        """Register an aligner provider."""
        self._aligner_providers[provider.name] = provider

    def register_diarization_provider(self, provider: DiarizationProvider) -> None:
        """Register a diarization provider."""
        self._diarization_providers[provider.name] = provider

    def register_unified_provider(self, provider: UnifiedProvider) -> None:
        """Register a unified provider."""
        self._unified_providers[provider.name] = provider

    def register_turn_builder_provider(self, provider: TurnBuilderProvider) -> None:
        """Register a turn builder provider."""
        self._turn_builder_providers[provider.name] = provider

    def register_word_writer(self, writer: WordWriter) -> None:
        """Register a word writer."""
        self._word_writers[writer.name] = writer

    def register_output_writer(self, writer: OutputWriter) -> None:
        """Register an output writer."""
        self._output_writers[writer.name] = writer

    def get_transcriber_provider(self, name: str) -> TranscriberProvider:
        """Get a transcriber provider by name."""
        if name not in self._transcriber_providers:
            available = list(self._transcriber_providers.keys())
            raise ValueError(f"Transcriber provider '{name}' not found. Available: {available}")
        return self._transcriber_providers[name]

    def get_aligner_provider(self, name: str) -> AlignerProvider:
        """Get an aligner provider by name."""
        if name not in self._aligner_providers:
            available = list(self._aligner_providers.keys())
            raise ValueError(f"Aligner provider '{name}' not found. Available: {available}")
        return self._aligner_providers[name]

    def get_diarization_provider(self, name: str) -> DiarizationProvider:
        """Get a diarization provider by name."""
        if name not in self._diarization_providers:
            available = list(self._diarization_providers.keys())
            raise ValueError(f"Diarization provider '{name}' not found. Available: {available}")
        return self._diarization_providers[name]

    def get_unified_provider(self, name: str) -> UnifiedProvider:
        """Get a unified provider by name."""
        if name not in self._unified_providers:
            available = list(self._unified_providers.keys())
            raise ValueError(f"Unified provider '{name}' not found. Available: {available}")
        return self._unified_providers[name]

    def get_turn_builder_provider(self, name: str) -> TurnBuilderProvider:
        """Get a turn builder provider by name."""
        if name not in self._turn_builder_providers:
            available = list(self._turn_builder_providers.keys())
            raise ValueError(f"Turn builder provider '{name}' not found. Available: {available}")
        return self._turn_builder_providers[name]

    def get_word_writer(self, name: str) -> WordWriter:
        """Get a word writer by name."""
        if name not in self._word_writers:
            available = list(self._word_writers.keys())
            raise ValueError(f"Word writer '{name}' not found. Available: {available}")
        return self._word_writers[name]

    def get_output_writer(self, name: str) -> OutputWriter:
        """Get an output writer by name."""
        if name not in self._output_writers:
            available = list(self._output_writers.keys())
            raise ValueError(f"Output writer '{name}' not found. Available: {available}")
        return self._output_writers[name]

    def list_transcriber_providers(self) -> Dict[str, str]:
        """List all registered transcriber providers with their descriptions."""
        return {name: provider.description for name, provider in self._transcriber_providers.items()}

    def list_aligner_providers(self) -> Dict[str, str]:
        """List all registered aligner providers with their descriptions."""
        return {name: provider.description for name, provider in self._aligner_providers.items()}

    def list_diarization_providers(self) -> Dict[str, str]:
        """List all registered diarization providers with their descriptions."""
        return {name: provider.description for name, provider in self._diarization_providers.items()}

    def list_unified_providers(self) -> Dict[str, str]:
        """List all registered unified providers with their descriptions."""
        return {name: provider.description for name, provider in self._unified_providers.items()}

    def list_turn_builder_providers(self) -> Dict[str, str]:
        """List all registered turn builder providers with their descriptions."""
        return {name: provider.description for name, provider in self._turn_builder_providers.items()}

    def list_word_writers(self) -> Dict[str, str]:
        """List all registered word writers with their descriptions."""
        return {name: writer.description for name, writer in self._word_writers.items()}

    def list_output_writers(self) -> Dict[str, str]:
        """List all registered output writers with their descriptions."""
        return {name: writer.description for name, writer in self._output_writers.items()}


# Global registry instance
registry = PluginRegistry()