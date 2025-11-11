#!/usr/bin/env python3
"""
Plugin system for local-transcribe components.

This module defines the abstract base classes and plugin registry for extensible
ASR, diarization, and output writer components.
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


class ASRProvider(ABC):
    """Abstract base class for ASR (Automatic Speech Recognition) providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this ASR provider."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this provider."""
        pass

    @abstractmethod
    def get_required_models(self) -> List[str]:
        """Return a list of model identifiers required by this provider (e.g., Hugging Face repo IDs)."""
        pass

    def preload_models(self, models: List[str]) -> None:
        """Preload the specified models to cache. Default implementation does nothing."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure the specified models are available, downloading if necessary. Default implementation does nothing."""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Return a list of available model names for this provider."""
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

    def get_required_models(self) -> List[str]:
        """Return a list of model identifiers required by this provider (e.g., Hugging Face repo IDs). Default empty."""
        return []

    def preload_models(self, models: List[str]) -> None:
        """Preload the specified models to cache. Default implementation does nothing."""
        pass

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure the specified models are available, downloading if necessary. Default implementation does nothing."""
        pass

    @abstractmethod
    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Perform speaker diarization on audio and return conversation turns.

        Args:
            audio_path: Path to the audio file
            words: Word segments from ASR (may have speaker=None)
            **kwargs: Provider-specific configuration options

        Returns:
            List of Turn objects with speaker assignments and timing
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
        self._asr_providers: Dict[str, ASRProvider] = {}
        self._diarization_providers: Dict[str, DiarizationProvider] = {}
        self._output_writers: Dict[str, OutputWriter] = {}

    def register_asr_provider(self, provider: ASRProvider) -> None:
        """Register an ASR provider."""
        self._asr_providers[provider.name] = provider

    def register_diarization_provider(self, provider: DiarizationProvider) -> None:
        """Register a diarization provider."""
        self._diarization_providers[provider.name] = provider

    def register_output_writer(self, writer: OutputWriter) -> None:
        """Register an output writer."""
        self._output_writers[writer.name] = writer

    def get_asr_provider(self, name: str) -> ASRProvider:
        """Get an ASR provider by name."""
        if name not in self._asr_providers:
            available = list(self._asr_providers.keys())
            raise ValueError(f"ASR provider '{name}' not found. Available: {available}")
        return self._asr_providers[name]

    def get_diarization_provider(self, name: str) -> DiarizationProvider:
        """Get a diarization provider by name."""
        if name not in self._diarization_providers:
            available = list(self._diarization_providers.keys())
            raise ValueError(f"Diarization provider '{name}' not found. Available: {available}")
        return self._diarization_providers[name]

    def get_output_writer(self, name: str) -> OutputWriter:
        """Get an output writer by name."""
        if name not in self._output_writers:
            available = list(self._output_writers.keys())
            raise ValueError(f"Output writer '{name}' not found. Available: {available}")
        return self._output_writers[name]

    def list_asr_providers(self) -> Dict[str, str]:
        """List all registered ASR providers with their descriptions."""
        return {name: provider.description for name, provider in self._asr_providers.items()}

    def list_diarization_providers(self) -> Dict[str, str]:
        """List all registered diarization providers with their descriptions."""
        return {name: provider.description for name, provider in self._diarization_providers.items()}

    def list_output_writers(self) -> Dict[str, str]:
        """List all registered output writers with their descriptions."""
        return {name: writer.description for name, writer in self._output_writers.items()}


# Global registry instance
registry = PluginRegistry()