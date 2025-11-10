#!/usr/bin/env python3
"""
Concrete plugin implementations for existing local-transcribe components.
"""

from typing import List, Optional
from pathlib import Path

from local_transcribe.core.plugins import ASRProvider, DiarizationProvider, OutputWriter, WordSegment, Turn, registry
from local_transcribe.asr.asr import transcribe_with_alignment as _transcribe_with_alignment
from local_transcribe.diarize.diarize import diarize_mixed as _diarize_mixed
from local_transcribe.dual_track.merge import merge_turn_streams
from local_transcribe.dual_track.turns import build_turns as _build_turns
from local_transcribe.output_writers.txt_writer import write_timestamped_txt, write_plain_txt, write_asr_words
from local_transcribe.output_writers.srt_vtt import write_srt, write_vtt
from local_transcribe.output_writers.csv_writer import write_conversation_csv
from local_transcribe.output_writers.markdown_writer import write_conversation_markdown


class WhisperASRProvider(ASRProvider):
    """ASR provider using Whisper-based transcription with alignment."""

    @property
    def name(self) -> str:
        return "whisper"

    @property
    def description(self) -> str:
        return "Whisper-based ASR with faster-whisper and WhisperX alignment"

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio using the existing Whisper pipeline.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            **kwargs: Should include 'asr_model' key
        """
        asr_model = kwargs.get('asr_model', 'medium.en')

        # Call the existing function
        words_dicts = _transcribe_with_alignment(
            audio_path=audio_path,
            asr_model=asr_model,
            role=role
        )

        # Convert to WordSegment objects
        return [
            WordSegment(
                text=word['text'],
                start=word['start'],
                end=word['end'],
                speaker=word['speaker']
            )
            for word in words_dicts
        ]


class PyAnnoteDiarizationProvider(DiarizationProvider):
    """Diarization provider using pyannote.audio."""

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def description(self) -> str:
        return "Speaker diarization using pyannote.audio models"

    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Perform diarization using the existing pyannote pipeline.

        Args:
            audio_path: Path to audio file
            words: Word segments from ASR (speaker should be None for combined mode)
            **kwargs: Additional configuration options
        """
        # Convert WordSegment objects back to dict format expected by existing code
        words_dicts = [
            {
                'text': word.text,
                'start': word.start,
                'end': word.end,
                'speaker': word.speaker
            }
            for word in words
        ]

        # Call existing diarization function
        turns_dicts = _diarize_mixed(audio_path, words_dicts)

        # Convert to Turn objects
        return [
            Turn(
                speaker=turn['speaker'],
                start=turn['start'],
                end=turn['end'],
                text=turn['text']
            )
            for turn in turns_dicts
        ]


class DualTrackDiarizationProvider(DiarizationProvider):
    """Diarization provider for dual-track audio (no actual diarization needed)."""

    @property
    def name(self) -> str:
        return "dual-track"

    @property
    def description(self) -> str:
        return "No-op diarization for pre-separated dual-track audio"

    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        For dual-track mode, speakers are already assigned during ASR.
        Just group words into turns by speaker.

        Args:
            audio_path: Path to audio file (unused)
            words: Word segments with pre-assigned speakers
            **kwargs: Should include 'speaker_label' for the track
        """
        speaker_label = kwargs.get('speaker_label', 'Unknown')

        # Convert WordSegment objects back to dict format
        words_dicts = [
            {
                'text': word.text,
                'start': word.start,
                'end': word.end,
                'speaker': word.speaker
            }
            for word in words
        ]

        # Use existing build_turns function
        turns_dicts = _build_turns(words_dicts, speaker_label)

        # Convert to Turn objects
        return [
            Turn(
                speaker=turn['speaker'],
                start=turn['start'],
                end=turn['end'],
                text=turn['text']
            )
            for turn in turns_dicts
        ]


class TimestampedTextWriter(OutputWriter):
    """Output writer for timestamped text format."""

    @property
    def name(self) -> str:
        return "timestamped-txt"

    @property
    def description(self) -> str:
        return "Timestamped text format with speaker labels"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """Write turns to timestamped text file."""
        # Convert Turn objects back to dict format
        turns_dicts = [
            {
                'speaker': turn.speaker,
                'start': turn.start,
                'end': turn.end,
                'text': turn.text
            }
            for turn in turns
        ]

        write_timestamped_txt(turns_dicts, Path(output_path))


class PlainTextWriter(OutputWriter):
    """Output writer for plain text format."""

    @property
    def name(self) -> str:
        return "plain-txt"

    @property
    def description(self) -> str:
        return "Plain text format without timestamps"

    @property
    def supported_formats(self) -> List[str]:
        return [".txt"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """Write turns to plain text file."""
        # Convert Turn objects back to dict format
        turns_dicts = [
            {
                'speaker': turn.speaker,
                'start': turn.start,
                'end': turn.end,
                'text': turn.text
            }
            for turn in turns
        ]

        write_plain_txt(turns_dicts, Path(output_path))


class SRTWriter(OutputWriter):
    """Output writer for SRT subtitle format."""

    @property
    def name(self) -> str:
        return "srt"

    @property
    def description(self) -> str:
        return "SRT subtitle format"

    @property
    def supported_formats(self) -> List[str]:
        return [".srt"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """Write turns to SRT file."""
        # Convert Turn objects back to dict format
        turns_dicts = [
            {
                'speaker': turn.speaker,
                'start': turn.start,
                'end': turn.end,
                'text': turn.text
            }
            for turn in turns
        ]

        write_srt(turns_dicts, Path(output_path))


class VTTWriter(OutputWriter):
    """Output writer for WebVTT subtitle format."""

    @property
    def name(self) -> str:
        return "vtt"

    @property
    def description(self) -> str:
        return "WebVTT subtitle format"

    @property
    def supported_formats(self) -> List[str]:
        return [".vtt"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """Write turns to VTT file."""
        # Convert Turn objects back to dict format
        turns_dicts = [
            {
                'speaker': turn.speaker,
                'start': turn.start,
                'end': turn.end,
                'text': turn.text
            }
            for turn in turns
        ]

        write_vtt(turns_dicts, Path(output_path))


class CSVWriter(OutputWriter):
    """Output writer for CSV format."""

    @property
    def name(self) -> str:
        return "csv"

    @property
    def description(self) -> str:
        return "CSV format with conversation data"

    @property
    def supported_formats(self) -> List[str]:
        return [".csv"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """Write turns to CSV file."""
        # Convert Turn objects back to dict format
        turns_dicts = [
            {
                'speaker': turn.speaker,
                'start': turn.start,
                'end': turn.end,
                'text': turn.text
            }
            for turn in turns
        ]

        write_conversation_csv(turns_dicts, Path(output_path))


class MarkdownWriter(OutputWriter):
    """Output writer for Markdown format."""

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Markdown format with speaker formatting"

    @property
    def supported_formats(self) -> List[str]:
        return [".md"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """Write turns to Markdown file."""
        # Convert Turn objects back to dict format
        turns_dicts = [
            {
                'speaker': turn.speaker,
                'start': turn.start,
                'end': turn.end,
                'text': turn.text
            }
            for turn in turns
        ]

        write_conversation_markdown(turns_dicts, Path(output_path))


# Register all built-in plugins
def register_builtin_plugins():
    """Register all built-in plugin implementations."""
    registry.register_asr_provider(WhisperASRProvider())
    registry.register_diarization_provider(PyAnnoteDiarizationProvider())
    registry.register_diarization_provider(DualTrackDiarizationProvider())
    registry.register_output_writer(TimestampedTextWriter())
    registry.register_output_writer(PlainTextWriter())
    registry.register_output_writer(SRTWriter())
    registry.register_output_writer(VTTWriter())
    registry.register_output_writer(CSVWriter())
    registry.register_output_writer(MarkdownWriter())


# Auto-register on import
register_builtin_plugins()