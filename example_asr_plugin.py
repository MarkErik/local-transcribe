#!/usr/bin/env python3
"""
Example ASR Plugin Template
"""

from local_transcribe.core import ASRProvider, WordSegment, registry

class ExampleASRProvider(ASRProvider):
    """Example ASR provider implementation."""

    @property
    def name(self) -> str:
        return "example-asr"

    @property
    def description(self) -> str:
        return "Example ASR provider (replace with your implementation)"

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Implement your ASR transcription logic here.

        This should return word-level segments with accurate timestamps.
        """
        # TODO: Implement actual transcription
        return [
            WordSegment(
                text="example",
                start=0.0,
                end=1.0,
                speaker=role
            )
        ]

# Register the plugin
registry.register_asr_provider(ExampleASRProvider())
