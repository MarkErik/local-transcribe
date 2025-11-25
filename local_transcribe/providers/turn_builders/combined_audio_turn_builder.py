#!/usr/bin/env python3
"""
General turn builder provider.
"""

from typing import List

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class CombinedAudioTurnBuilderProvider(TurnBuilderProvider):
    """Multi-speaker turn builder that groups words into turns based on speaker and timing."""

    @property
    def name(self) -> str:
        return "multi_speaker"

    @property
    def short_name(self) -> str:
        return "Multi-speaker"

    @property
    def description(self) -> str:
        return "Multi-speaker turn builder"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from words with speakers.

        Args:
            words: Word segments with speaker assignments
            **kwargs: Options
        """


    def _build_turns(self, words: List[WordSegment], max_gap_s: float, max_chars: int) -> List[dict]:
        """
        Group word-level tokens into readable turns.
        """



def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(CombinedAudioTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()