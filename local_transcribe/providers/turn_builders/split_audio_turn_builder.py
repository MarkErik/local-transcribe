#!/usr/bin/env python3
"""
Split audio turn builder provider that creates optimal turns and merges them into a transcript.
"""

from typing import List, Dict
import re
import numpy as np

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class SplitAudioTurnBuilderProvider(TurnBuilderProvider):
    """
    Split audio turn builder that takes individual speaker words data with timestamps,
    creates optimal turns, and merges them into a cohesive transcript.
    
    """

    @property
    def name(self) -> str:
        return "split_audio_turn_builder"

    @property
    def short_name(self) -> str:
        return "Split Audio"

    @property
    def description(self) -> str:
        return "Optimal turn builder for split audio mode that creates cohesive turns and merges them"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build and merge turns from word segments with speakers.
        
        Args:
            words: Word segments with speaker assignments (from all speakers)
            **kwargs: Configuration options
            
        Returns:
            List of merged Turn objects ready for output
        """



def register_turn_builder_plugins():
    """Register split audio turn builder plugin."""
    registry.register_turn_builder_provider(SplitAudioTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()