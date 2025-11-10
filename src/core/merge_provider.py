#!/usr/bin/env python3
"""
Plugin for merging conversation turns from multiple speakers.
"""

from typing import List
from core.plugins import registry
from dual_track.merge import merge_turn_streams as _merge_turn_streams


class TurnMergeProvider:
    """Provider for merging conversation turns from multiple speakers."""

    @property
    def name(self) -> str:
        return "default-merge"

    @property
    def description(self) -> str:
        return "Default turn merging using chronological ordering"

    def merge_turns(self, turn_streams: List[List], **kwargs) -> List:
        """
        Merge multiple turn streams into a single chronological stream.

        Args:
            turn_streams: List of turn lists to merge
            **kwargs: Additional configuration options

        Returns:
            Merged list of turns in chronological order
        """
        if len(turn_streams) == 1:
            return turn_streams[0]
        elif len(turn_streams) == 2:
            return _merge_turn_streams(turn_streams[0], turn_streams[1])
        else:
            # For more than 2 streams, merge pairwise
            result = turn_streams[0]
            for stream in turn_streams[1:]:
                result = _merge_turn_streams(result, stream)
            return result


# Register the default merge provider
merge_provider = TurnMergeProvider()
# Note: We'll add this to the registry when we integrate it into the main system