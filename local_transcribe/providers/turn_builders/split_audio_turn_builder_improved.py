#!/usr/bin/env python3
"""
Improved split audio turn builder provider that merges speaker turns only on explicit speaker changes.
"""

from typing import List

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class SplitAudioTurnBuilderImprovedProvider(TurnBuilderProvider):
    """
    Improved split audio turn builder that takes word segments from all speakers,
    groups consecutive words by speaker, and merges turns across gaps only if no
    other speaker intervened during the gap.
    """

    @property
    def name(self) -> str:
        return "split_audio_turn_builder_improved"

    @property
    def short_name(self) -> str:
        return "Split Audio Improved"

    @property
    def description(self) -> str:
        return "Timestamp-based turn builder that merges speaker segments only on speaker changes"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from word segments, merging within speakers unless another speaker intervenes.

        Args:
            words: Word segments with speaker assignments from all speakers
            **kwargs: Configuration options (unused for now)

        Returns:
            List of Turn objects
        """
        # Validate input
        if not words:
            return []

        # Ensure all words have speakers
        if any(word.speaker is None or not str(word.speaker).strip() for word in words):
            raise ValueError("All word segments must have speaker assignments")

        # Sort words by start time
        sorted_words = sorted(words, key=lambda w: w.start)

        # Step 1: Create basic turns by grouping consecutive same-speaker words
        basic_turns = self._create_basic_turns(sorted_words)

        # Step 2: Merge turns across gaps if no intervening speakers
        merged_turns = self._merge_turns_across_gaps(basic_turns, sorted_words)

        return merged_turns

    def _create_basic_turns(self, sorted_words: List[WordSegment]) -> List[Turn]:
        """
        Group consecutive words from the same speaker into basic turns.

        Args:
            sorted_words: Words sorted by start time

        Returns:
            List of basic Turn objects
        """
        turns = []
        if not sorted_words:
            return turns

        current_speaker = sorted_words[0].speaker
        current_words = [sorted_words[0]]
        current_start = sorted_words[0].start
        current_end = sorted_words[0].end

        for word in sorted_words[1:]:
            if word.speaker == current_speaker:
                current_words.append(word)
                current_end = word.end
            else:
                # Create turn for previous speaker
                turns.append(Turn(
                    speaker=current_speaker,
                    start=current_start,
                    end=current_end,
                    text=" ".join(w.text for w in current_words)
                ))
                # Start new turn
                current_speaker = word.speaker
                current_words = [word]
                current_start = word.start
                current_end = word.end

        # Add last turn
        turns.append(Turn(
            speaker=current_speaker,
            start=current_start,
            end=current_end,
            text=" ".join(w.text for w in current_words)
        ))

        return turns

    def _merge_turns_across_gaps(self, basic_turns: List[Turn], sorted_words: List[WordSegment]) -> List[Turn]:
        """
        Merge adjacent turns from the same speaker if no other speaker's words intervene.

        Args:
            basic_turns: Basic turns from consecutive grouping
            sorted_words: Original sorted words for checking interventions

        Returns:
            List of merged Turn objects
        """
        if len(basic_turns) <= 1:
            return basic_turns

        merged = [basic_turns[0]]

        for next_turn in basic_turns[1:]:
            prev_turn = merged[-1]
            if prev_turn.speaker == next_turn.speaker:
                # Check for intervening speakers between prev_turn.end and next_turn.start
                intervening = any(
                    w.start >= prev_turn.end and w.start < next_turn.start and w.speaker != prev_turn.speaker
                    for w in sorted_words
                )
                if not intervening:
                    # Merge turns
                    merged[-1] = Turn(
                        speaker=prev_turn.speaker,
                        start=prev_turn.start,
                        end=next_turn.end,
                        text=prev_turn.text + " " + next_turn.text
                    )
                else:
                    merged.append(next_turn)
            else:
                merged.append(next_turn)

        return merged


def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(SplitAudioTurnBuilderImprovedProvider())


# Auto-register on import
register_turn_builder_plugins()