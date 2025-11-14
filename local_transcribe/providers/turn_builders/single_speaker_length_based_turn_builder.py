#!/usr/bin/env python3
"""
Single speaker length-based turn builder provider.
"""

from typing import List, Optional

from local_transcribe.framework.plugins import TurnBuilderProvider, WordSegment, Turn, registry


class SingleSpeakerLengthBasedTurnBuilderProvider(TurnBuilderProvider):
    """Single speaker turn builder that groups words into turns based on length limits, preserving sentence boundaries."""

    @property
    def name(self) -> str:
        return "single_speaker_length_based"

    @property
    def description(self) -> str:
        return "Single speaker length-based turn builder grouping words by length limits with sentence boundary preservation"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from words (assumes all words from one speaker).

        Args:
            words: Word segments (all from the same speaker)
            **kwargs: Options like max_words
        """
        max_words = kwargs.get('max_words', 500)

        # Build turns
        turns_dicts = self._build_turns(words, max_words)

        # Convert to Turn
        turns = [
            Turn(
                speaker=t['speaker'],
                start=t['start'],
                end=t['end'],
                text=t['text']
            )
            for t in turns_dicts
        ]
        return turns

    def _build_turns(self, words: List[WordSegment], max_words: int) -> List[dict]:
        """
        Group word-level tokens into turns based on word count limits, preferring sentence boundaries.
        """
        sentence_endings = [". ", "! ", "? ", ".", "!", "?"]
        turns = []
        current_words = []
        current_start = None
        current_speaker = None

        for w in words:
            if not w.text:
                continue
            if current_start is None:
                current_start = w.start
                current_speaker = w.speaker

            current_words.append(w)

            # Check if we should split
            if len(current_words) >= max_words:
                # Find the best split point: last sentence end before max_words
                split_index = self._find_split_index(current_words, sentence_endings, max_words)
                if split_index > 0:
                    # Split at sentence end
                    turn_words = current_words[:split_index]
                    remaining_words = current_words[split_index:]
                else:
                    # No sentence end, split at max_words
                    turn_words = current_words[:max_words]
                    remaining_words = current_words[max_words:]

                # Create turn
                turn_text = " ".join(w.text for w in turn_words).strip()
                turn_end = turn_words[-1].end
                turns.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": turn_end,
                    "text": turn_text
                })

                # Reset for next turn
                current_words = remaining_words
                if current_words:
                    current_start = current_words[0].start
                else:
                    current_start = None

        # Add remaining words as final turn
        if current_words:
            turn_text = " ".join(w.text for w in current_words).strip()
            turn_end = current_words[-1].end
            turns.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": turn_end,
                "text": turn_text
            })

        return turns

    def _find_split_index(self, words: List[WordSegment], sentence_endings: List[str], max_words: int) -> int:
        """
        Find the index to split at the last sentence ending before max_words.
        """
        text_so_far = ""
        for i, w in enumerate(words[:max_words]):
            text_so_far += w.text + " "
            # Check if ends with sentence ending
            for ending in sentence_endings:
                if text_so_far.rstrip().endswith(ending.rstrip()):
                    return i + 1  # Include this word
        return 0  # No split found


def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(SingleSpeakerLengthBasedTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()