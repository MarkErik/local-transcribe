#!/usr/bin/env python3
"""
General turn builder provider.
"""

from typing import List, Optional

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class MultiSpeakerTurnBuilderProvider(TurnBuilderProvider):
    """Multi-speaker turn builder that groups words into turns based on speaker and timing."""

    @property
    def name(self) -> str:
        return "multi_speaker"

    @property
    def short_name(self) -> str:
        return "Multi-speaker"

    @property
    def description(self) -> str:
        return "Multi-speaker turn builder grouping words by speaker changes, gaps, and length"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from words with speakers.

        Args:
            words: Word segments with speaker assignments
            **kwargs: Options like max_gap_s, max_chars
        """
        max_gap_s = kwargs.get('max_gap_s', 0.8)
        max_chars = kwargs.get('max_chars', 120)

        # Build turns
        turns_dicts = self._build_turns(words, max_gap_s, max_chars)

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

    def _build_turns(self, words: List[WordSegment], max_gap_s: float, max_chars: int) -> List[dict]:
        """
        Group word-level tokens into readable turns.
        """
        turns = []
        buf = []
        cur_start = None
        last_end = None
        current_speaker = None

        for w in words:
            if not w.text:
                continue
            s, e, t, speaker = w.start, w.end, w.text, w.speaker
            if cur_start is None:
                cur_start = s
                buf = [t]
                current_speaker = speaker
            else:
                gap = s - (last_end if last_end is not None else s)
                speaker_changed = speaker != current_speaker
                if gap > max_gap_s or speaker_changed or sum(len(x)+1 for x in buf) + len(t) > max_chars:
                    # flush
                    turns.append({
                        "speaker": current_speaker,
                        "start": cur_start,
                        "end": last_end if last_end is not None else s,
                        "text": " ".join(buf).strip()
                    })
                    # new
                    cur_start = s
                    buf = [t]
                    current_speaker = speaker
                else:
                    buf.append(t)
            last_end = e

        if buf:
            turns.append({
                "speaker": current_speaker,
                "start": cur_start if cur_start is not None else 0.0,
                "end": last_end if last_end is not None else cur_start,
                "text": " ".join(buf).strip()
            })
        return turns


def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(MultiSpeakerTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()