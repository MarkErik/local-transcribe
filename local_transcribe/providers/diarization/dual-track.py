#!/usr/bin/env python3
"""
Dual-track diarization plugin implementation.
"""

from typing import List, Optional

from local_transcribe.framework.plugins import DiarizationProvider, WordSegment, Turn, registry


class DualTrackDiarizationProvider(DiarizationProvider):
    """No-op diarization for pre-separated dual-track audio."""

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
        Build turns from words with pre-assigned speakers.

        Args:
            audio_path: Path to audio file (ignored)
            words: Word segments with speaker labels
            **kwargs: Should include 'speaker_label'
        """
        speaker_label = kwargs.get('speaker_label', 'Unknown')

        # Build turns from words
        turns_dicts = self._build_turns(words, speaker_label)

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

    def _build_turns(self, words: List[WordSegment], speaker_label: str, max_gap_s: float = 0.8, max_chars: int = 120):
        """
        Group word-level tokens into readable turns.
        """
        turns = []
        buf = []
        cur_start = None
        last_end = None

        for w in words:
            if not w.text:
                continue
            s, e, t = w.start, w.end, w.text
            if cur_start is None:
                cur_start = s
                buf = [t]
            else:
                gap = s - (last_end if last_end is not None else s)
                if gap > max_gap_s or sum(len(x)+1 for x in buf) + len(t) > max_chars:
                    # flush
                    turns.append({
                        "speaker": speaker_label,
                        "start": cur_start,
                        "end": last_end if last_end is not None else s,
                        "text": " ".join(buf).strip()
                    })
                    # new
                    cur_start = s
                    buf = [t]
                else:
                    buf.append(t)
            last_end = e

        if buf:
            turns.append({
                "speaker": speaker_label,
                "start": cur_start if cur_start is not None else 0.0,
                "end": last_end if last_end is not None else cur_start,
                "text": " ".join(buf).strip()
            })
        return turns


def register_diarization_plugins():
    """Register diarization plugins."""
    registry.register_diarization_provider(DualTrackDiarizationProvider())


# Auto-register on import
register_diarization_plugins()