#!/usr/bin/env python3
"""
Split audio turn builder provider that creates optimal turns and merges them in a single step.
"""

from typing import List, Dict, Optional, Tuple
import re

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class SplitAudioTurnBuilderProvider(TurnBuilderProvider):
    """
    Split audio turn builder that takes individual speaker words data with timestamps,
    creates optimal turns, and merges them into a cohesive transcript.
    
    This approach focuses on:
    1. Preserving sentence完整性 (keeping sentences together)
    2. Natural speaker turn boundaries based on content and timing
    3. Minimizing unnecessary fragmentation of speech
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
        # Group words by speaker first
        speaker_words = self._group_words_by_speaker(words)
        
        # Build optimal turns for each speaker
        speaker_turns = {}
        for speaker, word_list in speaker_words.items():
            speaker_turns[speaker] = self._build_speaker_turns(word_list, **kwargs)
        
        # Merge all speaker turns into a single timeline
        merged_turns = self._merge_speaker_turns(speaker_turns, **kwargs)
        
        return merged_turns

    def _group_words_by_speaker(self, words: List[WordSegment]) -> Dict[str, List[WordSegment]]:
        """Group word segments by speaker."""
        speaker_words = {}
        for word in words:
            if word.speaker not in speaker_words:
                speaker_words[word.speaker] = []
            speaker_words[word.speaker].append(word)
        
        # Sort each speaker's words by timestamp
        for speaker in speaker_words:
            speaker_words[speaker].sort(key=lambda w: w.start)
            
        return speaker_words

    def _build_speaker_turns(self, words: List[WordSegment], **kwargs) -> List[dict]:
        """
        Build optimal turns for a single speaker.
        
        This method focuses on keeping sentences intact and creating natural
        turn boundaries based on content rather than arbitrary time limits.
        """
        if not words:
            return []
            
        # Configuration parameters
        max_gap_s = kwargs.get('max_gap_s', 1.5)  # Increased gap tolerance
        min_turn_duration_s = kwargs.get('min_turn_duration_s', 2.0)  # Minimum turn duration
        max_turn_duration_s = kwargs.get('max_turn_duration_s', 30.0)  # Maximum turn duration
        
        turns = []
        current_turn_words = []
        current_start = None
        
        for i, word in enumerate(words):
            if not word.text.strip():
                continue
                
            # Initialize first turn
            if current_start is None:
                current_start = word.start
                current_turn_words = [word]
                continue
            
            # Check if we should start a new turn
            last_word = current_turn_words[-1]
            gap = word.start - last_word.end
            
            # Conditions for starting a new turn:
            # 1. Large gap between words
            large_gap = gap > max_gap_s
            
            # 2. Sentence boundary with sufficient pause
            sentence_boundary = self._is_sentence_boundary(last_word.text) and gap > 0.5
            
            # 3. Turn getting too long
            turn_duration = word.end - current_start
            too_long = turn_duration > max_turn_duration_s
            
            # 4. Minimum duration met and natural break point
            min_duration_met = (last_word.end - current_start) >= min_turn_duration_s
            natural_break = self._is_natural_break_point(last_word.text, word.text)
            
            if large_gap or (sentence_boundary and min_duration_met) or too_long or (min_duration_met and natural_break):
                # Finalize current turn
                turn_text = " ".join(w.text for w in current_turn_words).strip()
                if turn_text:  # Only add non-empty turns
                    turns.append({
                        "speaker": words[0].speaker,  # All words from same speaker
                        "start": current_start,
                        "end": last_word.end,
                        "text": turn_text
                    })
                
                # Start new turn
                current_start = word.start
                current_turn_words = [word]
            else:
                # Continue current turn
                current_turn_words.append(word)
        
        # Add final turn if there are remaining words
        if current_turn_words:
            turn_text = " ".join(w.text for w in current_turn_words).strip()
            if turn_text:
                turns.append({
                    "speaker": words[0].speaker,
                    "start": current_start,
                    "end": current_turn_words[-1].end,
                    "text": turn_text
                })
        
        return turns

    def _merge_speaker_turns(self, speaker_turns: Dict[str, List[dict]], **kwargs) -> List[Turn]:
        """
        Merge turns from all speakers into a single timeline.
        
        This method intelligently interleaves speaker turns while preserving
        the natural flow of conversation and minimizing fragmentation.
        """
        # Collect all turns with speaker info
        all_turns = []
        for speaker, turns in speaker_turns.items():
            for turn in turns:
                all_turns.append({
                    "speaker": speaker,
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn["text"]
                })
        
        # Sort by start time
        all_turns.sort(key=lambda t: t["start"])
        
        # Resolve overlaps and create final Turn objects
        merged_turns = []
        for turn in all_turns:
            if not merged_turns:
                # First turn
                merged_turns.append(Turn(
                    speaker=turn["speaker"],
                    start=turn["start"],
                    end=turn["end"],
                    text=turn["text"]
                ))
            else:
                last_turn = merged_turns[-1]
                
                # Check for overlap or adjacency
                if turn["start"] <= last_turn.end + 0.1:  # Small tolerance for adjacent turns
                    # Overlapping or adjacent - check if we should merge
                    if self._should_merge_turns(last_turn, turn):
                        # Merge turns
                        merged_text = f"{last_turn.text} {turn['text']}"
                        merged_turns[-1] = Turn(
                            speaker=last_turn.speaker,  # Keep first speaker
                            start=last_turn.start,
                            end=max(last_turn.end, turn["end"]),
                            text=merged_text.strip()
                        )
                    else:
                        # Don't merge - adjust timestamps to avoid overlap
                        if turn["start"] < last_turn.end:
                            # Move start time slightly to avoid overlap
                            adjusted_start = last_turn.end + 0.01
                        else:
                            adjusted_start = turn["start"]
                        
                        merged_turns.append(Turn(
                            speaker=turn["speaker"],
                            start=adjusted_start,
                            end=turn["end"],
                            text=turn["text"]
                        ))
                else:
                    # No overlap - add as separate turn
                    merged_turns.append(Turn(
                        speaker=turn["speaker"],
                        start=turn["start"],
                        end=turn["end"],
                        text=turn["text"]
                    ))
        
        return merged_turns

    def _is_sentence_boundary(self, text: str) -> bool:
        """Check if text ends with a sentence boundary."""
        sentence_endings = r'[.!?]+\s*$'
        return bool(re.search(sentence_endings, text))

    def _is_natural_break_point(self, last_text: str, next_text: str) -> bool:
        """
        Check if there's a natural break point between two words.
        
        This looks for patterns that indicate a good place to split,
        such as clauses, phrases, or other linguistic boundaries.
        """
        # Simple heuristics for natural break points
        last_word = last_text.strip().split()[-1] if last_text.strip() else ""
        next_word = next_text.strip().split()[0] if next_text.strip() else ""
        
        # Break after certain conjunctions or transition words
        break_after = {'and', 'but', 'or', 'so', 'because', 'however', 'therefore'}
        if last_word.lower() in break_after:
            return True
            
        # Break before certain introductory phrases
        break_before = {'well', 'so', 'now', 'okay', 'right'}
        if next_word.lower() in break_before:
            return True
            
        # Break at comma (if it's the last character)
        if last_text.rstrip().endswith(','):
            return True
            
        return False

    def _should_merge_turns(self, turn1: Turn, turn2: dict) -> bool:
        """
        Determine if two turns should be merged.
        
        This prevents excessive fragmentation by merging short,
        related segments from the same speaker.
        """
        # Don't merge different speakers
        if turn1.speaker != turn2["speaker"]:
            return False
            
        # Merge very short turns from same speaker
        if len(turn1.text.split()) < 3 and len(turn2["text"].split()) < 3:
            return True
            
        # Merge if the gap is very small and both are short
        gap = turn2["start"] - turn1.end
        if gap < 0.5 and len(turn1.text.split()) + len(turn2["text"].split()) < 10:
            return True
            
        return False


def register_turn_builder_plugins():
    """Register split audio turn builder plugin."""
    registry.register_turn_builder_provider(SplitAudioTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()