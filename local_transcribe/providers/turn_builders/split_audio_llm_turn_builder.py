#!/usr/bin/env python3
"""
Split audio LLM turn builder provider that uses an LLM to intelligently merge speaker segments into turns.
"""

import json
import re
from typing import List, Dict, Any, Optional

import requests

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class SplitAudioLlmTurnBuilderProvider(TurnBuilderProvider):
    """
    Split audio LLM turn builder that takes individual speaker timestamped data,
    strips duration fields to save context, and uses an LLM to intelligently merge
    text into coherent turns.
    """

    @property
    def name(self) -> str:
        return "split_audio_llm"

    @property
    def short_name(self) -> str:
        return "Split Audio LLM"

    @property
    def description(self) -> str:
        return "LLM-based turn builder that merges speaker segments intelligently while stripping durations for context efficiency"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from word segments using LLM for intelligent merging.

        Args:
            words: Word segments with speaker assignments
            **kwargs: Configuration options (llm_url, timeout, etc.)

        Returns:
            List of Turn objects with merged text and reconstructed timings
        """
        # Validate input
        if not words:
            return []

        # Check that all words have speakers (required for split audio)
        if any(word.speaker is None for word in words):
            raise ValueError("All word segments must have speaker assignments for split audio LLM turn builder")

        # Strip durations and prepare data
        stripped_segments = self._strip_durations(words)

        # Build LLM prompt
        prompt = self._build_prompt(stripped_segments)

        # Query LLM
        llm_url = kwargs.get('llm_url', 'http://0.0.0.0:8080')
        timeout = kwargs.get('timeout', 300)  # Increased default timeout for LLM generation
        try:
            response_text = self._query_llm(prompt, llm_url, timeout)
        except Exception as e:
            print(f"LLM query failed: {e}. Falling back to basic turn building.")
            return self._fallback_build_turns(words)

        # Parse response
        try:
            parsed_turns = self._parse_response(response_text)
        except Exception as e:
            print(f"Failed to parse LLM response: {e}. Falling back to basic turn building.")
            return self._fallback_build_turns(words)

        # Reconstruct timings
        turns = self._reconstruct_timings(parsed_turns, words)

        return turns

    def _strip_durations(self, words: List[WordSegment]) -> List[Dict[str, Any]]:
        """
        Strip duration/timing fields from words, keeping only speaker and text with indices.

        Args:
            words: Original word segments

        Returns:
            List of dicts with speaker, text, and original index
        """
        return [
            {
                "speaker": word.speaker,
                "text": word.text,
                "index": i
            }
            for i, word in enumerate(words)
        ]

    def _build_prompt(self, stripped_segments: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for the LLM.

        Args:
            stripped_segments: Segments with speaker, text, index

        Returns:
            Formatted prompt string
        """
        segments_text = "\n".join([
            f"{i+1}. Speaker {seg['speaker']}: \"{seg['text']}\""
            for i, seg in enumerate(stripped_segments)
        ])

        prompt = f"""You are an expert at merging conversational transcripts into coherent turns. Given the following speaker segments (in order), merge them into logical turns. Each turn should group related sentences or ideas from the same speaker, avoiding unnecessary fragmentation.

Segments:
{segments_text}

Output a JSON list of turns, where each turn has "speaker" and "text" (concatenated from merged segments). Preserve the order and do not reorder speakers.

Example Output:
[{"speaker": "A", "text": "Hello, how are you? I'm doing well."}, {"speaker": "B", "text": "That's great to hear."}]

Important: Only output valid JSON, no additional text."""

        return prompt

    def _query_llm(self, prompt: str, url: str, timeout: int) -> str:
        """
        Query the LLM with the prompt.

        Args:
            prompt: The prompt to send
            url: LLM server URL
            timeout: Request timeout

        Returns:
            LLM response text
        """
        payload = {
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1  # Low temperature for consistent merging
        }

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()

        # Try to extract content from common LLM response formats
        data = response.json()
        if "response" in data:
            return data["response"]
        elif "content" in data:
            return data["content"]
        elif "text" in data:
            return data["text"]
        else:
            # Fallback: return the raw text
            return response.text

    def _parse_response(self, response_text: str) -> List[Dict[str, str]]:
        """
        Parse the LLM response into a list of turns.

        Args:
            response_text: Raw LLM response

        Returns:
            List of dicts with speaker and text
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            turns = json.loads(json_str)
        else:
            # Fallback: try parsing the entire response as JSON
            turns = json.loads(response_text)

        # Validate structure
        if not isinstance(turns, list):
            raise ValueError("LLM response must be a JSON list")

        for turn in turns:
            if not isinstance(turn, dict) or "speaker" not in turn or "text" not in turn:
                raise ValueError("Each turn must have 'speaker' and 'text' fields")

        return turns

    def _reconstruct_timings(self, parsed_turns: List[Dict[str, str]], original_words: List[WordSegment]) -> List[Turn]:
        """
        Reconstruct start/end times for merged turns by mapping back to original words.

        Args:
            parsed_turns: Turns from LLM with speaker and text
            original_words: Original word segments with timings

        Returns:
            List of Turn objects with timings
        """
        turns = []
        word_index = 0

        for turn_data in parsed_turns:
            speaker = turn_data["speaker"]
            turn_text = turn_data["text"]

            # Find the sequence of words that match this turn's text
            matched_words = []
            turn_text_clean = turn_text.replace('"', '').strip()

            # Simple approach: find contiguous words from the same speaker that match the text
            while word_index < len(original_words):
                word = original_words[word_index]
                if word.speaker == speaker:
                    # Check if this word's text is in the turn text
                    if word.text in turn_text_clean:
                        matched_words.append(word)
                        # Remove the matched text from turn_text_clean
                        turn_text_clean = turn_text_clean.replace(word.text, '', 1).strip()
                        word_index += 1
                        # If turn text is consumed, break
                        if not turn_text_clean:
                            break
                    else:
                        # If word doesn't match, move to next
                        word_index += 1
                else:
                    # Different speaker, stop this turn
                    break

            if matched_words:
                start = min(word.start for word in matched_words)
                end = max(word.end for word in matched_words)
                turns.append(Turn(
                    speaker=speaker,
                    start=start,
                    end=end,
                    text=turn_text
                ))
            else:
                # Fallback: if no words matched, skip this turn
                print(f"Warning: Could not match words for turn: {turn_data}")

        return turns

    def _fallback_build_turns(self, words: List[WordSegment]) -> List[Turn]:
        """
        Fallback turn building using simple speaker-based grouping.

        Args:
            words: Original word segments

        Returns:
            List of Turn objects
        """
        turns = []
        current_speaker = None
        current_text = []
        current_start = None
        current_end = None

        for word in words:
            if word.speaker != current_speaker:
                # Flush previous turn
                if current_text:
                    turns.append(Turn(
                        speaker=current_speaker,
                        start=current_start,
                        end=current_end,
                        text=" ".join(current_text)
                    ))
                # Start new turn
                current_speaker = word.speaker
                current_text = [word.text]
                current_start = word.start
                current_end = word.end
            else:
                current_text.append(word.text)
                current_end = word.end

        # Flush last turn
        if current_text:
            turns.append(Turn(
                speaker=current_speaker,
                start=current_start,
                end=current_end,
                text=" ".join(current_text)
            ))

        return turns


def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(SplitAudioLlmTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()