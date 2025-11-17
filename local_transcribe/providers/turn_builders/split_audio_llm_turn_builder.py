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
    includes start and end times for better context, and uses an LLM to intelligently merge
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
        return "LLM-based turn builder that merges speaker segments intelligently using start and end times for better context"

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
            print("DEBUG: No words provided, returning empty list.")
            return []

        print(f"DEBUG: Received {len(words)} word segments.")
        for i, word in enumerate(words[:5]):  # Print first 5 for brevity
            print(f"DEBUG: Word {i}: speaker={word.speaker}, text='{word.text}', start={word.start}, end={word.end}")

        # Check that all words have speakers (required for split audio)
        if any(word.speaker is None for word in words):
            raise ValueError("All word segments must have speaker assignments for split audio LLM turn builder")

        # Prepare segments for LLM
        prepared_segments = self._prepare_segments_for_llm(words)
        print(f"DEBUG: Prepared segments: {prepared_segments[:5]}")  # First 5

        # Build LLM prompt
        prompt = self._build_prompt(prepared_segments)
        print(f"DEBUG: Built prompt (first 500 chars): {prompt[:500]}...")

        # Query LLM
        llm_url = kwargs.get('llm_url', 'http://0.0.0.0:8080')
        timeout = kwargs.get('timeout', None)  # No timeout by default for LLM generation
        try:
            response_text = self._query_llm(prompt, llm_url, timeout)
            print(f"DEBUG: LLM response (first 500 chars): {response_text[:500]}...")
        except Exception as e:
            print(f"DEBUG: LLM query failed: {e}. Falling back to basic turn building.")
            return self._fallback_build_turns(words)

        # Parse response
        try:
            parsed_turns = self._parse_response(response_text)
            print(f"DEBUG: Parsed turns: {parsed_turns}")
        except Exception as e:
            print(f"DEBUG: Failed to parse LLM response: {e}. Falling back to basic turn building.")
            return self._fallback_build_turns(words)

        # Reconstruct timings
        turns = self._reconstruct_timings(parsed_turns, words)
        print(f"DEBUG: Reconstructed turns: {len(turns)} turns.")
        for i, turn in enumerate(turns):
            print(f"DEBUG: Turn {i}: speaker={turn.speaker}, start={turn.start}, end={turn.end}, text='{turn.text}'")

        return turns

    def _prepare_segments_for_llm(self, words: List[WordSegment]) -> List[Dict[str, Any]]:
        """
        Prepare segments for LLM, keeping speaker, text, indices, start and end times.

        Args:
            words: Original word segments

        Returns:
            List of dicts with speaker, text, original index, start and end times
        """
        return [
            {
                "speaker": word.speaker,
                "text": word.text,
                "index": i,
                "start": word.start,
                "end": word.end
            }
            for i, word in enumerate(words)
        ]

    def _build_prompt(self, prepared_segments: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for the LLM.

        Args:
            prepared_segments: Segments with speaker, text, index, start, end

        Returns:
            Formatted prompt string
        """
        segments_text = "\n".join([
            f"{i+1}. Speaker {seg['speaker']}: \"{seg['text']}\" (start: {seg['start']:.2f}, end: {seg['end']:.2f})"
            for i, seg in enumerate(stripped_segments)
        ])

        prompt = f"""Segments:
{segments_text}

Example Output:
[{{"speaker": "A", "text": "Hello, how are you? I'm doing well."}}, {{"speaker": "B", "text": "That's great to hear."}}]"""

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
            "messages": [
                {"role": "system", "content": "You are an expert at merging conversational transcripts into coherent turns. Given the following speaker segments (in order, with start and end times), merge them into logical turns. Each turn should group related sentences or ideas from the same speaker, avoiding unnecessary fragmentation. Consider timing gaps to determine natural turn boundaries. Output a JSON list of turns, where each turn has \"speaker\" and \"text\" (concatenated from merged segments). Preserve the order and do not reorder speakers. Do not add any words. Do not remove any words. Do not re-interpret or summarize. Important: Only output valid JSON, no additional text."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 32768,
            "temperature": 0.1,  # Low temperature for consistent merging
            "stream": False
        }

        response = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

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

        print(f"DEBUG: Starting timing reconstruction with {len(parsed_turns)} parsed turns and {len(original_words)} original words.")

        for turn_data in parsed_turns:
            speaker = turn_data["speaker"]
            turn_text = turn_data["text"]

            print(f"DEBUG: Processing turn for speaker {speaker}: '{turn_text}'")

            # Find the sequence of words that match this turn's text
            matched_words = []
            turn_text_clean = turn_text.replace('"', '').strip()

            print(f"DEBUG: Cleaned turn text: '{turn_text_clean}'")

            # Simple approach: find contiguous words from the same speaker that match the text
            while word_index < len(original_words):
                word = original_words[word_index]
                print(f"DEBUG: Checking word at index {word_index}: speaker={word.speaker}, text='{word.text}'")
                if word.speaker == speaker:
                    # Check if this word's text is in the turn text
                    if word.text in turn_text_clean:
                        matched_words.append(word)
                        print(f"DEBUG: Matched word '{word.text}', remaining text: '{turn_text_clean}'")
                        # Remove the matched text from turn_text_clean
                        turn_text_clean = turn_text_clean.replace(word.text, '', 1).strip()
                        word_index += 1
                        # If turn text is consumed, break
                        if not turn_text_clean:
                            print("DEBUG: Turn text fully consumed.")
                            break
                    else:
                        # If word doesn't match, move to next
                        print(f"DEBUG: Word '{word.text}' not in remaining text, skipping.")
                        word_index += 1
                else:
                    # Different speaker, stop this turn
                    print(f"DEBUG: Different speaker ({word.speaker}), stopping turn.")
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
                print(f"DEBUG: Created turn: start={start}, end={end}")
            else:
                # Fallback: if no words matched, skip this turn
                print(f"DEBUG: Warning: Could not match words for turn: {turn_data}")

        print(f"DEBUG: Reconstruction complete, {len(turns)} turns created.")
        return turns

    def _fallback_build_turns(self, words: List[WordSegment]) -> List[Turn]:
        """
        Fallback turn building using simple speaker-based grouping.

        Args:
            words: Original word segments

        Returns:
            List of Turn objects
        """
        print("DEBUG: Using fallback turn building (simple speaker grouping).")
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

        print(f"DEBUG: Fallback created {len(turns)} turns.")
        return turns


def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(SplitAudioLlmTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()