#!/usr/bin/env python3
"""
LLM-based stitcher provider for merging chunked transcripts.
"""

import json
import requests
from typing import List, Dict, Any, Optional
from local_transcribe.framework.plugin_interfaces import StitcherProvider, registry


class LLMStitcherProvider(StitcherProvider):
    """Stitcher provider using a remote LLM server to merge chunked transcripts."""

    def __init__(self, url: str = "http://0.0.0.0:8080"):
        self.url = url.rstrip('/')

    @property
    def name(self) -> str:
        return "llm_stitcher"

    @property
    def short_name(self) -> str:
        return "LLM Stitcher"

    @property
    def description(self) -> str:
        return "LLM-based stitching of chunked transcripts"

    def stitch_chunks(self, chunks: List[Dict[str, Any]], **kwargs) -> str:
        """Stitch chunked transcript data using incremental LLM processing."""
        # Allow URL override via kwargs
        url = kwargs.get('llm_stitcher_url', self.url)
        if url != self.url:
            self.url = url.rstrip('/')
        
        if not chunks:
            return ""

        # Start with the first two chunks
        stitched = self._stitch_two_chunks(chunks[0], chunks[1] if len(chunks) > 1 else None, **kwargs)

        # Iteratively add remaining chunks
        for i in range(2, len(chunks)):
            stitched = self._stitch_two_chunks({"chunk_id": 0, "words": stitched.split()}, chunks[i], **kwargs)

        return stitched

    def _stitch_two_chunks(self, chunk1: Dict[str, Any], chunk2: Optional[Dict[str, Any]], **kwargs) -> str:
        """Stitch two chunks using the LLM."""
        if chunk2 is None:
            # Only one chunk, return its words
            return " ".join(chunk1["words"])

        # Prepare the prompt
        words1 = " ".join(chunk1["words"])
        words2 = " ".join(chunk2["words"])

        system_message = (
            "You are an expert at merging transcript chunks."
            "Your task is to combine the provided chunks intelligently as the chunks may have overlapping words."
            "The chunks are from a transcription of audio, and words may have been truncated at boundaries, so there may not be exact word matches at the last words of a chunk, or the first word of a chunk. That is why you have to be intelligent about what is considered an overlap that needs to be merged."
            "Do not add any new words, summarize, or alter the existing content."
            "Only merge the given chunks to resolve overlaps."
            "Return the merged text as a single string."
            "Respond with only the merged text, no explanations or additional content."
        )

        user_content = f"First chunk: {words1}\n\nSecond chunk: {words2}\n\nMerged:"

        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 4096,  # Allow enough for merged text
            "temperature": 0.1,  # Low temperature for consistency
            "stream": False
        }

        try:
            response = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            # Extract the assistant's message
            merged_text = result["choices"][0]["message"]["content"].strip()

            # Quality check: Ensure no new words added
            original_words = set(words1.split() + words2.split())
            merged_words = set(merged_text.split())
            if not merged_words.issubset(original_words):
                print("[!] Warning: LLM added new words during stitching. Using simple concatenation.")
                return f"{words1} {words2}"

            return merged_text

        except requests.RequestException as e:
            print(f"Error communicating with LLM server: {e}")
            return f"{words1} {words2}"  # Fallback to concatenation
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing LLM response: {e}")
            return f"{words1} {words2}"  # Fallback to concatenation


def register_stitcher_plugins():
    """Register stitcher plugins."""
    # Default instance; can be overridden
    provider = LLMStitcherProvider()
    registry.register_stitcher_provider(provider)


# Auto-register on import
register_stitcher_plugins()