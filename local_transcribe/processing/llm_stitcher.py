#!/usr/bin/env python3
"""
LLM-based stitcher for merging chunked transcripts.
"""

import json
import requests
from typing import List, Dict, Any, Optional


def stitch_chunks(chunks: List[Dict[str, Any]], **kwargs) -> str:
    """Stitch chunked transcript data using incremental LLM processing."""
    # Allow URL override via kwargs
    url = kwargs.get('llm_stitcher_url', 'http://0.0.0.0:8080')
    if not url.startswith(('http://', 'https://')):
        url = f'http://{url}'
    
    if not chunks:
        return ""

    # Start with the first two chunks
    stitched = _stitch_two_chunks(chunks[0], chunks[1] if len(chunks) > 1 else None, url=url, **kwargs)

    # Iteratively add remaining chunks
    for i in range(2, len(chunks)):
        stitched = _stitch_two_chunks({"chunk_id": 0, "words": stitched.split()}, chunks[i], url=url, **kwargs)

    return stitched


def _stitch_two_chunks(chunk1: Dict[str, Any], chunk2: Optional[Dict[str, Any]], url: str, **kwargs) -> str:
    """Stitch two chunks using the LLM."""
    if chunk2 is None:
        # Only one chunk, return its words
        return " ".join(chunk1["words"])

    # Allow timeout override via kwargs
    timeout = kwargs.get('timeout', None)  # No timeout by default

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
        "max_tokens": 32768,  # Allow enough for merged text
        "temperature": 0.1,  # Low temperature for consistency
        "stream": False
    }

    try:
        response = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
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