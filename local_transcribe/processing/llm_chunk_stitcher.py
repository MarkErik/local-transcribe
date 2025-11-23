#!/usr/bin/env python3
"""
LLM-based stitcher for merging chunked transcripts.
"""

import json
import requests
from typing import List, Dict, Any, Optional
from local_transcribe.lib.program_logger import log_progress


def stitch_chunks(chunks: List[Dict[str, Any]], **kwargs) -> str:
    """Stitch chunked transcript data using incremental LLM processing."""
    # Allow URL override via kwargs
    url = kwargs.get('llm_stitcher_url', 'http://0.0.0.0:8080')
    if not url.startswith(('http://', 'https://')):
        url = f'http://{url}'
    
    # Internal settings for stitching optimization
    use_accumulated = False
    overlap_window = 100
    
    if not chunks:
        return ""

    log_progress(f"Processing {len(chunks)} chunks with LLM stitching")

    # Start with the first two chunks
    stitched = _stitch_two_chunks(chunks[0], chunks[1] if len(chunks) > 1 else None, url=url, **kwargs)

    # Iteratively add remaining chunks
    for i in range(2, len(chunks)):
        log_progress(f"Adding chunk {i + 1} of {len(chunks)} to stitched text")
        stitched_words = stitched.split()
        prefix = ""
        if not use_accumulated and len(stitched_words) > overlap_window:
            prefix = " ".join(stitched_words[:-overlap_window]) + " "
        if not use_accumulated:
            chunk1_words = stitched_words[-overlap_window:] if len(stitched_words) > overlap_window else stitched_words
        else:
            chunk1_words = stitched_words
        stitched = _stitch_two_chunks({"chunk_id": 0, "words": chunk1_words}, chunks[i], url=url, **kwargs)
        stitched = prefix + stitched

    word_count = len(stitched.split())
    log_progress(f"LLM stitching complete: {word_count} words total")

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

    log_progress(f"Stitching: '{words1[:50]}...' + '{words2[:50]}...'")

    system_message = (
        "You are an expert at merging transcript chunks."
        "Your task is to combine the provided chunks intelligently as the chunks may have overlapping words."
        "The chunks are from a transcription of audio, and words may have been truncated at boundaries, so there may not be exact word matches at the last words of a chunk, or the first word of a chunk. That is why you have to be intelligent about what is considered an overlap that needs to be merged."
        "Do not add any new words, summarize, or alter the existing content."
        "Only merge the given chunks to resolve overlaps."
        "Return the merged text as a single string."
        "Respond with only the merged text, no explanations, or additional content."
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

        log_progress(f"LLM result: '{merged_text[:100]}...'")

        # Quality check: Ensure no new words added
        original_words = set(words1.split() + words2.split())
        merged_words = set(merged_text.split())
        if not merged_words.issubset(original_words):
            log_progress("Quality check failed: LLM added unexpected words")
            print("[!] Warning: LLM added new words during stitching. Using simple concatenation.")
            return f"{words1} {words2}"

        return merged_text

    except requests.RequestException as e:
        log_progress("LLM stitching failed (network error); falling back to concatenation")
        print(f"Error communicating with LLM server: {e}")
        return f"{words1} {words2}"  # Fallback to concatenation
    except (KeyError, json.JSONDecodeError) as e:
        log_progress("LLM stitching failed (response parsing error); falling back to concatenation")
        print(f"Error parsing LLM response: {e}")
        return f"{words1} {words2}"  # Fallback to concatenation