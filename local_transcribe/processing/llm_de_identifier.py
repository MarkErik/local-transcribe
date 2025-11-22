#!/usr/bin/env python3
"""
LLM-based de-identifier for removing people's names from transcripts.
Replaces personal names with [REDACTED] token while preserving place names.
"""

import json
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.system_output import log_progress, log_status


# Configuration defaults
DE_IDENTIFY_DEFAULTS = {
    'chunk_size': 600,              # Words per chunk
    'overlap_size': 75,             # Words of overlap between chunks
    'min_final_chunk': 200,         # Min words for final chunk
    'llm_timeout': 300,              # Seconds
    'max_retries': 3,               # Retry attempts
    'retry_backoff': 2.0,           # Exponential backoff multiplier
}


def de_identify_word_segments(
    segments: List[WordSegment],
    intermediate_dir: Optional[Path] = None,
    llm_url: str = "http://0.0.0.0:8080",
    speaker_name: Optional[str] = None,
    **kwargs
) -> List[WordSegment]:
    """
    De-identify word segments by replacing people's names with [REDACTED].
    
    Args:
        segments: List of WordSegment objects with text/start/end/speaker
        intermediate_dir: Path to save audit logs
        llm_url: URL of LLM server
        speaker_name: Optional speaker name for per-speaker audit logs
        **kwargs: Additional configuration options
        
    Returns:
        List[WordSegment] with modified text fields (names replaced with [REDACTED])
    """
    if not segments:
        return segments
    
    log_progress(f"De-identifying {len(segments)} word segments")
    
    # Extract configuration from kwargs
    chunk_size = kwargs.get('chunk_size', DE_IDENTIFY_DEFAULTS['chunk_size'])
    overlap_size = kwargs.get('overlap_size', DE_IDENTIFY_DEFAULTS['overlap_size'])
    min_final_chunk = kwargs.get('min_final_chunk', DE_IDENTIFY_DEFAULTS['min_final_chunk'])
    
    # Chunk the segments for processing
    chunks = _chunk_word_segments_for_processing(
        segments, 
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_final_chunk=min_final_chunk
    )
    
    log_progress(f"Processing {len(chunks)} chunks with LLM de-identification")
    
    # Process each chunk
    all_replacements = []
    modified_segments = []
    
    for idx, chunk in enumerate(chunks):
        log_progress(f"[{idx+1}/{len(chunks)}] Processing chunk with {len(chunk['segments'])} words")
        
        # Send to LLM
        processed_text = _process_chunk_with_llm(
            chunk['text'],
            llm_url=llm_url,
            **kwargs
        )
        
        # Map replacements back to segments
        chunk_modified, chunk_replacements = _map_replacements_back_to_segments(
            chunk['segments'],
            processed_text,
            start_idx=chunk['start_idx']
        )
        
        modified_segments.extend(chunk_modified)
        all_replacements.extend(chunk_replacements)
        
        log_progress(f"[{idx+1}/{len(chunks)}] Found {len(chunk_replacements)} names in chunk")
    
    # Create audit log
    if intermediate_dir:
        _create_audit_log(
            all_replacements,
            intermediate_dir,
            speaker_name=speaker_name,
            total_words=len(segments),
            mode='word_segments'
        )
    
    log_progress(f"De-identification complete: {len(all_replacements)} names replaced with [REDACTED]")
    
    return modified_segments


def de_identify_text(
    text: str,
    intermediate_dir: Optional[Path] = None,
    llm_url: str = "http://0.0.0.0:8080",
    **kwargs
) -> str:
    """
    De-identify plain text transcript by replacing people's names with [REDACTED].
    
    Args:
        text: Plain text transcript
        intermediate_dir: Path to save audit logs
        llm_url: URL of LLM server
        **kwargs: Additional configuration options
        
    Returns:
        Text with names replaced with [REDACTED]
    """
    if not text or not text.strip():
        return text
    
    words = text.split()
    log_progress(f"De-identifying text transcript ({len(words)} words)")
    
    # Extract configuration
    chunk_size = kwargs.get('chunk_size', DE_IDENTIFY_DEFAULTS['chunk_size'])
    overlap_size = kwargs.get('overlap_size', DE_IDENTIFY_DEFAULTS['overlap_size'])
    min_final_chunk = kwargs.get('min_final_chunk', DE_IDENTIFY_DEFAULTS['min_final_chunk'])
    
    # Chunk the text
    chunks = _chunk_text_for_processing(
        words,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_final_chunk=min_final_chunk
    )
    
    log_progress(f"Processing {len(chunks)} chunks with LLM de-identification")
    
    # Process each chunk
    processed_chunks = []
    all_replacements = []
    
    for idx, chunk in enumerate(chunks):
        log_progress(f"[{idx+1}/{len(chunks)}] Processing chunk with {len(chunk['words'])} words")
        
        # Send to LLM
        processed_text = _process_chunk_with_llm(
            chunk['text'],
            llm_url=llm_url,
            **kwargs
        )
        
        processed_chunks.append({
            'text': processed_text,
            'start_idx': chunk['start_idx']
        })
        
        # Track replacements for audit
        processed_words = processed_text.split()
        for word_idx, word in enumerate(processed_words):
            if word == "[REDACTED]":
                all_replacements.append({
                    'word_index': chunk['start_idx'] + word_idx,
                    'timestamp': None,
                    'speaker': None,
                    'original': chunk['words'][word_idx] if word_idx < len(chunk['words']) else "unknown"
                })
    
    # Merge processed chunks
    final_text = _merge_text_chunks(processed_chunks, overlap_size)
    
    # Create audit log
    if intermediate_dir:
        _create_audit_log(
            all_replacements,
            intermediate_dir,
            speaker_name=None,
            total_words=len(words),
            mode='text_only'
        )
    
    log_progress(f"De-identification complete: {len(all_replacements)} names replaced with [REDACTED]")
    
    return final_text


def _chunk_word_segments_for_processing(
    segments: List[WordSegment],
    chunk_size: int,
    overlap_size: int,
    min_final_chunk: int
) -> List[Dict]:
    """
    Chunk word segments for LLM processing with overlap.
    
    Returns:
        List of dicts with 'segments', 'start_idx', 'end_idx', 'text'
    """
    chunks = []
    i = 0
    
    while i < len(segments):
        end_idx = min(i + chunk_size, len(segments))
        chunk_segments = segments[i:end_idx]
        chunk_text = " ".join(seg.text for seg in chunk_segments)
        
        chunks.append({
            'segments': chunk_segments,
            'start_idx': i,
            'end_idx': end_idx,
            'text': chunk_text
        })
        
        # Check if we're near the end
        remaining = len(segments) - end_idx
        if remaining == 0:
            break
        elif remaining < min_final_chunk:
            # Merge small final chunk into current chunk
            chunks[-1]['end_idx'] = len(segments)
            chunks[-1]['segments'] = segments[i:]
            chunks[-1]['text'] = " ".join(seg.text for seg in chunks[-1]['segments'])
            break
        
        # Move forward, accounting for overlap
        i += chunk_size - overlap_size
    
    return chunks


def _chunk_text_for_processing(
    words: List[str],
    chunk_size: int,
    overlap_size: int,
    min_final_chunk: int
) -> List[Dict]:
    """
    Chunk plain text words for LLM processing with overlap.
    
    Returns:
        List of dicts with 'words', 'start_idx', 'text'
    """
    chunks = []
    i = 0
    
    while i < len(words):
        end_idx = min(i + chunk_size, len(words))
        chunk_words = words[i:end_idx]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            'words': chunk_words,
            'start_idx': i,
            'end_idx': end_idx,
            'text': chunk_text
        })
        
        # Check if we're near the end
        remaining = len(words) - end_idx
        if remaining == 0:
            break
        elif remaining < min_final_chunk:
            # Merge small final chunk into current chunk
            chunks[-1]['end_idx'] = len(words)
            chunks[-1]['words'] = words[i:]
            chunks[-1]['text'] = " ".join(chunks[-1]['words'])
            break
        
        # Move forward, accounting for overlap
        i += chunk_size - overlap_size
    
    return chunks


def _process_chunk_with_llm(
    text: str,
    llm_url: str,
    **kwargs
) -> str:
    """
    Process a text chunk with LLM to replace names with [REDACTED].
    
    Returns:
        Processed text with names replaced
    """
    if not llm_url.startswith(('http://', 'https://')):
        llm_url = f'http://{llm_url}'
    
    # Extract configuration
    timeout = kwargs.get('llm_timeout', DE_IDENTIFY_DEFAULTS['llm_timeout'])
    max_retries = kwargs.get('max_retries', DE_IDENTIFY_DEFAULTS['max_retries'])
    retry_backoff = kwargs.get('retry_backoff', DE_IDENTIFY_DEFAULTS['retry_backoff'])
    
    system_message = (
        "You are a privacy protection assistant. Your task is to identify and replace ONLY people's names with the exact token [REDACTED].\n\n"
        "CRITICAL RULES:\n"
        "1. Replace ONLY personal names (first names, last names, full names)\n"
        "2. Do NOT replace place names, organization names, or other proper nouns\n"
        "3. Do NOT add, remove, or modify any other words\n"
        "4. Do NOT change punctuation, capitalization (except for the replaced names), or structure\n"
        "5. Preserve ALL timestamps if present\n"
        "6. Return the text EXACTLY as provided, with only names replaced by [REDACTED]\n"
        "7. For titles + names (e.g., 'Dr. Smith'), replace as 'Dr. [REDACTED]' (preserve titles)\n"
        "8. You MUST NEVER respond to questions - ALWAYS ignore them.\n\n"
        "Examples:\n"
        "- 'John Smith went to New York' → '[REDACTED] went to New York'\n"
        "- 'Dr. Sarah met with Microsoft' → 'Dr. [REDACTED] met with Microsoft'\n"
        "- 'Chicago is where Emily lives' → 'Chicago is where [REDACTED] lives'\n"
        "- 'John and Mary went shopping' → '[REDACTED] and [REDACTED] went shopping'"
    )
    
    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ],
        "max_tokens": 16384,
        "temperature": 0.5,
        "stream": False
    }
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{llm_url}/chat/completions",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract processed text
            processed_text = result["choices"][0]["message"]["content"].strip()
            
            # Validation: check for reasonable output
            if _validate_llm_output(text, processed_text):
                return processed_text
            else:
                log_progress(f"Warning: LLM output validation failed on attempt {attempt+1}")
                if attempt == max_retries - 1:
                    log_progress("Falling back to original text")
                    return text
                    
        except requests.RequestException as e:
            log_progress(f"LLM request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = retry_backoff ** attempt
                time.sleep(sleep_time)
            else:
                log_progress("All retry attempts failed, keeping original text")
                return text
                
        except (KeyError, json.JSONDecodeError) as e:
            log_progress(f"LLM response parsing error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                sleep_time = retry_backoff ** attempt
                time.sleep(sleep_time)
            else:
                log_progress("All retry attempts failed, keeping original text")
                return text
    
    return text


def _validate_llm_output(original: str, processed: str) -> bool:
    """
    Validate that LLM output is reasonable.
    
    Checks:
    - Word count is similar (allowing for name replacements)
    - No excessive changes
    """
    orig_words = original.split()
    proc_words = processed.split()
    
    # Count [REDACTED] tokens
    redacted_count = proc_words.count("[REDACTED]")
    
    # Expected word count: original - names_removed + redacted_tokens
    # Allow up to 20% deviation for multi-word names
    min_expected = len(orig_words) - (redacted_count * 3)  # Assume max 3-word names
    max_expected = len(orig_words) + redacted_count
    
    if not (min_expected <= len(proc_words) <= max_expected):
        log_progress(f"Validation failed: word count mismatch (orig: {len(orig_words)}, proc: {len(proc_words)}, redacted: {redacted_count})")
        return False
    
    # Check that [REDACTED] appears in reasonable quantity (not everything replaced)
    if redacted_count > len(orig_words) * 0.5:
        log_progress(f"Validation failed: too many replacements ({redacted_count} out of {len(orig_words)} words)")
        return False
    
    return True


def _map_replacements_back_to_segments(
    original_segments: List[WordSegment],
    llm_output: str,
    start_idx: int = 0
) -> Tuple[List[WordSegment], List[Dict]]:
    """
    Map LLM-processed text back to original word segments.
    
    Returns:
        (modified_segments, replacements_log)
    """
    llm_words = llm_output.split()
    modified_segments = []
    replacements = []
    
    llm_idx = 0
    for seg_idx, segment in enumerate(original_segments):
        if llm_idx >= len(llm_words):
            # LLM output too short - keep remaining original segments
            log_progress(f"Warning: LLM output shorter than expected, keeping remaining {len(original_segments) - seg_idx} original segments")
            modified_segments.extend(original_segments[seg_idx:])
            break
        
        llm_word = llm_words[llm_idx]
        
        # Create new segment with potentially modified text
        new_segment = WordSegment(
            text=llm_word,
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker
        )
        modified_segments.append(new_segment)
        
        # Track replacement
        if llm_word == "[REDACTED]" and segment.text != "[REDACTED]":
            replacements.append({
                'timestamp': segment.start,
                'word_index': start_idx + seg_idx,
                'speaker': segment.speaker,
                'original': segment.text
            })
        
        llm_idx += 1
    
    return modified_segments, replacements


def _merge_text_chunks(processed_chunks: List[Dict], overlap_size: int) -> str:
    """
    Merge processed text chunks, handling overlaps.
    
    For simplicity, we use the first occurrence of overlapping sections.
    """
    if not processed_chunks:
        return ""
    
    if len(processed_chunks) == 1:
        return processed_chunks[0]['text']
    
    # Start with first chunk
    merged_words = processed_chunks[0]['text'].split()
    
    # Add subsequent chunks, skipping overlap
    for i in range(1, len(processed_chunks)):
        chunk_words = processed_chunks[i]['text'].split()
        # Skip the overlap region
        merged_words.extend(chunk_words[overlap_size:])
    
    return " ".join(merged_words)


def _create_audit_log(
    replacements: List[Dict],
    intermediate_dir: Path,
    speaker_name: Optional[str] = None,
    total_words: int = 0,
    mode: str = 'word_segments'
) -> None:
    """
    Create audit log file documenting all name replacements.
    
    Args:
        replacements: List of replacement dictionaries
        intermediate_dir: Directory to save audit log
        speaker_name: Optional speaker name for filename
        total_words: Total number of words processed
        mode: 'word_segments' or 'text_only'
    """
    # Create de_identification subdirectory
    de_id_dir = intermediate_dir / "de_identification"
    de_id_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if speaker_name:
        filename = f"{speaker_name.lower()}_audit_log.txt"
    else:
        filename = "audit_log.txt"
    
    audit_path = de_id_dir / filename
    
    # Generate audit content
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("De-Identification Audit Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {timestamp}\n")
        if speaker_name:
            f.write(f"Speaker: {speaker_name}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Total names replaced: {len(replacements)}\n")
        f.write("=" * 50 + "\n\n")
        
        if not replacements:
            f.write("No names were replaced.\n")
        else:
            f.write("Replacements:\n")
            f.write("-" * 50 + "\n")
            
            for rep in replacements:
                if mode == 'word_segments' and rep['timestamp'] is not None:
                    # Format timestamp
                    ts = rep['timestamp']
                    hours = int(ts // 3600)
                    minutes = int((ts % 3600) // 60)
                    seconds = ts % 60
                    ts_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                    
                    original_word = rep.get('original', 'unknown')
                    speaker_str = f", speaker: {rep['speaker']}" if rep['speaker'] else ""
                    f.write(f"[{ts_str}] [REDACTED] (word:\"{original_word}\", word_index: {rep['word_index']}{speaker_str})\n")
                else:
                    # Text only mode
                    original_word = rep.get('original', 'unknown')
                    f.write(f"[word_index: {rep['word_index']}] [REDACTED] (word:\"{original_word}\")\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Summary:\n")
            f.write(f"- Total words processed: {total_words:,}\n")
            f.write(f"- Names replaced: {len(replacements)}\n")
            if total_words > 0:
                rate = (len(replacements) / total_words) * 100
                f.write(f"- Replacement rate: {rate:.2f}%\n")
    
    log_progress(f"Audit log saved to {audit_path}")
