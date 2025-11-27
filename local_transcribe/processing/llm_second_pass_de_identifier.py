#!/usr/bin/env python3
"""
Second-pass LLM de-identifier for catching missed names using a global name list.

This module performs a targeted review of already-redacted transcripts using names
discovered across all speakers in the first pass. The LLM is given explicit context
about which names to look for, improving detection of missed instances.
"""

import json
import re
import requests
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_progress, log_status, log_debug


# Configuration defaults
SECOND_PASS_DEFAULTS = {
    'chunk_size': 400,            # Words per chunk
    'overlap_size': 75,           # Words of overlap between chunks
    'min_final_chunk': 200,       # Min words for final chunk
    'llm_timeout': 300,           # Seconds
    'temperature': 1.0,           # Temperature for LLM
    'parse_harmony': True,        # Parse Harmony format responses
}


@dataclass
class DiscoveredName:
    """Represents a name discovered during first-pass de-identification."""
    name: str
    source_speaker: Optional[str] = None
    occurrences: int = 1
    
    def __hash__(self):
        return hash(self.name.lower())
    
    def __eq__(self, other):
        if isinstance(other, DiscoveredName):
            return self.name.lower() == other.name.lower()
        return False


@dataclass 
class SecondPassResult:
    """Result from second-pass de-identification."""
    segments: List[WordSegment]
    additional_replacements: List[Dict]
    names_from_list_found: Set[str] = field(default_factory=set)


def extract_names_from_replacements(replacements: List[Dict]) -> Set[str]:
    """
    Extract unique names from first-pass replacement records.
    
    Args:
        replacements: List of replacement dicts with 'original' key
        
    Returns:
        Set of unique name strings (lowercased for deduplication, original case preserved)
    """
    names = set()
    for rep in replacements:
        original = rep.get('original', '')
        if original and original != '[REDACTED]' and original.strip():
            # Keep original casing but deduplicate case-insensitively
            names.add(original.strip())
    return names


def build_global_name_list(
    all_speaker_replacements: Dict[str, List[Dict]]
) -> List[DiscoveredName]:
    """
    Build a global list of discovered names from all speakers' first-pass results.
    
    Args:
        all_speaker_replacements: Dict mapping speaker_name -> list of replacements
        
    Returns:
        List of DiscoveredName objects with occurrence counts
    """
    name_counts: Dict[str, DiscoveredName] = {}
    
    for speaker_name, replacements in all_speaker_replacements.items():
        for rep in replacements:
            original = rep.get('original', '')
            if original and original != '[REDACTED]' and original.strip():
                name_lower = original.lower().strip()
                name_original = original.strip()
                
                if name_lower in name_counts:
                    name_counts[name_lower].occurrences += 1
                else:
                    name_counts[name_lower] = DiscoveredName(
                        name=name_original,
                        source_speaker=speaker_name,
                        occurrences=1
                    )
    
    # Sort by occurrence count (most frequent first)
    return sorted(name_counts.values(), key=lambda x: (-x.occurrences, x.name.lower()))


def second_pass_de_identify(
    segments: List[WordSegment],
    global_names: List[DiscoveredName],
    intermediate_dir: Optional[Path] = None,
    llm_url: str = "http://0.0.0.0:8080",
    speaker_name: Optional[str] = None,
    first_pass_replacements: Optional[List[Dict]] = None,
    **kwargs
) -> SecondPassResult:
    """
    Perform second-pass de-identification using a global name list.
    
    This processes already-redacted transcripts, looking specifically for names
    that were discovered in other speakers' transcripts during the first pass.
    
    Args:
        segments: List of WordSegment objects (already redacted from first pass)
        global_names: List of DiscoveredName objects from all speakers
        intermediate_dir: Path to save audit logs
        llm_url: URL of LLM server
        speaker_name: Speaker name for audit logs
        first_pass_replacements: Replacements from first pass (for combined audit)
        **kwargs: Additional configuration options
        
    Returns:
        SecondPassResult with modified segments and additional replacements
    """
    if not segments:
        return SecondPassResult(segments=segments, additional_replacements=[])
    
    if not global_names:
        log_progress("No names in global list, skipping second pass")
        return SecondPassResult(segments=segments, additional_replacements=[])
    
    log_progress(f"Second-pass de-identification for {len(segments)} segments using {len(global_names)} known names")
    
    # Check if DEBUG logging is enabled
    from local_transcribe.lib.program_logger import get_output_context
    debug_enabled = get_output_context().should_log("DEBUG")
    
    # Setup debug directory
    debug_dir = None
    if debug_enabled and intermediate_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = intermediate_dir / "de_identification" / "second_pass_debug" / timestamp
        debug_dir.mkdir(parents=True, exist_ok=True)
        log_debug(f"Second pass debug mode enabled - saving to {debug_dir}")
    
    # Extract configuration
    chunk_size = kwargs.get('chunk_size', SECOND_PASS_DEFAULTS['chunk_size'])
    overlap_size = kwargs.get('overlap_size', SECOND_PASS_DEFAULTS['overlap_size'])
    min_final_chunk = kwargs.get('min_final_chunk', SECOND_PASS_DEFAULTS['min_final_chunk'])
    
    # Count existing [REDACTED] tokens for validation
    original_redacted_count = sum(1 for seg in segments if seg.text == "[REDACTED]")
    log_debug(f"Original transcript has {original_redacted_count} [REDACTED] tokens")
    
    # Chunk the segments
    chunks = _chunk_segments_for_second_pass(
        segments,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        min_final_chunk=min_final_chunk
    )
    
    log_progress(f"Processing {len(chunks)} chunks in second pass")
    
    # Build the name list for the prompt
    name_list_str = _format_name_list_for_prompt(global_names)
    
    # Process each chunk
    all_additional_replacements = []
    modified_segments = []
    names_found = set()
    session_data = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'speaker': speaker_name,
        'pass': 'second',
        'total_chunks': len(chunks),
        'total_words': len(segments),
        'global_names_count': len(global_names),
        'global_names': [n.name for n in global_names],
        'original_redacted_count': original_redacted_count,
        'chunks_passed': 0,
        'chunks_failed': 0,
        'failed_chunks': [],
        'additional_replacements': 0,
        'config': {
            'chunk_size': chunk_size,
            'overlap_size': overlap_size,
            'min_final_chunk': min_final_chunk,
            'llm_url': llm_url,
            'llm_timeout': kwargs.get('llm_timeout', SECOND_PASS_DEFAULTS['llm_timeout']),
            'temperature': kwargs.get('temperature', SECOND_PASS_DEFAULTS['temperature'])
        }
    }
    
    for idx, chunk in enumerate(chunks):
        chunk_num = idx + 1
        log_progress(f"[{chunk_num}/{len(chunks)}] Second pass on chunk with {len(chunk['segments'])} words")
        
        # Count [REDACTED] in this chunk for validation
        chunk_redacted_count = sum(1 for seg in chunk['segments'] if seg.text == "[REDACTED]")
        
        # Save input chunk for debug
        if debug_dir:
            chunk_data = {
                'text': chunk['text'],
                'start_idx': chunk['start_idx'],
                'end_idx': chunk['end_idx'],
                'redacted_count': chunk_redacted_count,
                'name_list': [n.name for n in global_names]
            }
            _save_second_pass_debug(
                chunk_num,
                chunk_data,
                None,
                None,
                debug_dir,
                mode='input'
            )
        
        # Send to LLM with name list
        processed_text, response_time_ms, validation_result, raw_response = _process_chunk_second_pass(
            chunk['text'],
            name_list_str,
            chunk_redacted_count,
            llm_url=llm_url,
            **kwargs
        )
        
        # Save output for debug - parse Harmony format for readable output
        if debug_dir:
            parse_harmony = kwargs.get('parse_harmony', SECOND_PASS_DEFAULTS['parse_harmony'])
            if parse_harmony and raw_response is not None:
                debug_response = _parse_harmony_response(raw_response)
            else:
                debug_response = raw_response if raw_response is not None else processed_text
            _save_second_pass_debug(
                chunk_num,
                chunk_data,
                debug_response,
                validation_result,
                debug_dir,
                response_time_ms=response_time_ms,
                mode='output',
                original_text=chunk['text']
            )
        
        # Track validation results
        if validation_result and validation_result['passed']:
            session_data['chunks_passed'] += 1
        else:
            session_data['chunks_failed'] += 1
            if validation_result:
                session_data['failed_chunks'].append({
                    'chunk_number': chunk_num,
                    'reason': validation_result['reason']
                })
        
        # Map back to segments
        chunk_modified, chunk_replacements, chunk_names = _map_second_pass_replacements(
            chunk['segments'],
            processed_text,
            global_names,
            start_idx=chunk['start_idx']
        )
        
        modified_segments.extend(chunk_modified)
        all_additional_replacements.extend(chunk_replacements)
        names_found.update(chunk_names)
        
        if chunk_replacements:
            log_progress(f"[{chunk_num}/{len(chunks)}] Found {len(chunk_replacements)} additional names")
    
    # Update session data
    session_data['additional_replacements'] = len(all_additional_replacements)
    session_data['names_from_list_found'] = list(names_found)
    
    # Save session summary
    if debug_dir:
        _save_second_pass_session_summary(debug_dir, session_data)
    
    # Create combined audit log
    if intermediate_dir:
        _create_combined_audit_log(
            first_pass_replacements or [],
            all_additional_replacements,
            intermediate_dir,
            speaker_name=speaker_name,
            total_words=len(segments),
            global_names=global_names
        )
    
    log_progress(f"Second pass complete: {len(all_additional_replacements)} additional names replaced")
    
    return SecondPassResult(
        segments=modified_segments,
        additional_replacements=all_additional_replacements,
        names_from_list_found=names_found
    )


def _format_name_list_for_prompt(names: List[DiscoveredName]) -> str:
    """Format the name list for inclusion in the LLM prompt."""
    if not names:
        return "No names provided."
    
    # Group names and show occurrence counts for context
    name_lines = []
    for name in names:
        if name.occurrences > 1:
            name_lines.append(f"- {name.name} (appeared {name.occurrences} times)")
        else:
            name_lines.append(f"- {name.name}")
    
    return "\n".join(name_lines)


def _chunk_segments_for_second_pass(
    segments: List[WordSegment],
    chunk_size: int,
    overlap_size: int,
    min_final_chunk: int
) -> List[Dict]:
    """Chunk segments for second-pass processing (same logic as first pass)."""
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
        
        remaining = len(segments) - end_idx
        if remaining == 0:
            break
        elif remaining < min_final_chunk:
            chunks[-1]['end_idx'] = len(segments)
            chunks[-1]['segments'] = segments[i:]
            chunks[-1]['text'] = " ".join(seg.text for seg in chunks[-1]['segments'])
            break
        
        i += chunk_size - overlap_size
    
    return chunks


def _parse_harmony_response(raw_response: str) -> str:
    """Parse Harmony-formatted response (same as first pass)."""
    pattern = r'<\|channel\|>(\w+)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)'
    matches = re.findall(pattern, raw_response, re.DOTALL)
    
    for channel, content in matches:
        if channel == 'final':
            return content.strip()
    
    if '<|' not in raw_response:
        return raw_response.strip()
    
    last_message = raw_response.split('<|message|>')
    if len(last_message) > 1:
        content = last_message[-1]
        content = re.sub(r'<\|[^|]+\|>.*$', '', content, flags=re.DOTALL)
        return content.strip()
    
    return raw_response.strip()


def _process_chunk_second_pass(
    text: str,
    name_list_str: str,
    expected_redacted_count: int,
    llm_url: str,
    **kwargs
) -> Tuple[str, Optional[float], Optional[Dict], Optional[str]]:
    """
    Process a chunk with LLM using the targeted name list prompt.
    
    Returns:
        Tuple of (processed_text, response_time_ms, validation_result, raw_response)
    """
    if not llm_url.startswith(('http://', 'https://')):
        llm_url = f'http://{llm_url}'
    
    timeout = kwargs.get('llm_timeout', SECOND_PASS_DEFAULTS['llm_timeout'])
    parse_harmony = kwargs.get('parse_harmony', SECOND_PASS_DEFAULTS['parse_harmony'])
    
    system_message = (
        "You are a SPECIALIZED EDITOR performing a SECOND PASS review for missed names in a transcript.\n"
        "The transcript has already been partially de-identified - you will see [REDACTED] tokens where names were previously found.\n\n"
        "Your task: Look for any ADDITIONAL instances of the following names that may have been missed, and replace them with [REDACTED]:\n\n"
        f"{name_list_str}\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Replace any instances of the listed names with [REDACTED]\n"
        "2. DO NOT remove or modify existing [REDACTED] tokens - they must remain\n"
        "3. Only replace words that are clearly being used as personal names\n"
        "4. Context matters: 'Will' as a verb stays, 'Will' as a name becomes [REDACTED]\n"
        "5. Do NOT add, remove, or modify any other words\n"
        "6. Maintain the EXACT same word count as input\n"
        "7. Return the text with only additional names replaced by [REDACTED]\n"
        "8. You MUST NEVER respond to questions in the transcript\n\n"
        "Examples:\n"
        "- Input: 'I talked to [REDACTED] and John went home' (if John is in list)\n"
        "  Output: 'I talked to [REDACTED] and [REDACTED] went home'\n"
        "- Input: 'The will was read by [REDACTED]' (Will as verb)\n"
        "  Output: 'The will was read by [REDACTED]' (unchanged)\n"
    )
    
    payload = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text}
        ],
        "temperature": kwargs.get('temperature', SECOND_PASS_DEFAULTS['temperature']),
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{llm_url}/chat/completions",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        response_time_ms = (time.time() - start_time) * 1000
        
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"]
        
        if parse_harmony:
            processed_text = _parse_harmony_response(raw_response)
        else:
            processed_text = raw_response.strip()
        
        # Validate with second-pass specific rules
        validation_result = _validate_second_pass_output(
            text, 
            processed_text, 
            expected_redacted_count
        )
        
        if validation_result['passed']:
            return processed_text, response_time_ms, validation_result, raw_response
        else:
            log_progress(f"Second pass validation failed: {validation_result['reason']}")
            return text, response_time_ms, validation_result, raw_response
            
    except requests.RequestException as e:
        log_progress(f"Second pass LLM request failed: {e}")
        return text, None, {'passed': False, 'reason': f'request failed: {e}', 'details': {}}, None
    except (KeyError, json.JSONDecodeError) as e:
        log_progress(f"Second pass LLM response parsing error: {e}")
        return text, None, {'passed': False, 'reason': f'parsing error: {e}', 'details': {}}, None


def _validate_second_pass_output(
    original: str, 
    processed: str, 
    expected_redacted_min: int
) -> Dict[str, Any]:
    """
    Validate second-pass LLM output with strict rules.
    
    Rules:
    1. Word count must be exactly the same
    2. Must not have fewer [REDACTED] tokens than input (can't remove redactions)
    3. Shouldn't have unreasonably many new redactions
    """
    orig_words = original.split()
    proc_words = processed.split()
    
    # Count [REDACTED] tokens
    orig_redacted = orig_words.count("[REDACTED]")
    proc_redacted = proc_words.count("[REDACTED]")
    new_redactions = proc_redacted - orig_redacted
    
    # Rule 1: Word count must match
    if len(orig_words) != len(proc_words):
        return {
            'passed': False,
            'reason': f'word count mismatch (orig: {len(orig_words)}, proc: {len(proc_words)})',
            'details': {
                'original_word_count': len(orig_words),
                'processed_word_count': len(proc_words),
                'original_redacted': orig_redacted,
                'processed_redacted': proc_redacted
            }
        }
    
    # Rule 2: Must not have fewer [REDACTED] tokens (can't remove existing redactions)
    if proc_redacted < orig_redacted:
        return {
            'passed': False,
            'reason': f'removed existing redactions (orig: {orig_redacted}, proc: {proc_redacted})',
            'details': {
                'original_redacted': orig_redacted,
                'processed_redacted': proc_redacted,
                'redactions_removed': orig_redacted - proc_redacted
            }
        }
    
    # Rule 3: Sanity check - not too many new redactions (max 2% of remaining words)
    non_redacted_words = len(orig_words) - orig_redacted
    if non_redacted_words > 0 and new_redactions > non_redacted_words * 0.02:
        return {
            'passed': False,
            'reason': f'too many new redactions ({new_redactions} added, max allowed: {int(non_redacted_words * 0.2)})',
            'details': {
                'original_redacted': orig_redacted,
                'processed_redacted': proc_redacted,
                'new_redactions': new_redactions,
                'max_allowed': int(non_redacted_words * 0.02)
            }
        }
    
    return {
        'passed': True,
        'reason': 'validation passed',
        'details': {
            'original_word_count': len(orig_words),
            'processed_word_count': len(proc_words),
            'original_redacted': orig_redacted,
            'processed_redacted': proc_redacted,
            'new_redactions': new_redactions
        }
    }


def _map_second_pass_replacements(
    original_segments: List[WordSegment],
    llm_output: str,
    global_names: List[DiscoveredName],
    start_idx: int = 0
) -> Tuple[List[WordSegment], List[Dict], Set[str]]:
    """
    Map second-pass replacements back to segments.
    
    Returns:
        (modified_segments, replacements_log, names_found_from_list)
    """
    llm_words = llm_output.split()
    modified_segments = []
    replacements = []
    names_found = set()
    
    # Build lowercase name lookup
    name_lookup = {n.name.lower(): n.name for n in global_names}
    
    llm_idx = 0
    for seg_idx, segment in enumerate(original_segments):
        if llm_idx >= len(llm_words):
            modified_segments.extend(original_segments[seg_idx:])
            break
        
        llm_word = llm_words[llm_idx]
        
        new_segment = WordSegment(
            text=llm_word,
            start=segment.start,
            end=segment.end,
            speaker=segment.speaker
        )
        modified_segments.append(new_segment)
        
        # Track NEW replacements (original wasn't already [REDACTED])
        if llm_word == "[REDACTED]" and segment.text != "[REDACTED]":
            # Try to match against known names
            orig_lower = segment.text.lower().strip('.,!?;:')
            matched_name = name_lookup.get(orig_lower, segment.text)
            names_found.add(matched_name)
            
            replacements.append({
                'timestamp': segment.start,
                'word_index': start_idx + seg_idx,
                'speaker': segment.speaker,
                'original': segment.text,
                'matched_from_list': matched_name,
                'pass': 'second'
            })
        
        llm_idx += 1
    
    return modified_segments, replacements, names_found


def _save_second_pass_debug(
    chunk_idx: int,
    chunk_data: Dict,
    llm_response: Optional[str],
    validation_result: Optional[Dict],
    debug_dir: Path,
    response_time_ms: Optional[float] = None,
    mode: str = 'input',
    original_text: Optional[str] = None
) -> None:
    """Save debug files for second pass.
    
    Args:
        chunk_idx: Chunk number (1-indexed for display)
        chunk_data: Dict with 'text', 'start_idx', 'end_idx', etc.
        llm_response: LLM response text (for output mode) - should be parsed if using Harmony format
        validation_result: Dict with validation info (for output mode)
        debug_dir: Directory to save debug files
        response_time_ms: Response time in milliseconds
        mode: 'input' or 'output'
        original_text: Original input text for diff generation (optional, uses chunk_data['text'] if not provided)
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    chunk_num_str = f"{chunk_idx:03d}"
    
    if mode == 'input':
        json_data = {
            'chunk_number': chunk_idx,
            'pass': 'second',
            'word_count': len(chunk_data['text'].split()),
            'redacted_count': chunk_data.get('redacted_count', 0),
            'name_list': chunk_data.get('name_list', []),
            'text': chunk_data['text']
        }
        
        json_path = debug_dir / f"chunk_{chunk_num_str}_input.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        txt_path = debug_dir / f"chunk_{chunk_num_str}_input.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"CHUNK {chunk_idx} - SECOND PASS INPUT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Word count: {len(chunk_data['text'].split())}\n")
            f.write(f"Existing [REDACTED] count: {chunk_data.get('redacted_count', 0)}\n")
            f.write(f"Names to look for: {len(chunk_data.get('name_list', []))}\n")
            f.write("-" * 60 + "\n\n")
            f.write(chunk_data['text'])
            f.write("\n")
    
    elif mode == 'output' and llm_response is not None:
        json_data = {
            'chunk_number': chunk_idx,
            'pass': 'second',
            'response_time_ms': response_time_ms,
            'input_word_count': len(chunk_data['text'].split()),
            'output_word_count': len(llm_response.split()),
            'text': llm_response
        }
        
        if validation_result:
            json_data['validation_passed'] = validation_result.get('passed', False)
            json_data['validation_reason'] = validation_result.get('reason', '')
            json_data['validation_details'] = validation_result.get('details', {})
        
        json_path = debug_dir / f"chunk_{chunk_num_str}_output.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        txt_path = debug_dir / f"chunk_{chunk_num_str}_output.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"CHUNK {chunk_idx} - SECOND PASS OUTPUT\n")
            f.write("=" * 60 + "\n")
            if response_time_ms:
                f.write(f"Response time: {response_time_ms/1000:.2f}s\n")
            if validation_result:
                status = "PASSED" if validation_result.get('passed', False) else "FAILED"
                f.write(f"Validation: {status}")
                if not validation_result.get('passed', False):
                    f.write(f" - {validation_result.get('reason', '')}")
                f.write("\n")
            f.write("-" * 60 + "\n\n")
            f.write(llm_response)
            f.write("\n")
        
        # Generate diff if validation failed
        if validation_result and not validation_result.get('passed', False):
            # Use original_text if provided, otherwise fall back to chunk_data['text']
            diff_original = original_text if original_text is not None else chunk_data['text']
            _generate_word_diff(
                chunk_idx,
                diff_original,
                llm_response,
                validation_result,
                debug_dir
            )


def _generate_word_diff(
    chunk_idx: int,
    original_text: str,
    llm_text: str,
    validation_result: Dict,
    debug_dir: Path
) -> None:
    """
    Generate word-by-word diff for failed validations.
    
    Args:
        chunk_idx: Chunk number
        original_text: Original text sent to LLM
        llm_text: LLM response text (should be parsed, not raw Harmony format)
        validation_result: Validation result dict
        debug_dir: Directory to save diff file
    """
    orig_words = original_text.split()
    llm_words = llm_text.split()
    
    chunk_num_str = f"{chunk_idx:03d}"
    diff_path = debug_dir / f"chunk_{chunk_num_str}_diff.txt"
    
    with open(diff_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"CHUNK {chunk_idx} - WORD-BY-WORD DIFF (SECOND PASS)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Validation failed: {validation_result.get('reason', 'unknown')}\n\n")
        
        # Summary
        missing = max(0, len(orig_words) - len(llm_words))
        extra = max(0, len(llm_words) - len(orig_words))
        f.write(f"Missing words: {missing}\n")
        f.write(f"Extra words: {extra}\n\n")
        
        # Word-by-word comparison
        f.write("Word-by-word comparison:\n")
        f.write("-" * 60 + "\n")
        
        max_len = max(len(orig_words), len(llm_words))
        differences = []
        
        for i in range(max_len):
            orig_word = orig_words[i] if i < len(orig_words) else None
            llm_word = llm_words[i] if i < len(llm_words) else None
            
            if orig_word is None:
                f.write(f"[{i:03d}] ✗ EXTRA: \"{llm_word}\"\n")
                differences.append((i, 'EXTRA', None, llm_word))
            elif llm_word is None:
                f.write(f"[{i:03d}] ✗ MISSING: \"{orig_word}\"\n")
                differences.append((i, 'MISSING', orig_word, None))
            elif orig_word != llm_word:
                f.write(f"[{i:03d}] ✗ CHANGED: \"{orig_word}\" → \"{llm_word}\"\n")
                differences.append((i, 'CHANGED', orig_word, llm_word))
            else:
                # Only show first/last few matches to keep file readable
                if i < 5 or i >= max_len - 5:
                    f.write(f"[{i:03d}] ✓ {orig_word} → {llm_word}\n")
                elif i == 5:
                    f.write(f"... ({max_len - 10} matching words omitted) ...\n")
        
        # Show context around differences
        if differences:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Context around differences:\n")
            f.write("-" * 60 + "\n")
            
            for diff_idx, diff_type, orig, llm in differences[:10]:  # Show first 10 diffs
                context_start = max(0, diff_idx - 3)
                context_end = min(len(orig_words), diff_idx + 4)
                
                f.write(f"\nPosition {diff_idx} ({diff_type}):\n")
                f.write(f"  Original: \"{' '.join(orig_words[context_start:context_end])}\"\n")
                
                llm_context_end = min(len(llm_words), context_start + (context_end - context_start))
                f.write(f"  LLM:      \"{' '.join(llm_words[context_start:llm_context_end])}\"\n")
            
            if len(differences) > 10:
                f.write(f"\n... and {len(differences) - 10} more differences\n")


def _save_second_pass_session_summary(debug_dir: Path, session_data: Dict) -> None:
    """Save second pass session summary."""
    json_path = debug_dir / "session_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    txt_path = debug_dir / "debug_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("SECOND PASS DE-IDENTIFICATION DEBUG SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {session_data.get('timestamp', '')}\n")
        if session_data.get('speaker'):
            f.write(f"Speaker: {session_data['speaker']}\n")
        f.write(f"Total chunks: {session_data.get('total_chunks', 0)}\n")
        f.write(f"Total words: {session_data.get('total_words', 0):,}\n")
        f.write(f"Global names searched: {session_data.get('global_names_count', 0)}\n\n")
        
        f.write("Names searched for:\n")
        for name in session_data.get('global_names', []):
            f.write(f"  - {name}\n")
        
        f.write("\nResults:\n")
        passed = session_data.get('chunks_passed', 0)
        failed = session_data.get('chunks_failed', 0)
        total = session_data.get('total_chunks', 0)
        if total > 0:
            pass_rate = (passed / total) * 100
            f.write(f"- Chunks passed: {passed}/{total} ({pass_rate:.1f}%)\n")
        f.write(f"- Additional names replaced: {session_data.get('additional_replacements', 0)}\n")
        
        names_found = session_data.get('names_from_list_found', [])
        if names_found:
            f.write(f"- Names from list that were found: {', '.join(names_found)}\n")
        
        f.write("=" * 60 + "\n")


def _create_combined_audit_log(
    first_pass_replacements: List[Dict],
    second_pass_replacements: List[Dict],
    intermediate_dir: Path,
    speaker_name: Optional[str] = None,
    total_words: int = 0,
    global_names: Optional[List[DiscoveredName]] = None
) -> None:
    """
    Create combined audit log with both first and second pass replacements.
    """
    de_id_dir = intermediate_dir / "de_identification"
    de_id_dir.mkdir(parents=True, exist_ok=True)
    
    if speaker_name:
        filename = f"{speaker_name.lower()}_combined_audit_log.txt"
    else:
        filename = "combined_audit_log.txt"
    
    audit_path = de_id_dir / filename
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    total_replacements = len(first_pass_replacements) + len(second_pass_replacements)
    
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write("Combined De-Identification Audit Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {timestamp}\n")
        if speaker_name:
            f.write(f"Speaker: {speaker_name}\n")
        f.write(f"Total words processed: {total_words:,}\n")
        f.write(f"Total names replaced: {total_replacements}\n")
        f.write(f"  - First pass: {len(first_pass_replacements)}\n")
        f.write(f"  - Second pass: {len(second_pass_replacements)}\n")
        f.write("=" * 60 + "\n\n")
        
        # First pass section
        f.write("FIRST PASS REPLACEMENTS (General De-identification)\n")
        f.write("-" * 60 + "\n")
        if not first_pass_replacements:
            f.write("No names replaced in first pass.\n")
        else:
            for rep in first_pass_replacements:
                ts = rep.get('timestamp')
                if ts is not None:
                    hours = int(ts // 3600)
                    minutes = int((ts % 3600) // 60)
                    seconds = ts % 60
                    ts_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                    original = rep.get('original', 'unknown')
                    f.write(f"[{ts_str}] \"{original}\" → [REDACTED]\n")
                else:
                    original = rep.get('original', 'unknown')
                    word_idx = rep.get('word_index', '?')
                    f.write(f"[word {word_idx}] \"{original}\" → [REDACTED]\n")
        
        f.write("\n")
        
        # Second pass section
        f.write("SECOND PASS REPLACEMENTS (Targeted Name Review)\n")
        f.write("-" * 60 + "\n")
        if global_names:
            f.write(f"Names searched for: {', '.join(n.name for n in global_names)}\n\n")
        
        if not second_pass_replacements:
            f.write("No additional names found in second pass.\n")
        else:
            for rep in second_pass_replacements:
                ts = rep.get('timestamp')
                if ts is not None:
                    hours = int(ts // 3600)
                    minutes = int((ts % 3600) // 60)
                    seconds = ts % 60
                    ts_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                    original = rep.get('original', 'unknown')
                    matched = rep.get('matched_from_list', original)
                    f.write(f"[{ts_str}] \"{original}\" → [REDACTED] (matched: {matched})\n")
                else:
                    original = rep.get('original', 'unknown')
                    word_idx = rep.get('word_index', '?')
                    matched = rep.get('matched_from_list', original)
                    f.write(f"[word {word_idx}] \"{original}\" → [REDACTED] (matched: {matched})\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 60 + "\n")
        
        # Unique names
        all_names = set()
        for rep in first_pass_replacements:
            all_names.add(rep.get('original', '').lower())
        for rep in second_pass_replacements:
            all_names.add(rep.get('original', '').lower())
        all_names.discard('')
        
        f.write(f"Unique names redacted: {len(all_names)}\n")
        if all_names:
            f.write("Names: " + ", ".join(sorted(all_names)) + "\n")
        
        if total_words > 0:
            rate = (total_replacements / total_words) * 100
            f.write(f"Replacement rate: {rate:.2f}%\n")
        
        f.write("=" * 60 + "\n")
    
    log_progress(f"Combined audit log saved to {audit_path}")
