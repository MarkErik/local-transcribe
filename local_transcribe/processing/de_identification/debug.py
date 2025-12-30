#!/usr/bin/env python3
"""
Debug file generation for de-identification.

Creates detailed debug files for troubleshooting LLM de-identification,
including input/output dumps, validation results, and word diffs.
"""

import json
import difflib
from typing import Dict, Optional, Any, List
from pathlib import Path

from .core import Chunk, ChunkProcessingResult, format_timestamp


class DebugFileWriter:
    """Writes debug files for de-identification operations."""
    
    def __init__(self, debug_dir: Path):
        """
        Initialize the debug file writer.
        
        Args:
            debug_dir: Directory to save debug files
        """
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def save_chunk_input(
        self,
        chunk_num: int,
        chunk: Chunk,
        pass_name: str = 'first',
        extra_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save input chunk for debugging.
        
        Args:
            chunk_num: Chunk number (1-indexed)
            chunk: The chunk being processed
            pass_name: 'first' or 'second'
            extra_data: Additional data to include
        """
        chunk_str = f"{chunk_num:03d}"
        
        # Build JSON data
        json_data = {
            'chunk_number': chunk_num,
            'pass': pass_name,
            'word_count': len(chunk.text.split()),
            'start_idx': chunk.start_idx,
            'end_idx': chunk.end_idx,
            'text': chunk.text
        }
        
        # Add segments if available
        if chunk.segments:
            json_data['segments'] = [
                {
                    'word': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'speaker': seg.speaker
                }
                for seg in chunk.segments
            ]
            if chunk.segments:
                json_data['timestamp_range'] = {
                    'start': chunk.segments[0].start,
                    'end': chunk.segments[-1].end
                }
        
        if extra_data:
            json_data.update(extra_data)
        
        # Save JSON
        json_path = self.debug_dir / f"chunk_{chunk_str}_{pass_name}_input.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save text
        txt_path = self.debug_dir / f"chunk_{chunk_str}_{pass_name}_input.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"CHUNK {chunk_num} - {pass_name.upper()} PASS INPUT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Word count: {len(chunk.text.split())}\n")
            f.write(f"Index range: {chunk.start_idx}-{chunk.end_idx}\n")
            
            if chunk.segments:
                start_str = format_timestamp(chunk.segments[0].start)
                end_str = format_timestamp(chunk.segments[-1].end)
                f.write(f"Timestamp range: {start_str} - {end_str}\n")
            
            if extra_data and 'redacted_count' in extra_data:
                f.write(f"Existing [REDACTED]: {extra_data['redacted_count']}\n")
            
            f.write("-" * 60 + "\n\n")
            f.write(chunk.text)
            f.write("\n")
    
    def save_chunk_output(
        self,
        chunk_num: int,
        chunk: Chunk,
        result: ChunkProcessingResult,
        pass_name: str = 'first'
    ) -> None:
        """
        Save output chunk and validation results for debugging.
        
        Args:
            chunk_num: Chunk number (1-indexed)
            chunk: The original chunk
            result: Processing result from LLM
            pass_name: 'first' or 'second'
        """
        chunk_str = f"{chunk_num:03d}"
        
        # Save each attempt
        for attempt in result.attempt_logs:
            attempt_num = attempt.get('attempt', 1)
            attempt_suffix = f"_attempt{attempt_num}" if len(result.attempt_logs) > 1 else ""
            
            processed_text = attempt.get('processed_text', result.processed_text)
            raw_response = attempt.get('raw_response')
            validation = attempt.get('validation_result', {})
            response_time = attempt.get('response_time_ms', result.response_time_ms)
            
            # Build JSON data
            json_data = {
                'chunk_number': chunk_num,
                'pass': pass_name,
                'attempt': attempt_num,
                'response_time_ms': response_time,
                'input_word_count': len(chunk.text.split()),
                'output_word_count': len(processed_text.split()) if processed_text else 0,
                'validation_passed': validation.get('passed', False),
                'validation_reason': validation.get('reason', ''),
                'validation_details': validation.get('details', {}),
                'text': processed_text
            }
            
            # Save JSON
            json_path = self.debug_dir / f"chunk_{chunk_str}_{pass_name}{attempt_suffix}_output.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Save text
            txt_path = self.debug_dir / f"chunk_{chunk_str}_{pass_name}{attempt_suffix}_output.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                title_suffix = f" (attempt {attempt_num})" if len(result.attempt_logs) > 1 else ""
                f.write(f"CHUNK {chunk_num} - {pass_name.upper()} PASS OUTPUT{title_suffix}\n")
                f.write("=" * 60 + "\n")
                if response_time:
                    f.write(f"Response time: {response_time/1000:.2f}s\n")
                
                status = "PASSED" if validation.get('passed', False) else "FAILED"
                f.write(f"Validation: {status}")
                if not validation.get('passed', False):
                    f.write(f" - {validation.get('reason', '')}")
                f.write("\n")
                
                f.write(f"Input word count: {len(chunk.text.split())}\n")
                f.write(f"Output word count: {len(processed_text.split()) if processed_text else 0}\n")
                f.write("-" * 60 + "\n\n")
                f.write(processed_text or "")
                f.write("\n")
            
            # Generate diff if validation failed
            if not validation.get('passed', False) and processed_text:
                self._generate_word_diff(
                    chunk_num,
                    chunk.text,
                    processed_text,
                    validation,
                    pass_name,
                    attempt_suffix
                )
    
    def _generate_word_diff(
        self,
        chunk_num: int,
        original_text: str,
        llm_text: str,
        validation: Dict[str, Any],
        pass_name: str,
        attempt_suffix: str = ""
    ) -> None:
        """
        Generate word-by-word diff for failed validations.
        
        Uses difflib.SequenceMatcher to detect insertions, deletions,
        and substitutions.
        """
        orig_words = original_text.split()
        llm_words = llm_text.split()
        
        chunk_str = f"{chunk_num:03d}"
        diff_path = self.debug_dir / f"chunk_{chunk_str}_{pass_name}{attempt_suffix}_diff.txt"
        
        matcher = difflib.SequenceMatcher(None, orig_words, llm_words)
        opcodes = matcher.get_opcodes()
        
        with open(diff_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"CHUNK {chunk_num} - SEQUENCE-ALIGNED DIFF ({pass_name.upper()} PASS)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Validation failed: {validation.get('reason', 'unknown')}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Original word count: {len(orig_words)}\n")
            f.write(f"LLM output word count: {len(llm_words)}\n")
            f.write(f"Difference: {len(llm_words) - len(orig_words):+d} words\n\n")
            
            # Categorize operations
            insertions = []
            deletions = []
            replacements = []
            valid_redactions = []
            
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'insert':
                    insertions.append((j1, j2, llm_words[j1:j2]))
                elif tag == 'delete':
                    deletions.append((i1, i2, orig_words[i1:i2]))
                elif tag == 'replace':
                    orig_slice = orig_words[i1:i2]
                    llm_slice = llm_words[j1:j2]
                    
                    if len(orig_slice) == len(llm_slice):
                        for idx, (o, l) in enumerate(zip(orig_slice, llm_slice)):
                            if l == '[REDACTED]' and o != '[REDACTED]':
                                valid_redactions.append((i1 + idx, o))
                            elif o != l:
                                replacements.append((i1 + idx, o, l))
                    else:
                        replacements.append((i1, orig_slice, llm_slice))
            
            f.write(f"Valid name redactions: {len(valid_redactions)}\n")
            f.write(f"Insertions (LLM added words): {len(insertions)}\n")
            f.write(f"Deletions (LLM removed words): {len(deletions)}\n")
            f.write(f"Replacements/Changes: {len(replacements)}\n\n")
            
            # Show valid redactions
            if valid_redactions:
                f.write("=" * 70 + "\n")
                f.write("VALID NAME REDACTIONS (Expected behavior)\n")
                f.write("-" * 70 + "\n")
                for pos, orig_word in valid_redactions:
                    context_start = max(0, pos - 2)
                    context_end = min(len(orig_words), pos + 3)
                    context = orig_words[context_start:context_end]
                    rel_pos = pos - context_start
                    context_display = context.copy()
                    context_display[rel_pos] = f">>>{context_display[rel_pos]}<<<"
                    f.write(f"[{pos:03d}] \"{orig_word}\" → [REDACTED]\n")
                    f.write(f"       Context: {' '.join(context_display)}\n")
                f.write("\n")
            
            # Show problems
            if deletions:
                f.write("=" * 70 + "\n")
                f.write("⚠️  DELETIONS (LLM removed these words - PROBLEMATIC)\n")
                f.write("-" * 70 + "\n")
                for i1, i2, deleted_words in deletions:
                    f.write(f"[{i1:03d}-{i2-1:03d}] DELETED {len(deleted_words)} word(s): \"{' '.join(deleted_words)}\"\n")
                    context_start = max(0, i1 - 3)
                    context_end = min(len(orig_words), i2 + 3)
                    f.write(f"           Original context: \"{' '.join(orig_words[context_start:context_end])}\"\n\n")
            
            if insertions:
                f.write("=" * 70 + "\n")
                f.write("⚠️  INSERTIONS (LLM added these words - PROBLEMATIC)\n")
                f.write("-" * 70 + "\n")
                for j1, j2, inserted_words in insertions:
                    f.write(f"[LLM pos {j1:03d}-{j2-1:03d}] INSERTED {len(inserted_words)} word(s): \"{' '.join(inserted_words)}\"\n")
                    context_start = max(0, j1 - 3)
                    context_end = min(len(llm_words), j2 + 3)
                    f.write(f"           LLM context: \"{' '.join(llm_words[context_start:context_end])}\"\n\n")
            
            if replacements:
                f.write("=" * 70 + "\n")
                f.write("⚠️  REPLACEMENTS/CHANGES (Not simple redactions)\n")
                f.write("-" * 70 + "\n")
                for item in replacements[:20]:
                    if len(item) == 3 and isinstance(item[1], str):
                        pos, orig_word, llm_word = item
                        f.write(f"[{pos:03d}] \"{orig_word}\" → \"{llm_word}\"\n")
                    else:
                        pos, orig_slice, llm_slice = item
                        if isinstance(orig_slice, list):
                            f.write(f"[{pos:03d}] LENGTH MISMATCH:\n")
                            f.write(f"        Original ({len(orig_slice)} words): \"{' '.join(orig_slice)}\"\n")
                            f.write(f"        LLM ({len(llm_slice)} words):      \"{' '.join(llm_slice)}\"\n\n")
                if len(replacements) > 20:
                    f.write(f"\n... and {len(replacements) - 20} more replacements\n")
            
            # Full operation log
            f.write("\n" + "=" * 70 + "\n")
            f.write("FULL DIFF OPERATIONS (for detailed analysis)\n")
            f.write("-" * 70 + "\n")
            f.write("Legend: 'equal'=unchanged, 'replace'=modified, 'delete'=removed, 'insert'=added\n\n")
            
            for tag, i1, i2, j1, j2 in opcodes:
                if tag == 'equal':
                    word_count = i2 - i1
                    if word_count <= 6:
                        f.write(f"[EQUAL]  orig[{i1}:{i2}] = llm[{j1}:{j2}]: \"{' '.join(orig_words[i1:i2])}\"\n")
                    else:
                        preview = ' '.join(orig_words[i1:i1+3]) + ' ... ' + ' '.join(orig_words[i2-2:i2])
                        f.write(f"[EQUAL]  orig[{i1}:{i2}] = llm[{j1}:{j2}]: ({word_count} words) \"{preview}\"\n")
                elif tag == 'replace':
                    f.write(f"[REPLACE] orig[{i1}:{i2}] → llm[{j1}:{j2}]:\n")
                    f.write(f"          - \"{' '.join(orig_words[i1:i2])}\"\n")
                    f.write(f"          + \"{' '.join(llm_words[j1:j2])}\"\n")
                elif tag == 'delete':
                    f.write(f"[DELETE] orig[{i1}:{i2}]: \"{' '.join(orig_words[i1:i2])}\"\n")
                elif tag == 'insert':
                    f.write(f"[INSERT] llm[{j1}:{j2}]: \"{' '.join(llm_words[j1:j2])}\"\n")
            
            f.write("\n" + "=" * 70 + "\n")
    
    def save_session_summary(
        self,
        session_data: Dict[str, Any],
        pass_name: str = 'combined'
    ) -> None:
        """
        Save session summary for debugging.
        
        Args:
            session_data: Session statistics and metadata
            pass_name: Name for the summary file
        """
        # JSON summary
        json_path = self.debug_dir / f"{pass_name}_session_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Text summary
        txt_path = self.debug_dir / f"{pass_name}_debug_summary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"DE-IDENTIFICATION DEBUG SUMMARY ({pass_name.upper()})\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {session_data.get('timestamp', '')}\n")
            if session_data.get('speaker'):
                f.write(f"Speaker: {session_data['speaker']}\n")
            f.write(f"Total chunks: {session_data.get('total_chunks', 0)}\n")
            f.write(f"Total words: {session_data.get('total_words', 0):,}\n\n")
            
            f.write("Configuration:\n")
            config = session_data.get('config', {})
            for key, value in config.items():
                f.write(f"- {key}: {value}\n")
            
            f.write("\nResults:\n")
            passed = session_data.get('chunks_passed', 0)
            failed = session_data.get('chunks_failed', 0)
            total = session_data.get('total_chunks', 0)
            if total > 0:
                pass_rate = (passed / total) * 100
                f.write(f"- Chunks passed: {passed}/{total} ({pass_rate:.1f}%)\n")
                f.write(f"- Chunks failed: {failed}/{total} ({100-pass_rate:.1f}%)\n")
            f.write(f"- Total replacements: {session_data.get('total_replacements', 0)}\n")
            
            failed_chunks = session_data.get('failed_chunks', [])
            if failed_chunks:
                f.write("\nFailed Chunks:\n")
                for chunk_info in failed_chunks:
                    chunk_num = chunk_info.get('chunk_number', 0)
                    reason = chunk_info.get('reason', 'unknown')
                    f.write(f"- Chunk {chunk_num}: {reason}\n")
            
            f.write("\n" + "=" * 60 + "\n")
