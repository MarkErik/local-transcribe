#!/usr/bin/env python3
"""
Chunk stitcher for intelligently stitching overlapping transcript chunks from audio transcription.

This module provides functionality to stitch overlapping chunks from transcriber output,
handling cases where words may be cut off at chunk boundaries (e.g. "generational" -> "rational")
and dealing with slight differences in similar-sounding words.

Supports both:
- Simple string words: chunks with "words" as List[str]
- Timestamped words: chunks with "words" as List[Dict] with "text", "start", "end" keys
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from local_transcribe.lib.program_logger import log_progress, log_debug, get_output_context
from difflib import SequenceMatcher
from local_transcribe.framework.plugin_interfaces import WordSegment


class ChunkStitcher:
    """
    A class for stitching overlapping transcript chunks with intelligent overlap detection.
    """
    
    def __init__(self, min_overlap_ratio: float = 0.6, similarity_threshold: float = 0.7,
                 intermediate_dir: Optional[Path] = None):
        """
        Initialize the chunk stitcher with configurable thresholds.
        
        Args:
            min_overlap_ratio: Minimum ratio of overlapping words to consider a valid overlap
            similarity_threshold: Threshold for word similarity using SequenceMatcher
            intermediate_dir: Optional path for saving debug files when DEBUG logging is enabled
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.similarity_threshold = similarity_threshold
        self.intermediate_dir = intermediate_dir
        
        # Setup debug directory if DEBUG logging is enabled
        self.debug_dir = None
        debug_enabled = get_output_context().should_log("DEBUG")
        if debug_enabled and intermediate_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = Path(intermediate_dir) / "chunk_stitching" / "local_stitcher_debug" / timestamp
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Debug mode enabled - saving debug files to {self.debug_dir}")
    
    def _save_debug_input(self, chunks: List[Dict[str, Any]], has_timestamps: bool) -> None:
        """Save input chunks for debugging."""
        if not self.debug_dir:
            return
        
        # Save JSON file
        json_data = {
            'total_chunks': len(chunks),
            'has_timestamps': has_timestamps,
            'chunks': []
        }
        
        for chunk in chunks:
            chunk_data = {
                'chunk_id': chunk.get('chunk_id'),
                'word_count': len(chunk['words'])
            }
            if has_timestamps and chunk['words']:
                chunk_data['words'] = chunk['words']
                first_word = chunk['words'][0]
                last_word = chunk['words'][-1]
                chunk_data['time_range'] = {
                    'start': first_word.get('start', 0),
                    'end': last_word.get('end', 0)
                }
            else:
                chunk_data['words'] = chunk['words']
            json_data['chunks'].append(chunk_data)
        
        json_path = self.debug_dir / "00_input_chunks.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable text file
        txt_path = self.debug_dir / "00_input_chunks.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("INPUT CHUNKS FOR STITCHING\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Format: {'timestamped' if has_timestamps else 'string words'}\n")
            f.write("-" * 60 + "\n\n")
            
            for chunk in chunks:
                chunk_id = chunk.get('chunk_id', '?')
                words = chunk['words']
                f.write(f"CHUNK {chunk_id} ({len(words)} words)\n")
                if has_timestamps and words:
                    first_word = words[0]
                    last_word = words[-1]
                    f.write(f"Time range: {first_word.get('start', 0):.2f}s - {last_word.get('end', 0):.2f}s\n")
                    f.write("-" * 40 + "\n")
                    for word in words:
                        f.write(f"[{word.get('start', 0):.2f}-{word.get('end', 0):.2f}] {word.get('text', '')}\n")
                else:
                    f.write("-" * 40 + "\n")
                    f.write(" ".join(words) + "\n")
                f.write("\n")
    
    def _save_debug_stitch_step(self, step_num: int, chunk1_info: Dict, chunk2_info: Dict, 
                                 overlap_info: Dict, result_info: Dict) -> None:
        """Save debug info for a single stitching step."""
        if not self.debug_dir:
            return
        
        step_str = f"{step_num:03d}"
        
        # Save JSON file
        json_data = {
            'step': step_num,
            'chunk1': chunk1_info,
            'chunk2': chunk2_info,
            'overlap_detection': overlap_info,
            'result': result_info
        }
        
        json_path = self.debug_dir / f"step_{step_str}_stitch.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable text file
        txt_path = self.debug_dir / f"step_{step_str}_stitch.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"STITCH STEP {step_num}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("CHUNK 1 (accumulated):\n")
            f.write(f"  Word count: {chunk1_info['word_count']}\n")
            if 'last_10_words' in chunk1_info:
                f.write(f"  Last 10 words: {chunk1_info['last_10_words']}\n")
            f.write("\n")
            
            f.write("CHUNK 2 (incoming):\n")
            f.write(f"  Chunk ID: {chunk2_info.get('chunk_id', '?')}\n")
            f.write(f"  Word count: {chunk2_info['word_count']}\n")
            if 'first_10_words' in chunk2_info:
                f.write(f"  First 10 words: {chunk2_info['first_10_words']}\n")
            f.write("\n")
            
            f.write("OVERLAP DETECTION:\n")
            f.write(f"  Overlap found: {overlap_info['overlap_found']}\n")
            if overlap_info['overlap_found']:
                f.write(f"  Overlap start (in chunk1): {overlap_info['overlap_start']}\n")
                f.write(f"  Overlap length: {overlap_info['overlap_length']}\n")
                f.write(f"  Overlapping words: {overlap_info.get('overlapping_words', [])}\n")
                if overlap_info.get('fuzzy_matches'):
                    f.write(f"  Fuzzy matches: {overlap_info['fuzzy_matches']}\n")
            f.write("\n")
            
            f.write("RESULT:\n")
            f.write(f"  Total words after stitch: {result_info['word_count']}\n")
            f.write(f"  Words taken from chunk1: {result_info.get('words_from_chunk1', '?')}\n")
            f.write(f"  Words taken from chunk2: {result_info.get('words_from_chunk2', '?')}\n")
    
    def _save_debug_output(self, result: Union[str, List[WordSegment]], has_timestamps: bool) -> None:
        """Save final stitched output for debugging."""
        if not self.debug_dir:
            return
        
        if has_timestamps:
            # Handle List[WordSegment] or List[Dict]
            if result and isinstance(result[0], WordSegment):
                words_list = [{'text': w.text, 'start': w.start, 'end': w.end, 'speaker': w.speaker} for w in result]
            else:
                words_list = result
            
            json_data = {
                'total_words': len(words_list),
                'has_timestamps': True,
                'words': words_list
            }
            
            json_path = self.debug_dir / "99_final_output.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            txt_path = self.debug_dir / "99_final_output.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("FINAL STITCHED OUTPUT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total words: {len(words_list)}\n")
                if words_list:
                    f.write(f"Time range: {words_list[0].get('start', 0):.2f}s - {words_list[-1].get('end', 0):.2f}s\n")
                f.write("-" * 60 + "\n\n")
                for word in words_list:
                    f.write(f"[{word.get('start', 0):.2f}-{word.get('end', 0):.2f}] {word.get('text', '')}\n")
        else:
            # Handle string result
            words = result.split() if isinstance(result, str) else result
            
            json_data = {
                'total_words': len(words),
                'has_timestamps': False,
                'text': result if isinstance(result, str) else " ".join(words)
            }
            
            json_path = self.debug_dir / "99_final_output.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            txt_path = self.debug_dir / "99_final_output.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("FINAL STITCHED OUTPUT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total words: {len(words)}\n")
                f.write("-" * 60 + "\n\n")
                f.write(result if isinstance(result, str) else " ".join(words))
                f.write("\n")
    
    def _save_debug_session_summary(self, chunks: List[Dict[str, Any]], 
                                     has_timestamps: bool,
                                     total_stitch_steps: int,
                                     final_word_count: int) -> None:
        """Save session summary for debugging."""
        if not self.debug_dir:
            return
        
        summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'min_overlap_ratio': self.min_overlap_ratio,
                'similarity_threshold': self.similarity_threshold
            },
            'input': {
                'total_chunks': len(chunks),
                'has_timestamps': has_timestamps,
                'total_input_words': sum(len(c['words']) for c in chunks)
            },
            'processing': {
                'stitch_steps': total_stitch_steps
            },
            'output': {
                'final_word_count': final_word_count
            }
        }
        
        json_path = self.debug_dir / "session_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        txt_path = self.debug_dir / "session_summary.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CHUNK STITCHING SESSION SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Min overlap ratio: {self.min_overlap_ratio}\n")
            f.write(f"  Similarity threshold: {self.similarity_threshold}\n")
            f.write(f"\nInput:\n")
            f.write(f"  Total chunks: {len(chunks)}\n")
            f.write(f"  Format: {'timestamped' if has_timestamps else 'string words'}\n")
            f.write(f"  Total input words: {summary['input']['total_input_words']}\n")
            f.write(f"\nProcessing:\n")
            f.write(f"  Stitch steps: {total_stitch_steps}\n")
            f.write(f"\nOutput:\n")
            f.write(f"  Final word count: {final_word_count}\n")
    
    def stitch_chunks(self, chunks: List[Dict[str, Any]]) -> Union[str, List[WordSegment]]:
        """
        Stitch a list of transcript chunks into a single transcript.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'words' keys
                   Words can be either List[str] or List[Dict] with timestamps
            
        Returns:
            If words are strings: Stitched transcript as a string
            If words have timestamps: List[WordSegment] with preserved timestamps
        """
        if not chunks:
            return ""
        
        # Detect format: check if words have timestamps
        # Check all chunks in case the first chunk has empty words
        has_timestamps = False
        for chunk in chunks:
            if chunk["words"]:
                first_word = chunk["words"][0]
                has_timestamps = (
                    isinstance(first_word, dict) and
                    "text" in first_word and
                    "start" in first_word
                )
                break  # Found a non-empty chunk, use its format
        
        # Save debug input
        self._save_debug_input(chunks, has_timestamps)
        
        if has_timestamps:
            return self._stitch_chunks_with_timestamps(chunks)
        else:
            return self._stitch_chunks_simple(chunks)
    
    def _stitch_chunks_simple(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Stitch chunks with simple string words (existing behavior).
        
        Args:
            chunks: List of chunk dictionaries with 'words' as List[str]
            
        Returns:
            Stitched transcript as a string
        """
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            result = " ".join(chunks[0]["words"])
            self._save_debug_output(result, has_timestamps=False)
            self._save_debug_session_summary(chunks, has_timestamps=False, 
                                            total_stitch_steps=0, 
                                            final_word_count=len(chunks[0]["words"]))
            return result
        
        # Start with the first chunk
        stitched_words = chunks[0]["words"].copy()
        
        log_progress(f"Processing {len(chunks)} chunks (string words)")
        
        # Iteratively stitch each subsequent chunk
        for i in range(1, len(chunks)):
            log_progress(f"Processing chunk {i + 1} of {len(chunks)}")
            current_chunk_words = chunks[i]["words"]
            chunk_id = chunks[i].get('chunk_id', i + 1)
            stitched_words = self._stitch_two_chunks_with_debug(
                stitched_words, current_chunk_words, 
                step_num=i, chunk_id=chunk_id, has_timestamps=False
            )
        
        log_progress(f"Stitch complete: {len(stitched_words)} words total")
        
        result = " ".join(stitched_words)
        self._save_debug_output(result, has_timestamps=False)
        self._save_debug_session_summary(chunks, has_timestamps=False,
                                        total_stitch_steps=len(chunks) - 1,
                                        final_word_count=len(stitched_words))
        
        return result
    
    def _stitch_chunks_with_timestamps(self, chunks: List[Dict[str, Any]]) -> List[WordSegment]:
        """
        Stitch chunks with timestamped words, preserving timing information.
        
        Args:
            chunks: List of chunk dictionaries with 'words' as List[Dict]
                   Each dict has "text", "start", "end"
            
        Returns:
            List[WordSegment] with preserved timestamps
        """
        if not chunks:
            return []
        
        if len(chunks) == 1:
            # Convert to WordSegments
            result = [
                WordSegment(text=w["text"], start=w["start"], end=w["end"], speaker=w.get("speaker"))
                for w in chunks[0]["words"]
            ]
            self._save_debug_output(result, has_timestamps=True)
            self._save_debug_session_summary(chunks, has_timestamps=True,
                                            total_stitch_steps=0,
                                            final_word_count=len(result))
            return result
        
        # Start with the first chunk
        stitched_words = chunks[0]["words"].copy()
        
        log_progress(f"Processing {len(chunks)} chunks (timestamped words)")
        
        # Iteratively stitch each subsequent chunk
        for i in range(1, len(chunks)):
            log_progress(f"Processing chunk {i + 1} of {len(chunks)}")
            current_chunk_words = chunks[i]["words"]
            chunk_id = chunks[i].get('chunk_id', i + 1)
            stitched_words = self._stitch_two_chunks_with_timestamps_debug(
                stitched_words, current_chunk_words,
                step_num=i, chunk_id=chunk_id
            )
        
        log_progress(f"Stitch complete: {len(stitched_words)} words total")
        
        # Convert to WordSegments
        result = [
            WordSegment(text=w["text"], start=w["start"], end=w["end"], speaker=w.get("speaker"))
            for w in stitched_words
        ]
        
        self._save_debug_output(result, has_timestamps=True)
        self._save_debug_session_summary(chunks, has_timestamps=True,
                                        total_stitch_steps=len(chunks) - 1,
                                        final_word_count=len(result))
        
        return result
    
    def _stitch_two_chunks(self, chunk1: List[str], chunk2: List[str]) -> List[str]:
        """
        Stitch two chunks, handling overlaps intelligently.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            
        Returns:
            Stitched word list
        """
        # Find the overlap region
        overlap_start, overlap_length = self._find_overlap(chunk1, chunk2)
        
        if overlap_length == 0:
            # No overlap found, simply concatenate
            log_progress("No overlap detected between chunks; concatenating directly")
            return chunk1 + chunk2
        
        overlapping_words = chunk1[overlap_start:overlap_start + overlap_length]
        log_progress(f"Overlap found: start={overlap_start}, length={overlap_length}, words={overlapping_words}")
        
        # Handle the overlap
        if overlap_length == len(chunk2):
            # Second chunk is entirely contained in first
            return chunk1
        
        # Take the non-overlapping part of chunk1 and all of chunk2
        # This handles cases where chunk2 might have better transcription at the boundary
        return chunk1[:overlap_start] + chunk2
    
    def _stitch_two_chunks_with_debug(self, chunk1: List[str], chunk2: List[str], 
                                       step_num: int, chunk_id: Any, has_timestamps: bool) -> List[str]:
        """
        Stitch two chunks with debug output for string words.
        """
        # Capture info before stitching
        chunk1_info = {
            'word_count': len(chunk1),
            'last_10_words': chunk1[-10:] if len(chunk1) >= 10 else chunk1
        }
        chunk2_info = {
            'chunk_id': chunk_id,
            'word_count': len(chunk2),
            'first_10_words': chunk2[:10] if len(chunk2) >= 10 else chunk2
        }
        
        # Find the overlap region
        overlap_start, overlap_length = self._find_overlap(chunk1, chunk2)
        
        # Capture overlap info
        overlap_info = {
            'overlap_found': overlap_length > 0,
            'overlap_start': overlap_start,
            'overlap_length': overlap_length,
            'overlapping_words': chunk1[overlap_start:overlap_start + overlap_length] if overlap_length > 0 else []
        }
        
        if overlap_length == 0:
            log_progress("No overlap detected between chunks; concatenating directly")
            result = chunk1 + chunk2
        elif overlap_length == len(chunk2):
            result = chunk1
        else:
            overlapping_words = chunk1[overlap_start:overlap_start + overlap_length]
            log_progress(f"Overlap found: start={overlap_start}, length={overlap_length}, words={overlapping_words}")
            result = chunk1[:overlap_start] + chunk2
        
        # Capture result info
        result_info = {
            'word_count': len(result),
            'words_from_chunk1': overlap_start if overlap_length > 0 else len(chunk1),
            'words_from_chunk2': len(chunk2) if overlap_length == 0 else (len(chunk2) if overlap_length != len(chunk2) else 0)
        }
        
        # Save debug info
        self._save_debug_stitch_step(step_num, chunk1_info, chunk2_info, overlap_info, result_info)
        
        return result
    
    def _stitch_two_chunks_with_timestamps(self, chunk1: List[Dict[str, Any]], chunk2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stitch two chunks with timestamped words, handling overlaps intelligently.
        
        Args:
            chunk1: First chunk's word list (dicts with "text", "start", "end")
            chunk2: Second chunk's word list (dicts with "text", "start", "end")
            
        Returns:
            Stitched word list with timestamps preserved
        """
        # Find the overlap region using text comparison
        overlap_start, overlap_length = self._find_overlap_with_timestamps(chunk1, chunk2)
        
        if overlap_length == 0:
            # No overlap found, simply concatenate
            log_progress("No overlap detected between chunks; concatenating directly")
            return chunk1 + chunk2
        
        overlapping_words = [w["text"] for w in chunk1[overlap_start:overlap_start + overlap_length]]
        log_progress(f"Overlap found: start={overlap_start}, length={overlap_length}, words={overlapping_words}")
        
        # Handle the overlap
        if overlap_length == len(chunk2):
            # Second chunk is entirely contained in first
            return chunk1
        
        # Take the non-overlapping part of chunk1 and all of chunk2
        # This preserves timestamps from chunk2 at the boundary
        return chunk1[:overlap_start] + chunk2
    
    def _stitch_two_chunks_with_timestamps_debug(self, chunk1: List[Dict[str, Any]], chunk2: List[Dict[str, Any]],
                                                   step_num: int, chunk_id: Any) -> List[Dict[str, Any]]:
        """
        Stitch two chunks with timestamped words and debug output.
        """
        # Capture info before stitching
        last_10 = chunk1[-10:] if len(chunk1) >= 10 else chunk1
        first_10 = chunk2[:10] if len(chunk2) >= 10 else chunk2
        
        chunk1_info = {
            'word_count': len(chunk1),
            'last_10_words': [w.get('text', '') for w in last_10],
            'time_range': {
                'start': chunk1[0].get('start', 0) if chunk1 else 0,
                'end': chunk1[-1].get('end', 0) if chunk1 else 0
            }
        }
        chunk2_info = {
            'chunk_id': chunk_id,
            'word_count': len(chunk2),
            'first_10_words': [w.get('text', '') for w in first_10],
            'time_range': {
                'start': chunk2[0].get('start', 0) if chunk2 else 0,
                'end': chunk2[-1].get('end', 0) if chunk2 else 0
            }
        }
        
        # Find the overlap region using text comparison
        overlap_start, overlap_length = self._find_overlap_with_timestamps(chunk1, chunk2)
        
        # Capture overlap info
        overlap_info = {
            'overlap_found': overlap_length > 0,
            'overlap_start': overlap_start,
            'overlap_length': overlap_length,
            'overlapping_words': [w.get('text', '') for w in chunk1[overlap_start:overlap_start + overlap_length]] if overlap_length > 0 else []
        }
        
        if overlap_length == 0:
            log_progress("No overlap detected between chunks; concatenating directly")
            result = chunk1 + chunk2
        elif overlap_length == len(chunk2):
            result = chunk1
        else:
            overlapping_words = [w["text"] for w in chunk1[overlap_start:overlap_start + overlap_length]]
            log_progress(f"Overlap found: start={overlap_start}, length={overlap_length}, words={overlapping_words}")
            result = chunk1[:overlap_start] + chunk2
        
        # Capture result info
        result_info = {
            'word_count': len(result),
            'words_from_chunk1': overlap_start if overlap_length > 0 else len(chunk1),
            'words_from_chunk2': len(chunk2) if overlap_length == 0 else (len(chunk2) if overlap_length != len(chunk2) else 0),
            'time_range': {
                'start': result[0].get('start', 0) if result else 0,
                'end': result[-1].get('end', 0) if result else 0
            }
        }
        
        # Save debug info
        self._save_debug_stitch_step(step_num, chunk1_info, chunk2_info, overlap_info, result_info)
        
        return result
    
    def _find_overlap_with_timestamps(self, chunk1: List[Dict[str, Any]], chunk2: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Find the overlap between two chunks with timestamped words.
        
        Args:
            chunk1: First chunk's word list (dicts with "text", "start", "end")
            chunk2: Second chunk's word list (dicts with "text", "start", "end")
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length)
        """
        # Maximum possible overlap length
        max_possible_overlap = min(len(chunk1), len(chunk2))
        
        # Check for overlaps from largest to smallest
        for overlap_size in range(max_possible_overlap, 0, -1):
            # Try to find this size overlap at the end of chunk1 and start of chunk2
            chunk1_end = [w["text"] for w in chunk1[-overlap_size:]]
            chunk2_start = [w["text"] for w in chunk2[:overlap_size]]
            
            # Check for exact match
            if chunk1_end == chunk2_start:
                return len(chunk1) - overlap_size, overlap_size
            
            # Check for fuzzy match (similar words)
            if self._is_fuzzy_match(chunk1_end, chunk2_start):
                return len(chunk1) - overlap_size, overlap_size
        
        # Check for partial word overlaps
        partial_overlap = self._find_partial_word_overlap_with_timestamps(chunk1, chunk2)
        if partial_overlap:
            return partial_overlap
        
        # No overlap found
        return 0, 0
    
    def _find_partial_word_overlap_with_timestamps(self, chunk1: List[Dict[str, Any]], chunk2: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """
        Check for partial word overlaps at chunk boundaries with timestamped words.
        
        Args:
            chunk1: First chunk's word list (dicts)
            chunk2: Second chunk's word list (dicts)
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length) if found, else None
        """
        if not chunk1 or not chunk2:
            return None
        
        last_word_chunk1 = chunk1[-1]["text"]
        first_word_chunk2 = chunk2[0]["text"]
        
        # Check if one word could be a continuation of the other
        if self._is_partial_word_match(last_word_chunk1, first_word_chunk2):
            log_progress(f"Partial word overlap: replacing '{last_word_chunk1}' with '{first_word_chunk2}'")
            return len(chunk1) - 1, 1
        
        # Check for two-word partial matches
        if len(chunk1) >= 2 and len(chunk2) >= 2:
            last_two_chunk1 = [chunk1[-2]["text"], chunk1[-1]["text"]]
            first_two_chunk2 = [chunk2[0]["text"], chunk2[1]["text"]]
            
            if (self._is_partial_word_match(last_two_chunk1[0], first_two_chunk2[0]) and
                self._words_similar(last_two_chunk1[1], first_two_chunk2[1])):
                log_progress(f"Partial word overlap: replacing '{last_two_chunk1}' with '{first_two_chunk2}'")
                return len(chunk1) - 2, 2
        
        return None
    
    def _find_overlap(self, chunk1: List[str], chunk2: List[str]) -> Tuple[int, int]:
        """
        Find the overlap between two chunks.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length)
        """
        # Maximum possible overlap length
        max_possible_overlap = min(len(chunk1), len(chunk2))
        
        # Check for overlaps from largest to smallest
        for overlap_size in range(max_possible_overlap, 0, -1):
            # Try to find this size overlap at the end of chunk1 and start of chunk2
            chunk1_end = chunk1[-overlap_size:]
            chunk2_start = chunk2[:overlap_size]
            
            # Check for exact match
            if chunk1_end == chunk2_start:
                return len(chunk1) - overlap_size, overlap_size
            
            # Check for fuzzy match (similar words)
            if self._is_fuzzy_match(chunk1_end, chunk2_start):
                return len(chunk1) - overlap_size, overlap_size
        
        # Check for partial word overlaps (e.g., "generational" -> "rational")
        # This handles cases where words might be cut off at boundaries
        partial_overlap = self._find_partial_word_overlap(chunk1, chunk2)
        if partial_overlap:
            return partial_overlap
        
        # No overlap found
        return 0, 0
    
    def _is_fuzzy_match(self, words1: List[str], words2: List[str]) -> bool:
        """
        Check if two word lists are fuzzy matches (similar words).
        
        Args:
            words1: First list of words
            words2: Second list of words
            
        Returns:
            True if the lists are fuzzy matches
        """
        if len(words1) != len(words2):
            return False
        
        # Check each corresponding word pair
        matches = 0
        fuzzy_pairs = []
        for w1, w2 in zip(words1, words2):
            if self._words_similar(w1, w2):
                matches += 1
                if w1 != w2:
                    ratio = SequenceMatcher(None, w1.lower(), w2.lower()).ratio()
                    fuzzy_pairs.append(f"'{w1}' ~ '{w2}' (ratio={ratio:.2f})")
        
        # Require at least min_overlap_ratio of words to be similar
        is_match = matches / len(words1) >= self.min_overlap_ratio
        if is_match and fuzzy_pairs:
            log_progress(f"Fuzzy matches in overlap: {', '.join(fuzzy_pairs)}")
        
        return is_match
    
    def _words_similar(self, word1: str, word2: str) -> bool:
        """
        Check if two words are similar using SequenceMatcher.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if words are similar
        """
        # Direct comparison
        if word1 == word2:
            return True
        
        # Case-insensitive comparison
        if word1.lower() == word2.lower():
            return True
        
        # Similarity ratio
        ratio = SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
        return ratio >= self.similarity_threshold
    
    def _find_partial_word_overlap(self, chunk1: List[str], chunk2: List[str]) -> Optional[Tuple[int, int]]:
        """
        Check for partial word overlaps at chunk boundaries.
        
        Args:
            chunk1: First chunk's word list
            chunk2: Second chunk's word list
            
        Returns:
            Tuple of (overlap_start_index_in_chunk1, overlap_length) if found, else None
        """
        # Check if the last word of chunk1 could be a partial match for the first word of chunk2
        if not chunk1 or not chunk2:
            return None
        
        last_word_chunk1 = chunk1[-1]
        first_word_chunk2 = chunk2[0]
        
        # Check if one word could be a continuation of the other
        if self._is_partial_word_match(last_word_chunk1, first_word_chunk2):
            # Replace the partial word in chunk1 with the complete word from chunk2
            log_progress(f"Partial word overlap: replacing '{last_word_chunk1}' with '{first_word_chunk2}'")
            return len(chunk1) - 1, 1
        
        # Check for two-word partial matches
        if len(chunk1) >= 2 and len(chunk2) >= 2:
            last_two_chunk1 = chunk1[-2:]
            first_two_chunk2 = chunk2[:2]
            
            if (self._is_partial_word_match(last_two_chunk1[0], first_two_chunk2[0]) and
                self._words_similar(last_two_chunk1[1], first_two_chunk2[1])):
                log_progress(f"Partial word overlap: replacing '{last_two_chunk1}' with '{first_two_chunk2}'")
                return len(chunk1) - 2, 2
        
        return None
    
    def _is_partial_word_match(self, word1: str, word2: str) -> bool:
        """
        Check if one word could be a partial match for another (e.g. "generational" -> "rational").
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if one word could be a partial match for the other
        """
        # If one word is much longer, check if it contains the other
        if len(word1) > len(word2) * 1.5 and word2.lower() in word1.lower():
            return True
        
        if len(word2) > len(word1) * 1.5 and word1.lower() in word2.lower():
            return True
        
        # Check for common suffixes/prefixes that might indicate partial transcription
        # For example, "generational" and "rational" share the "-ational" suffix
        min_len = min(len(word1), len(word2))
        
        # Check for significant overlap at the end
        for i in range(min_len // 2, min_len):
            if word1.lower()[-i:] == word2.lower()[-i:]:
                return True
        
        # Check for significant overlap at the beginning
        for i in range(min_len // 2, min_len):
            if word1.lower()[:i] == word2.lower()[:i]:
                return True
        
        return False


def stitch_chunks(chunks: List[Dict[str, Any]], **kwargs) -> Union[str, List[WordSegment]]:
    """
    Convenience function to stitch transcript chunks.
    
    Args:
        chunks: List of chunk dictionaries with 'chunk_id' and 'words' keys
               Words can be either List[str] or List[Dict] with timestamps
        **kwargs: Additional arguments (min_overlap_ratio, similarity_threshold, intermediate_dir)
        
    Returns:
        If words are strings: Stitched transcript as a string
        If words have timestamps: List[WordSegment] with preserved timestamps
    """
    stitcher = ChunkStitcher(
        min_overlap_ratio=kwargs.get('min_overlap_ratio', 0.6),
        similarity_threshold=kwargs.get('similarity_threshold', 0.7),
        intermediate_dir=kwargs.get('intermediate_dir')
    )
    return stitcher.stitch_chunks(chunks)