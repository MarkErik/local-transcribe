#!/usr/bin/env python3
"""
Split audio LLM turn builder provider that uses an LLM to intelligently merge speaker segments into turns.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import math

import requests

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class SplitAudioLlmTurnBuilderProvider(TurnBuilderProvider):
    """
    Split audio LLM turn builder that takes individual speaker timestamped data,
    includes start and end times for better context, and uses an LLM to intelligently merge
    text into coherent turns.
    """
    
    # Configuration parameters for chunking
    DEFAULT_CHUNK_SIZE = 100  # Number of words per LLM processing chunk
    DEFAULT_OVERLAP_RATIO = 0.2  # Fraction of overlap between adjacent chunks
    DEFAULT_USE_GROUND_TRUTH = True  # Enable ground truth verification mode
    DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Jaccard similarity threshold for discrepancy detection
    SMALL_TRANSCRIPT_THRESHOLD = 500  # Words below this use direct processing without chunking

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

        # Check if we should use chunking for long transcripts
        # Use internal flag to prevent recursion
        use_chunking = kwargs.get('use_chunking', True)
        is_recursive_call = kwargs.get('_is_recursive_call', False)
        
        if use_chunking and not is_recursive_call and len(words) > self.SMALL_TRANSCRIPT_THRESHOLD:
            print(f"DEBUG: Transcript is long ({len(words)} words), using ground truth-aware chunking")
            try:
                return self.build_turns_ground_truth_aware(words, **kwargs)
            except Exception as e:
                print(f"DEBUG: Chunking failed: {e}. Falling back to direct processing.")
        
        # Direct processing for small transcripts or when chunking is disabled
        print("DEBUG: Using direct LLM processing")
        
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
            parsed_turns = self._parse_response(response_text, words)
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

    def build_turns_ground_truth_aware(self, words: List[WordSegment], **kwargs) -> List[Turn]:
        """
        Complete implementation of ground truth-aware turn building with chunking.
        
        Args:
            words: Original word segments (ground truth)
            **kwargs: Configuration options including chunk_size, overlap_ratio
            
        Returns:
            List of Turn objects maintaining ground truth accuracy
        """
        # Get configuration parameters
        chunk_size = kwargs.get('chunk_size', self.DEFAULT_CHUNK_SIZE)
        overlap_ratio = kwargs.get('overlap_ratio', self.DEFAULT_OVERLAP_RATIO)
        use_ground_truth = kwargs.get('use_ground_truth', self.DEFAULT_USE_GROUND_TRUTH)
        
        print(f"DEBUG: Starting ground truth-aware turn building with {len(words)} words")
        print(f"DEBUG: Configuration - chunk_size={chunk_size}, overlap_ratio={overlap_ratio}, use_ground_truth={use_ground_truth}")
        
        # For small transcripts, use direct processing without chunking
        # Add _is_recursive_call flag to prevent infinite recursion
        if len(words) <= self.SMALL_TRANSCRIPT_THRESHOLD:
            print(f"DEBUG: Transcript is small ({len(words)} words), using direct processing")
            kwargs_copy = dict(kwargs)
            kwargs_copy['_is_recursive_call'] = True
            return self.build_turns(words, **kwargs_copy)
        
        # Calculate overlap size based on ratio
        overlap_size = max(1, int(chunk_size * overlap_ratio))
        print(f"DEBUG: Using chunk_size={chunk_size}, overlap_size={overlap_size}")
        
        # Create overlapping chunks
        chunks_with_indices = self._create_overlapping_chunks(words, chunk_size, overlap_size)
        print(f"DEBUG: Created {len(chunks_with_indices)} chunks")
        
        # Process each chunk with LLM
        processed_chunks = []
        for i, (chunk_words, original_indices) in enumerate(chunks_with_indices):
            print(f"DEBUG: Processing chunk {i+1}/{len(chunks_with_indices)} with {len(chunk_words)} words")
            
            try:
                # Use existing build_turns method for this chunk with recursion flag
                kwargs_copy = dict(kwargs)
                kwargs_copy['_is_recursive_call'] = True
                chunk_turns = self.build_turns(chunk_words, **kwargs_copy)
                processed_chunks.append({
                    'turns': chunk_turns,
                    'original_indices': original_indices,
                    'chunk_words': chunk_words
                })
                print(f"DEBUG: Chunk {i+1} processed successfully, got {len(chunk_turns)} turns")
            except Exception as e:
                print(f"DEBUG: Failed to process chunk {i+1}: {e}. Using ground truth fallback.")
                # Fallback to ground truth reconstruction
                fallback_turns = self._group_ground_truth_by_speaker(original_indices, words)
                processed_chunks.append({
                    'turns': fallback_turns,
                    'original_indices': original_indices,
                    'chunk_words': chunk_words
                })
        
        # Merge chunks with ground truth verification
        if use_ground_truth and len(processed_chunks) > 1:
            print("DEBUG: Merging chunks with ground truth verification")
            final_turns = self._merge_chunks_with_ground_truth(processed_chunks, words)
        else:
            print("DEBUG: Merging chunks without ground truth verification")
            final_turns = self._merge_chunks_simple(processed_chunks)
        
        print(f"DEBUG: Final result: {len(final_turns)} turns")
        return final_turns

    def _create_overlapping_chunks(self, words: List[WordSegment], chunk_size: int, overlap_size: int) -> List[Tuple[List[WordSegment], List[int]]]:
        """
        Create overlapping chunks while tracking original word indices.
        
        Args:
            words: Original word segments
            chunk_size: Number of words per chunk
            overlap_size: Number of overlapping words between chunks
            
        Returns:
            List of (chunk_words, original_indices) tuples
        """
        chunks = []
        total_words = len(words)
        
        if total_words <= chunk_size:
            # No need to chunk, return the entire transcript
            return [(words, list(range(total_words)))]
        
        # Calculate step size (non-overlapping portion)
        step_size = chunk_size - overlap_size
        
        # Create chunks with overlap
        for start_idx in range(0, total_words, step_size):
            end_idx = min(start_idx + chunk_size, total_words)
            
            # Get the words for this chunk
            chunk_words = words[start_idx:end_idx]
            original_indices = list(range(start_idx, end_idx))
            
            chunks.append((chunk_words, original_indices))
            
            # If we've reached the end, stop
            if end_idx >= total_words:
                break
        
        print(f"DEBUG: Created {len(chunks)} chunks with sizes: {[len(chunk[0]) for chunk in chunks]}")
        return chunks

    def _find_overlap_region(self, prev_chunk: dict, curr_chunk: dict, original_words: List[WordSegment]) -> Dict[str, Any]:
        """
        Find the overlap region between two chunks using original word indices.
        
        Args:
            prev_chunk: Previous chunk data with turns and original_indices
            curr_chunk: Current chunk data with turns and original_indices
            original_words: Original word segments
            
        Returns:
            Dictionary with overlap information including word indices and corresponding turns
        """
        prev_indices = set(prev_chunk['original_indices'])
        curr_indices = set(curr_chunk['original_indices'])
        
        # Find overlapping word indices
        overlap_indices = sorted(list(prev_indices.intersection(curr_indices)))
        
        if not overlap_indices:
            return {
                'has_overlap': False,
                'overlap_word_indices': [],
                'prev_overlap_turns': [],
                'curr_overlap_turns': []
            }
        
        # Find turns in previous chunk that correspond to overlap region
        prev_overlap_turns = []
        for turn in prev_chunk['turns']:
            # Check if this turn contains any of the overlap words
            turn_words_in_overlap = False
            for word_idx in overlap_indices:
                if (word_idx >= prev_chunk['original_indices'][0] and
                    word_idx <= prev_chunk['original_indices'][-1]):
                    # This word is in the previous chunk, check if it's part of this turn
                    # For simplicity, we'll assume the turn contains the word if the word's text
                    # appears in the turn's text (this is a heuristic)
                    original_word = original_words[word_idx]
                    if original_word.text in turn.text:
                        turn_words_in_overlap = True
                        break
            
            if turn_words_in_overlap:
                prev_overlap_turns.append(turn)
        
        # Find turns in current chunk that correspond to overlap region
        curr_overlap_turns = []
        for turn in curr_chunk['turns']:
            # Check if this turn contains any of the overlap words
            turn_words_in_overlap = False
            for word_idx in overlap_indices:
                if (word_idx >= curr_chunk['original_indices'][0] and
                    word_idx <= curr_chunk['original_indices'][-1]):
                    # This word is in the current chunk, check if it's part of this turn
                    original_word = original_words[word_idx]
                    if original_word.text in turn.text:
                        turn_words_in_overlap = True
                        break
            
            if turn_words_in_overlap:
                curr_overlap_turns.append(turn)
        
        return {
            'has_overlap': True,
            'overlap_word_indices': overlap_indices,
            'prev_overlap_turns': prev_overlap_turns,
            'curr_overlap_turns': curr_overlap_turns
        }

    def _verify_overlap_with_ground_truth(self,
                                         prev_chunk_overlap_turns: List[Turn],
                                         curr_chunk_overlap_turns: List[Turn],
                                         overlap_word_indices: List[int],
                                         original_words: List[WordSegment]) -> List[Turn]:
        """
        Verify overlap region against ground truth and resolve discrepancies.
        
        Args:
            prev_chunk_overlap_turns: Turns from previous chunk in overlap region
            curr_chunk_overlap_turns: Turns from current chunk in overlap region
            overlap_word_indices: Original word indices in the overlap region
            original_words: Original word segments (ground truth)
            
        Returns:
            List of verified turns for the overlap region
        """
        # Reconstruct ground truth turns from original words
        ground_truth_turns = self._group_ground_truth_by_speaker(overlap_word_indices, original_words)
        
        print(f"DEBUG: Ground truth verification - {len(ground_truth_turns)} GT turns, "
              f"{len(prev_chunk_overlap_turns)} prev chunk turns, {len(curr_chunk_overlap_turns)} curr chunk turns")
        
        # If we have ground truth turns, prefer them over LLM output
        if ground_truth_turns:
            return ground_truth_turns
        
        # Fallback: merge the two sets of turns, preferring current chunk when there's conflict
        merged_turns = []
        
        # Add non-overlapping turns from previous chunk
        for turn in prev_chunk_overlap_turns:
            # Check if this turn overlaps with any current chunk turn
            has_conflict = False
            for curr_turn in curr_chunk_overlap_turns:
                if (turn.speaker == curr_turn.speaker and
                    self._calculate_text_similarity(turn.text, curr_turn.text) > self.DEFAULT_SIMILARITY_THRESHOLD):
                    has_conflict = True
                    break
            
            if not has_conflict:
                merged_turns.append(turn)
        
        # Add all turns from current chunk (they take precedence)
        merged_turns.extend(curr_chunk_overlap_turns)
        
        return merged_turns

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize texts into word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)

    def _group_ground_truth_by_speaker(self, word_indices: List[int], original_words: List[WordSegment]) -> List[Turn]:
        """
        Group ground truth words into turns by speaker.
        
        Creates turns directly from the original transcript text,
        ensuring 100% accuracy to the source material.
        
        Args:
            word_indices: Indices of words to group
            original_words: Original word segments
            
        Returns:
            List of Turn objects created from ground truth
        """
        if not word_indices:
            return []
        
        turns = []
        current_speaker = None
        current_text = []
        current_start = None
        current_end = None
        
        for idx in word_indices:
            if idx >= len(original_words):
                print(f"DEBUG: Warning: Word index {idx} out of range, skipping")
                continue
                
            word = original_words[idx]
            
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
                # Continue current turn
                current_text.append(word.text)
                current_end = max(current_end, word.end)
        
        # Flush last turn
        if current_text:
            turns.append(Turn(
                speaker=current_speaker,
                start=current_start,
                end=current_end,
                text=" ".join(current_text)
            ))
        
        print(f"DEBUG: Ground truth grouping created {len(turns)} turns from {len(word_indices)} words")
        return turns

    def _merge_chunks_with_ground_truth(self, processed_chunks: List[dict], original_words: List[WordSegment]) -> List[Turn]:
        """
        Merge processed chunks with ground truth verification in overlap regions.
        
        Args:
            processed_chunks: List of processed chunk data
            original_words: Original word segments
            
        Returns:
            Merged list of Turn objects
        """
        if not processed_chunks:
            return []
        
        if len(processed_chunks) == 1:
            return processed_chunks[0]['turns']
        
        # Start with all turns from the first chunk
        merged_turns = list(processed_chunks[0]['turns'])
        
        # Process subsequent chunks with overlap verification
        for i in range(1, len(processed_chunks)):
            prev_chunk = processed_chunks[i-1]
            curr_chunk = processed_chunks[i]
            
            # Find overlap region
            overlap_info = self._find_overlap_region(prev_chunk, curr_chunk, original_words)
            
            if overlap_info['has_overlap']:
                print(f"DEBUG: Processing overlap between chunks {i} and {i+1}")
                
                # Verify overlap with ground truth
                verified_overlap_turns = self._verify_overlap_with_ground_truth(
                    overlap_info['prev_overlap_turns'],
                    overlap_info['curr_overlap_turns'],
                    overlap_info['overlap_word_indices'],
                    original_words
                )
                
                # Remove overlapping turns from the END of merged_turns that came from prev_chunk
                # We need to identify which turns in merged_turns are from the overlap region
                turns_to_remove = []
                for j, turn in enumerate(merged_turns):
                    should_remove = False
                    for overlap_turn in overlap_info['prev_overlap_turns']:
                        if (turn.speaker == overlap_turn.speaker and
                            self._calculate_text_similarity(turn.text, overlap_turn.text) > 0.5):
                            should_remove = True
                            break
                    
                    if should_remove:
                        turns_to_remove.append(j)
                
                # Remove in reverse order to maintain indices
                for j in reversed(turns_to_remove):
                    merged_turns.pop(j)
                
                # Add verified overlap turns
                merged_turns.extend(verified_overlap_turns)
                
                # Add non-overlapping turns from current chunk
                for turn in curr_chunk['turns']:
                    is_in_overlap = False
                    for overlap_turn in overlap_info['curr_overlap_turns']:
                        if (turn.speaker == overlap_turn.speaker and
                            self._calculate_text_similarity(turn.text, overlap_turn.text) > 0.5):
                            is_in_overlap = True
                            break
                    
                    if not is_in_overlap:
                        merged_turns.append(turn)
            else:
                # No overlap, just add all turns from current chunk
                merged_turns.extend(curr_chunk['turns'])
        
        return merged_turns

    def _merge_chunks_simple(self, processed_chunks: List[dict]) -> List[Turn]:
        """
        Simple merging of chunks without ground truth verification.
        
        Args:
            processed_chunks: List of processed chunk data
            
        Returns:
            Merged list of Turn objects
        """
        if not processed_chunks:
            return []
        
        if len(processed_chunks) == 1:
            return processed_chunks[0]['turns']
        
        merged_turns = []
        
        # Add all turns from the first chunk
        merged_turns.extend(processed_chunks[0]['turns'])
        
        # For subsequent chunks, add only non-overlapping turns
        for i in range(1, len(processed_chunks)):
            curr_chunk = processed_chunks[i]
            
            # Simple heuristic: skip turns that are too similar to the last added turn
            for turn in curr_chunk['turns']:
                should_add = True
                
                if merged_turns:
                    last_turn = merged_turns[-1]
                    # If same speaker and high text similarity, might be overlap
                    if (last_turn.speaker == turn.speaker and
                        self._calculate_text_similarity(last_turn.text, turn.text) > 0.8):
                        should_add = False
                
                if should_add:
                    merged_turns.append(turn)
        
        return merged_turns

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
            for i, seg in enumerate(prepared_segments)
        ])

        prompt = f"""Segments:
{segments_text}

Example Output:
[{{"speaker": "A", "text": "Hello, how are you? I'm doing well."}}, {{"speaker": "B", "text": "That's great to hear."}}]"""

        return prompt

    def _query_llm(self, prompt: str, url: str, timeout: Optional[int]) -> str:
        """
        Query the LLM with the prompt.

        Args:
            prompt: The prompt to send
            url: LLM server URL
            timeout: Request timeout in seconds (None for no timeout)

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

    def _parse_response(self, response_text: str, original_words: List[WordSegment]) -> List[Dict[str, str]]:
        """
        Parse the LLM response into a list of turns.

        Args:
            response_text: Raw LLM response
            original_words: Original word segments (used to validate speaker IDs)

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

        # Get valid speaker IDs from original words
        valid_speakers = set(word.speaker for word in original_words if word.speaker is not None)
        
        for turn in turns:
            if not isinstance(turn, dict) or "speaker" not in turn or "text" not in turn:
                raise ValueError("Each turn must have 'speaker' and 'text' fields")
            
            # Validate speaker ID exists in original data
            if turn["speaker"] not in valid_speakers:
                raise ValueError(f"Invalid speaker ID '{turn['speaker']}' not found in original data. "
                               f"Valid speakers: {valid_speakers}")

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
            
            # Tokenize the turn text into expected words (split by whitespace and remove punctuation)
            # This creates a more robust matching approach
            turn_words = turn_text.replace('"', '').strip().split()
            turn_word_index = 0

            print(f"DEBUG: Turn expects {len(turn_words)} words")

            # Match words sequentially - words must appear in order
            start_word_index = word_index
            while word_index < len(original_words) and turn_word_index < len(turn_words):
                word = original_words[word_index]
                expected_word = turn_words[turn_word_index]
                
                print(f"DEBUG: Checking word at index {word_index}: speaker={word.speaker}, text='{word.text}', expected='{expected_word}'")
                
                if word.speaker == speaker:
                    # Normalize both words for comparison (case-insensitive, remove punctuation)
                    word_normalized = word.text.lower().strip('.,!?;:"\'-')
                    expected_normalized = expected_word.lower().strip('.,!?;:"\'-')
                    
                    if word_normalized == expected_normalized:
                        # Exact match found
                        matched_words.append(word)
                        turn_word_index += 1
                        word_index += 1
                        print(f"DEBUG: Matched word '{word.text}'")
                    else:
                        # Word doesn't match - this might indicate LLM modified the text
                        # Try to skip this original word and continue
                        print(f"DEBUG: Word mismatch: got '{word.text}', expected '{expected_word}', skipping original word")
                        word_index += 1
                else:
                    # Different speaker encountered
                    print(f"DEBUG: Different speaker ({word.speaker}), stopping turn.")
                    break

            # If we didn't match all expected words, fall back to speaker-based grouping
            if turn_word_index < len(turn_words):
                print(f"DEBUG: Warning: Only matched {turn_word_index}/{len(turn_words)} words for turn")
                # Reset and use simpler approach: collect all consecutive words from this speaker
                word_index = start_word_index
                matched_words = []
                while word_index < len(original_words):
                    word = original_words[word_index]
                    if word.speaker == speaker:
                        matched_words.append(word)
                        word_index += 1
                    else:
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