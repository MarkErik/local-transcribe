#!/usr/bin/env python3
"""
Improved split audio LLM turn builder that handles separate per-speaker transcripts.

This provider properly merges and sorts words from multiple speakers before using
an LLM to intelligently group them into natural conversational turns.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Tuple

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry


class SplitAudioLlmTurnBuilderImprovedProvider(TurnBuilderProvider):
    """
    Improved turn builder for split audio processing.
    
    Key improvements:
    - Properly merges and sorts words from separate speaker files
    - Better LLM prompting for natural turn detection
    - Robust word matching with fuzzy logic
    - Comprehensive validation and error handling
    - Smart chunking for large transcripts
    """
    
    # Configuration parameters
    DEFAULT_CHUNK_SIZE = 100
    DEFAULT_OVERLAP_RATIO = 0.2
    SMALL_TRANSCRIPT_THRESHOLD = 500
    DEFAULT_TIMING_GAP_THRESHOLD = 2.0  # seconds
    
    @property
    def name(self) -> str:
        return "split_audio_llm_turn_builder_improved"
    
    @property
    def short_name(self) -> str:
        return "Split Audio LLM (Improved)"
    
    @property
    def description(self) -> str:
        return "Improved LLM-based turn builder for split audio with proper word merging and sorting"
    
    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from word segments using improved LLM-based merging.
        
        Key steps:
        1. Merge and sort words by timestamp (handles separate speaker files)
        2. Validate chronological ordering
        3. For large transcripts, use chunking with overlap
        4. For small transcripts, process directly
        5. Query LLM with improved prompting
        6. Reconstruct timings with robust matching
        
        Args:
            words: Word segments (may be unsorted or from separate speakers)
            **kwargs: Configuration options
        
        Returns:
            List of Turn objects with natural turn boundaries
        """
        if not words:
            print("WARNING: No words provided to turn builder")
            return []
        
        print(f"\n{'='*60}")
        print(f"Split Audio LLM Turn Builder (Improved)")
        print(f"{'='*60}")
        print(f"Input: {len(words)} word segments")
        
        # CRITICAL: Merge and sort words by timestamp
        sorted_words = self._merge_and_sort_words(words)
        print(f"After merge/sort: {len(sorted_words)} words in chronological order")
        
        # Validate chronological ordering
        self._validate_chronological_order(sorted_words)
        
        # Log conversation statistics
        self._log_conversation_stats(sorted_words)
        
        # Check if we should use chunking
        use_chunking = kwargs.get('use_chunking', True)
        
        if use_chunking and len(sorted_words) > self.SMALL_TRANSCRIPT_THRESHOLD:
            print(f"Using chunked processing (transcript > {self.SMALL_TRANSCRIPT_THRESHOLD} words)")
            return self._build_turns_with_chunking(sorted_words, **kwargs)
        else:
            print(f"Using direct processing (transcript <= {self.SMALL_TRANSCRIPT_THRESHOLD} words)")
            return self._build_turns_direct(sorted_words, **kwargs)
    
    def _merge_and_sort_words(self, words: List[WordSegment]) -> List[WordSegment]:
        """
        Merge and sort words by timestamp.
        
        This is the critical fix - handles words from separate speaker files
        that need to be interleaved chronologically.
        
        Args:
            words: Original word segments (possibly unsorted)
        
        Returns:
            Words sorted by start timestamp
        """
        # Sort by start time
        sorted_words = sorted(words, key=lambda w: w.start)
        
        # Show first few words to verify ordering
        if sorted_words:
            print("\nFirst 10 words after sorting:")
            for i, word in enumerate(sorted_words[:10]):
                print(f"  {i}: {word.start:>8.2f}s - {word.speaker:<15} '{word.text}'")
        
        return sorted_words
    
    def _validate_chronological_order(self, words: List[WordSegment]) -> None:
        """Validate that words are in chronological order and check for issues."""
        if len(words) < 2:
            return
        
        issues = []
        for i in range(len(words) - 1):
            if words[i].start > words[i+1].start:
                issues.append(f"Word {i} starts after word {i+1}: {words[i].start:.2f} > {words[i+1].start:.2f}")
        
        if issues:
            print(f"WARNING: Found {len(issues)} chronological ordering issues:")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue}")
        else:
            print("✓ Chronological order validated")
    
    def _log_conversation_stats(self, words: List[WordSegment]) -> None:
        """Log statistics about the conversation."""
        if not words:
            return
        
        # Count speaker distribution
        speaker_counts = {}
        for word in words:
            speaker = word.speaker or "Unknown"
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        # Detect speaker changes (potential turn boundaries)
        speaker_changes = 0
        for i in range(len(words) - 1):
            if words[i].speaker != words[i+1].speaker:
                speaker_changes += 1
        
        # Time span
        duration = words[-1].end - words[0].start
        
        print(f"\nConversation Statistics:")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Speakers: {list(speaker_counts.keys())}")
        for speaker, count in speaker_counts.items():
            print(f"    {speaker}: {count} words")
        print(f"  Speaker changes: {speaker_changes}")
    
    def _build_turns_direct(self, words: List[WordSegment], **kwargs) -> List[Turn]:
        """
        Build turns by processing all words at once (for small transcripts).
        
        Args:
            words: Sorted word segments
            **kwargs: Configuration options
        
        Returns:
            List of Turn objects
        """
        print("\n--- Direct LLM Processing ---")
        
        # Prepare prompt
        prompt = self._build_improved_prompt(words, **kwargs)
        
        # Query LLM
        llm_url = kwargs.get('llm_url', 'http://localhost:8080')
        timeout = kwargs.get('timeout', None)
        
        try:
            llm_response = self._query_llm(prompt, llm_url, timeout)
        except Exception as e:
            print(f"ERROR: LLM query failed: {e}")
            print("Falling back to simple turn builder")
            return self._fallback_simple_turns(words)
        
        # Parse response
        try:
            parsed_turns = self._parse_llm_response(llm_response, words)
        except Exception as e:
            print(f"ERROR: Failed to parse LLM response: {e}")
            print("Falling back to simple turn builder")
            return self._fallback_simple_turns(words)
        
        # Reconstruct timings
        turns = self._reconstruct_timings_robust(parsed_turns, words)
        
        print(f"✓ Created {len(turns)} turns from {len(parsed_turns)} LLM outputs")
        return turns
    
    def _build_turns_with_chunking(self, words: List[WordSegment], **kwargs) -> List[Turn]:
        """
        Build turns using chunking for large transcripts.
        
        Args:
            words: Sorted word segments
            **kwargs: Configuration options
        
        Returns:
            List of Turn objects
        """
        print("\n--- Chunked Processing ---")
        
        chunk_size = kwargs.get('chunk_size', self.DEFAULT_CHUNK_SIZE)
        overlap_ratio = kwargs.get('overlap_ratio', self.DEFAULT_OVERLAP_RATIO)
        overlap_size = max(1, int(chunk_size * overlap_ratio))
        
        print(f"Chunk size: {chunk_size}, Overlap: {overlap_size}")
        
        # Create overlapping chunks
        chunks = self._create_overlapping_chunks(words, chunk_size, overlap_size)
        print(f"Created {len(chunks)} chunks")
        
        # Process each chunk
        all_turns = []
        for i, (chunk_words, start_idx, end_idx) in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{len(chunks)} (words {start_idx}-{end_idx})")
            
            try:
                # Build turns for this chunk
                chunk_turns = self._build_turns_direct(chunk_words, **kwargs)
                all_turns.extend(chunk_turns)
            except Exception as e:
                print(f"ERROR processing chunk {i+1}: {e}")
                # Fall back to simple turns for this chunk
                all_turns.extend(self._fallback_simple_turns(chunk_words))
        
        # Merge overlapping turns
        merged_turns = self._merge_overlapping_turns(all_turns)
        
        print(f"\n✓ Final result: {len(merged_turns)} turns from {len(chunks)} chunks")
        return merged_turns
    
    def _create_overlapping_chunks(
        self,
        words: List[WordSegment],
        chunk_size: int,
        overlap_size: int
    ) -> List[Tuple[List[WordSegment], int, int]]:
        """
        Create overlapping chunks of words.
        
        Args:
            words: Sorted word segments
            chunk_size: Number of words per chunk
            overlap_size: Number of overlapping words
        
        Returns:
            List of (chunk_words, start_index, end_index) tuples
        """
        chunks = []
        total_words = len(words)
        step_size = chunk_size - overlap_size
        
        for start_idx in range(0, total_words, step_size):
            end_idx = min(start_idx + chunk_size, total_words)
            chunk_words = words[start_idx:end_idx]
            chunks.append((chunk_words, start_idx, end_idx))
            
            if end_idx >= total_words:
                break
        
        return chunks
    
    def _build_improved_prompt(self, words: List[WordSegment], **kwargs) -> str:
        """
        Build an improved prompt for the LLM with clear instructions.
        
        Args:
            words: Word segments to process
            **kwargs: Configuration options
        
        Returns:
            Formatted prompt string
        """
        timing_gap_threshold = kwargs.get('timing_gap_threshold', self.DEFAULT_TIMING_GAP_THRESHOLD)
        
        # Build word list with timing information
        word_lines = []
        for i, word in enumerate(words):
            word_lines.append(f'{i+1}. [{word.start:.2f}s] {word.speaker}: "{word.text}"')
            
            # Add gap markers for significant pauses
            if i < len(words) - 1:
                gap = words[i+1].start - word.end
                if gap > timing_gap_threshold:
                    word_lines.append(f"    [PAUSE: {gap:.1f}s]")
        
        words_text = "\n".join(word_lines)
        
        prompt = f"""You are analyzing a conversation transcript. Your task is to group the individual words into natural conversational turns.

A "turn" is a continuous speech segment by one speaker. Turns typically end when:
- The speaker changes
- There is a significant pause (indicated by [PAUSE] markers)
- A thought or sentence naturally completes

IMPORTANT INSTRUCTIONS:
1. Group consecutive words from the same speaker into single turns
2. Respect natural conversation boundaries - don't split mid-sentence
3. Use timing gaps as hints for turn boundaries
4. Keep the EXACT original text - do not paraphrase or clean up
5. Output JSON format: [{{"speaker": "SpeakerName", "text": "exact words from input"}}]

CONVERSATION TRANSCRIPT:
{words_text}

Output the turns as a JSON array. Each turn should have "speaker" and "text" fields.
Example format:
[
  {{"speaker": "Interviewer", "text": "hello how are you today"}},
  {{"speaker": "Participant", "text": "i'm doing well thank you for asking"}}
]

JSON OUTPUT:"""
        
        return prompt
    
    def _query_llm(self, prompt: str, url: str, timeout: Optional[int]) -> str:
        """
        Query the LLM with the given prompt.
        
        Args:
            prompt: The prompt to send
            url: LLM server URL
            timeout: Request timeout in seconds
        
        Returns:
            LLM response text
        """
        print(f"Querying LLM at {url}...")
        print(f"Prompt length: {len(prompt)} characters")
        
        payload = {
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 16384,
            "stop": ["</s>", "CONVERSATION TRANSCRIPT:", "---"]
        }
        
        response = requests.post(
            f"{url}/completion",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("content", "")
        
        print(f"✓ LLM response received ({len(response_text)} characters)")
        return response_text
    
    def _parse_llm_response(self, response_text: str, original_words: List[WordSegment]) -> List[Dict[str, str]]:
        """
        Parse the LLM response into turn data.
        
        Args:
            response_text: Raw LLM response
            original_words: Original word segments (for validation)
        
        Returns:
            List of dicts with 'speaker' and 'text' keys
        """
        print("\n--- Parsing LLM Response ---")
        
        # Try to extract JSON from response
        response_text = response_text.strip()
        
        # Look for JSON array
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON array found in LLM response")
        
        json_text = response_text[start_idx:end_idx+1]
        
        # Parse JSON
        try:
            turns_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Attempted to parse: {json_text[:500]}...")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
        
        if not isinstance(turns_data, list):
            raise ValueError(f"Expected JSON array, got {type(turns_data)}")
        
        # Validate turn structure
        parsed_turns = []
        for i, turn in enumerate(turns_data):
            if not isinstance(turn, dict):
                print(f"WARNING: Turn {i} is not a dict, skipping")
                continue
            
            if 'speaker' not in turn or 'text' not in turn:
                print(f"WARNING: Turn {i} missing required fields, skipping")
                continue
            
            parsed_turns.append({
                'speaker': str(turn['speaker']),
                'text': str(turn['text']).strip()
            })
        
        print(f"✓ Parsed {len(parsed_turns)} turns from LLM response")
        
        # Show sample
        if parsed_turns:
            print("\nSample turns:")
            for i, turn in enumerate(parsed_turns[:3]):
                text_preview = turn['text'][:60] + "..." if len(turn['text']) > 60 else turn['text']
                print(f"  {i+1}. {turn['speaker']}: {text_preview}")
        
        return parsed_turns
    
    def _reconstruct_timings_robust(
        self,
        parsed_turns: List[Dict[str, str]],
        original_words: List[WordSegment]
    ) -> List[Turn]:
        """
        Reconstruct timing information using robust word matching.
        
        Args:
            parsed_turns: Turns from LLM with speaker and text
            original_words: Original word segments with timings
        
        Returns:
            List of Turn objects with timings
        """
        print("\n--- Reconstructing Timings ---")
        
        turns = []
        word_index = 0
        
        for turn_idx, turn_data in enumerate(parsed_turns):
            speaker = turn_data['speaker']
            turn_text = turn_data['text']
            
            # Tokenize turn text
            turn_words = turn_text.lower().split()
            
            # Find matching words in original transcript
            matched_words = []
            search_start = word_index
            
            # Try to match words sequentially
            turn_word_idx = 0
            max_lookahead = 20  # How many words ahead to search
            
            while word_index < len(original_words) and turn_word_idx < len(turn_words):
                # Try to find next turn word in original words
                found = False
                
                for lookahead in range(min(max_lookahead, len(original_words) - word_index)):
                    word = original_words[word_index + lookahead]
                    expected_word = turn_words[turn_word_idx]
                    
                    # Normalize for comparison
                    word_normalized = word.text.lower().strip('.,!?;:"\'-')
                    expected_normalized = expected_word.strip('.,!?;:"\'-')
                    
                    # Check for match
                    if (word.speaker == speaker and 
                        word_normalized == expected_normalized):
                        # Found match - add all skipped words from this speaker
                        for i in range(lookahead + 1):
                            w = original_words[word_index]
                            if w.speaker == speaker:
                                matched_words.append(w)
                            word_index += 1
                        
                        turn_word_idx += 1
                        found = True
                        break
                
                if not found:
                    # Couldn't find this word - skip ahead
                    if word_index < len(original_words):
                        word_index += 1
                    else:
                        break
            
            # Create turn if we matched words
            if matched_words:
                start = min(w.start for w in matched_words)
                end = max(w.end for w in matched_words)
                
                # Reconstruct text from matched words
                reconstructed_text = " ".join(w.text for w in matched_words)
                
                turns.append(Turn(
                    speaker=speaker,
                    start=start,
                    end=end,
                    text=reconstructed_text
                ))
            else:
                print(f"WARNING: Could not match words for turn {turn_idx}: {speaker}")
                print(f"  Expected: {turn_text[:80]}...")
                # Continue anyway - don't fail the entire process
        
        print(f"✓ Created {len(turns)} turns with timings")
        return turns
    
    def _merge_overlapping_turns(self, turns: List[Turn]) -> List[Turn]:
        """
        Merge turns that may overlap due to chunking.
        
        Args:
            turns: List of turns that may have overlaps
        
        Returns:
            Merged list of turns
        """
        if not turns:
            return []
        
        # Sort by start time
        sorted_turns = sorted(turns, key=lambda t: t.start)
        
        merged = []
        current_turn = None
        
        for turn in sorted_turns:
            if current_turn is None:
                current_turn = turn
            elif (current_turn.speaker == turn.speaker and 
                  turn.start <= current_turn.end + 1.0):  # Allow 1s gap for same speaker
                # Merge with current turn
                current_turn = Turn(
                    speaker=current_turn.speaker,
                    start=current_turn.start,
                    end=max(current_turn.end, turn.end),
                    text=current_turn.text + " " + turn.text
                )
            else:
                # New turn - save current and start new
                merged.append(current_turn)
                current_turn = turn
        
        if current_turn:
            merged.append(current_turn)
        
        return merged
    
    def _fallback_simple_turns(self, words: List[WordSegment]) -> List[Turn]:
        """
        Fallback: create turns based on simple speaker changes.
        
        Args:
            words: Word segments
        
        Returns:
            List of Turn objects
        """
        print("\n--- Using Fallback Simple Turn Builder ---")
        
        if not words:
            return []
        
        turns = []
        current_speaker = None
        current_words = []
        
        for word in words:
            if word.speaker != current_speaker:
                # Save previous turn
                if current_words:
                    text = " ".join(w.text for w in current_words)
                    start = min(w.start for w in current_words)
                    end = max(w.end for w in current_words)
                    
                    turns.append(Turn(
                        speaker=current_speaker,
                        start=start,
                        end=end,
                        text=text
                    ))
                
                # Start new turn
                current_speaker = word.speaker
                current_words = [word]
            else:
                current_words.append(word)
        
        # Save last turn
        if current_words:
            text = " ".join(w.text for w in current_words)
            start = min(w.start for w in current_words)
            end = max(w.end for w in current_words)
            
            turns.append(Turn(
                speaker=current_speaker,
                start=start,
                end=end,
                text=text
            ))
        
        print(f"✓ Created {len(turns)} simple turns")
        return turns


def register_turn_builder_plugins():
    """Register turn builder plugins."""
    registry.register_turn_builder_provider(SplitAudioLlmTurnBuilderImprovedProvider())


# Auto-register on import
register_turn_builder_plugins()
