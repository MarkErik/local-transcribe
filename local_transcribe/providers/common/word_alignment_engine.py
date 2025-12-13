"""
Word Alignment Engine for aligning transcribed text with audio segments.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import pathlib


class WordAlignmentEngine:
    """
    Base class for word alignment engines.
    """
    
    # Configuration constants
    DEFAULT_GAP_PENALTY = -0.5
    DEFAULT_MIN_SIMILARITY_THRESHOLD = 0.6
    DEFAULT_MIN_WORD_DURATION = 0.02  # 20ms minimum word duration
    DEFAULT_MAX_FALLBACK_SIMILARITY = 0.8
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the WordAlignmentEngine.
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Apply configuration with defaults
        self.config = config or {}
        self.gap_penalty = self.config.get('gap_penalty', self.DEFAULT_GAP_PENALTY)
        self.min_similarity_threshold = self.config.get('min_similarity_threshold', self.DEFAULT_MIN_SIMILARITY_THRESHOLD)
        self.min_word_duration = self.config.get('min_word_duration', self.DEFAULT_MIN_WORD_DURATION)
        self.max_fallback_similarity = self.config.get('max_fallback_similarity', self.DEFAULT_MAX_FALLBACK_SIMILARITY)
        
        self.logger.info(f"WordAlignmentEngine initialized with config: gap_penalty={self.gap_penalty}, "
                        f"min_similarity_threshold={self.min_similarity_threshold}, "
                        f"min_word_duration={self.min_word_duration}, "
                        f"max_fallback_similarity={self.max_fallback_similarity}")
    
    def align_words(self, audio_segments: List[Dict[str, Any]], transcript: str) -> List[Dict[str, Any]]:
        """
        Align words in transcript with audio segments.
        """
        raise NotImplementedError("Subclasses must implement align_words method")
    
    
    
    def parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str,
                                    segment_start_time: float = 0.0, segment_end_time: float = 0.0,
                                    speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse MFA TextGrid and return list of word dicts with timestamps.
        """
        word_dicts = []
        
        try:
            # Enhanced file reading with encoding detection
            with open(textgrid_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                self.logger.warning("TextGrid file is empty")
                return self.create_simple_alignment(original_transcript, segment_start_time,
                                                  segment_end_time - segment_start_time, speaker, None)
            
            # Build mapping of normalized to original words
            original_words = original_transcript.split()
            normalized_to_original = self._build_word_mapping(original_words)
            
            # Find word tier with improved error handling
            word_tier_bounds = self._find_word_tier(lines)
            if not word_tier_bounds:
                self.logger.warning("Could not find word tier in TextGrid")
                return self.create_simple_alignment(original_transcript, segment_start_time,
                                                  segment_end_time - segment_start_time, speaker, None)
            
            # Parse intervals with comprehensive error handling
            word_dicts = self._parse_textgrid_intervals(lines, word_tier_bounds,
                                                       normalized_to_original,
                                                       segment_start_time, speaker)
            
            # Validate results
            if not word_dicts:
                self.logger.warning("No valid words extracted from TextGrid")
                return self.create_simple_alignment(original_transcript, segment_start_time,
                                                  segment_end_time - segment_start_time, speaker, None)
            
            return word_dicts
            
        except Exception as e:
            self.logger.error(f"Failed to parse TextGrid {textgrid_path}: {e}")
            return self.create_simple_alignment(original_transcript, segment_start_time,
                                              segment_end_time - segment_start_time, speaker, None)
    
    def _build_word_mapping(self, original_words: List[str]) -> Dict[str, str]:
        """
        Build mapping of normalized words to original words.
        """
        normalized_to_original = {}
        
        for word in original_words:
            normalized = self.normalize_word_for_matching(word)
            if normalized:
                # Store the first occurrence of each normalized word
                if normalized not in normalized_to_original:
                    normalized_to_original[normalized] = word
        
        return normalized_to_original
    
    def _find_word_tier(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        """
        Find the word tier in TextGrid lines.
        """
        word_tier_start = None
        word_tier_end = None
        
        for i, line in enumerate(lines):
            # Look for word tier definition (matching MFA's actual output format)
            if 'name = "words"' in line:
                word_tier_start = i
                self.logger.debug(f"Found word tier at line {i}")
            # Find the next tier (phones) to know where words tier ends
            elif word_tier_start is not None and 'name = "phones"' in line:
                word_tier_end = i
                self.logger.debug(f"Word tier ends at line {i}")
                break
        
        if word_tier_start is None:
            return None
        
        # If we didn't find the phones tier, parse until end of file
        if word_tier_end is None:
            word_tier_end = len(lines)
            self.logger.debug("No phones tier found, parsing until end of file")
        
        return (word_tier_start, word_tier_end)
    
    def _parse_textgrid_intervals(self, lines: List[str], word_tier_bounds: Tuple[int, int],
                                 normalized_to_original: Dict[str, str],
                                 segment_start_time: float,
                                 speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse intervals from TextGrid word tier using the same logic as the working MFA aligner.
        """
        word_dicts = []
        start_line, end_line = word_tier_bounds
        interval_count = 0
        
        # Parse intervals only within the words tier
        i = start_line
        while i < end_line:
            line = lines[i].strip()
            if line.startswith('intervals ['):
                # Parse interval - skip to xmin, xmax, text lines
                i += 1  # Move to xmin line
                if i >= end_line:
                    break
                xmin_line = lines[i].strip()
                i += 1  # Move to xmax line
                if i >= end_line:
                    break
                xmax_line = lines[i].strip()
                i += 1  # Move to text line
                if i >= end_line:
                    break
                text_line = lines[i].strip()

                # Extract values
                try:
                    start = float(xmin_line.split('=')[1].strip())
                    end = float(xmax_line.split('=')[1].strip())
                    mfa_text = text_line.split('=')[1].strip().strip('"')

                    # Skip empty intervals, silence markers, and special tokens
                    if mfa_text and mfa_text not in ["", "<eps>", "sil", "sp", "spn"]:
                        interval_count += 1
                        
                        # Map MFA's normalized text back to original formatting
                        normalized_key = mfa_text.lower()
                        
                        if normalized_key in normalized_to_original:
                            # Get the first occurrence of this word from the original transcript
                            original_words = normalized_to_original[normalized_key]
                            original_text = original_words[0] if original_words else mfa_text
                        else:
                            # Fallback: use MFA's text if we can't find a mapping
                            original_text = mfa_text
                        
                        # Adjust timing relative to segment start
                        start_time = start + segment_start_time
                        end_time = end + segment_start_time
                        
                        # Validate timing
                        if start_time < end_time and (end_time - start_time) >= self.min_word_duration:
                            word_dict = {
                                "text": original_text,
                                "start": start_time,
                                "end": end_time,
                                "speaker": speaker
                            }
                            word_dicts.append(word_dict)
                            
                except (ValueError, IndexError) as e:
                    self.logger.debug(f"Failed to parse interval at line {i}: {e}")
                    pass

            i += 1
        
        self.logger.info(f"Successfully parsed {interval_count} intervals, created {len(word_dicts)} word dictionaries")
        return word_dicts
    
    def create_simple_alignment(self, transcript: str, segment_start_time: float = 0.0,
                               segment_duration: Optional[float] = None,
                               speaker: Optional[str] = None,
                               audio_wav: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Create simple even-distribution alignment as fallback.
        """
        words = transcript.split()
        if not words:
            return []
        
        # Calculate duration
        if segment_duration is not None:
            duration = segment_duration
        elif audio_wav is not None:
            duration = len(audio_wav) / 16000.0  # Assuming 16kHz sample rate
        else:
            # Estimate based on word count (average 0.5s per word)
            duration = len(words) * 0.5
        
        # Distribute time evenly among words
        word_duration = duration / len(words)
        
        word_dicts = []
        current_time = segment_start_time
        
        for word in words:
            word_end = current_time + word_duration
            word_dicts.append({
                "text": word,
                "start": current_time,
                "end": word_end,
                "speaker": speaker
            })
            current_time = word_end
        
        return word_dicts
    
    def normalize_word_for_matching(self, word: str) -> str:
        """
        Normalize word for comparison during alignment.
        """
        if not word:
            return ""
        return ''.join(c.lower() for c in word if c.isalnum())
    
    def calculate_word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate similarity between two words for alignment purposes.
        """
        # Handle empty strings
        if not word1 or not word2:
            return 0.0
        
        # Normalize both words
        norm1 = self.normalize_word_for_matching(word1)
        norm2 = self.normalize_word_for_matching(word2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Prefix match (handles "we" matching "we'll" or "well")
        if norm1.startswith(norm2) or norm2.startswith(norm1):
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return 0.7 + (0.2 * shorter / longer)
        
        # Character-level Levenshtein ratio
        return self._calculate_levenshtein_ratio(norm1, norm2)
    
    def _calculate_levenshtein_ratio(self, s1: str, s2: str) -> float:
        """
        Calculate Levenshtein distance ratio between two strings.
        """
        len1, len2 = len(s1), len(s2)
        
        # Early exit for very different lengths
        if abs(len1 - len2) > max(len1, len2) // 2:
            return 0.0
        
        # Ensure s1 is the longer string for consistency
        if len1 < len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        # Initialize DP table
        prev_row = list(range(len2 + 1))
        
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            
            prev_row = curr_row
        
        distance = prev_row[-1]
        max_len = max(len1, len2)
        ratio = 1.0 - (distance / max_len)
        
        # Only return meaningful similarity for close matches
        return ratio if ratio >= self.min_similarity_threshold else 0.0
    
    def align_word_sequences(self, source_words: List[str], target_words: List[Dict[str, Any]]) -> List[tuple]:
        """
        Align source and target word sequences using dynamic programming.
        """
        if not source_words and not target_words:
            return []
        if not source_words:
            return [(None, i) for i in range(len(target_words))]
        if not target_words:
            return [(i, None) for i in range(len(source_words))]
        
        n = len(source_words)
        m = len(target_words)
        
        # Extract target texts for comparison
        target_texts = [wd["text"] for wd in target_words]
        
        # Initialize DP table and backpointer
        dp = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]
        backptr = [[(0, 0, '') for _ in range(m + 1)] for _ in range(n + 1)]
        
        # Initialize first row and column with gap penalties
        for i in range(1, n + 1):
            dp[i][0] = dp[i-1][0] + self.gap_penalty
            backptr[i][0] = (i-1, 0, 'source_gap')
        
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j-1] + self.gap_penalty
            backptr[0][j] = (0, j-1, 'target_gap')
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                source_word = source_words[i-1]
                target_word = target_texts[j-1]
                
                # Calculate similarity score
                similarity = self.calculate_word_similarity(source_word, target_word)
                
                # Consider three operations:
                # 1. Match/substitute
                match_score = dp[i-1][j-1] + similarity
                
                # 2. Gap in target (source has extra word)
                source_gap_score = dp[i-1][j] + self.gap_penalty
                
                # 3. Gap in source (target has extra word - likely a split)
                target_gap_score = dp[i][j-1] + self.gap_penalty
                
                # Choose best operation
                if match_score >= source_gap_score and match_score >= target_gap_score:
                    dp[i][j] = match_score
                    backptr[i][j] = (i-1, j-1, 'match')
                elif source_gap_score >= target_gap_score:
                    dp[i][j] = source_gap_score
                    backptr[i][j] = (i-1, j, 'source_gap')
                else:
                    dp[i][j] = target_gap_score
                    backptr[i][j] = (i, j-1, 'target_gap')
        
        # Traceback to find optimal alignment
        return self._traceback_alignment(backptr, n, m)
    
    def _traceback_alignment(self, backptr: List[List[Tuple[int, int, str]]], n: int, m: int) -> List[tuple]:
        """
        Perform traceback to construct alignment from DP table.
        """
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i == 0:
                # Remaining target words have no source match
                alignment.append((None, j-1))
                j -= 1
            elif j == 0:
                # Remaining source words have no target match
                alignment.append((i-1, None))
                i -= 1
            else:
                # Get the move that led to this cell
                prev_i, prev_j, move = backptr[i][j]
                
                if move == 'match':
                    alignment.append((i-1, j-1))
                    i -= 1
                    j -= 1
                elif move == 'source_gap':
                    alignment.append((i-1, None))
                    i -= 1
                else:  # target_gap
                    alignment.append((None, j-1))
                    j -= 1
        
        alignment.reverse()
        return alignment
    
    def interpolate_timestamps(self, prev_end: float, next_start: float, num_words: int,
                              words: List[str], speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create word dicts with interpolated timestamps for words that were missed.
        """
        if num_words == 0 or not words:
            return []
        
        # Calculate available time window
        available_duration = next_start - prev_end
        if available_duration <= 0:
            # No time available, use minimum duration per word
            word_duration = self.min_word_duration
            available_duration = max(word_duration * num_words, self.min_word_duration)
        
        # Distribute time evenly among words
        word_duration = available_duration / num_words
        
        result = []
        current_time = prev_end
        
        for word in words:
            word_end = current_time + word_duration
            result.append({
                "text": word,
                "start": current_time,
                "end": word_end,
                "speaker": speaker
            })
            current_time = word_end
        
        return result
    
    def replace_words_with_original_text(self, word_dicts: List[Dict[str, Any]],
                                        original_transcript: str,
                                        segment_start_time: float,
                                        segment_end_time: float,
                                        speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Replace alignment engine words with original text, using alignment timestamps.
        """
        self.logger.info("Replacing alignment words with original text")
        
        original_words = original_transcript.split()
        mfa_count = len(word_dicts)
        original_count = len(original_words)
        
        self.logger.debug(f"Original word count: {original_count}, Alignment word count: {mfa_count}")
        
        # Fast path: word counts match - direct positional replacement
        if mfa_count == original_count:
            self.logger.debug("Word counts match - using direct positional replacement")
            return self._create_direct_replacement(word_dicts, original_words, speaker)
        
        # Complex case: counts differ - need sequence alignment
        self.logger.debug(f"Word counts differ ({original_count} vs {mfa_count}) - using sequence alignment")
        alignment = self.align_word_sequences(original_words, word_dicts)
        
        self.logger.debug(f"Alignment computed with {len(alignment)} entries")
        
        # Process alignment to build result
        return self._process_alignment_result(alignment, word_dicts, original_words,
                                            segment_start_time, segment_end_time, speaker)
    
    def _create_direct_replacement(self, word_dicts: List[Dict[str, Any]],
                                  original_words: List[str],
                                  speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create direct positional replacement when word counts match.
        """
        result = []
        for i, word_dict in enumerate(word_dicts):
            result.append({
                "text": original_words[i],
                "start": word_dict["start"],
                "end": word_dict["end"],
                "speaker": word_dict.get("speaker", speaker)
            })
        return result
    
    def _process_alignment_result(self, alignment: List[tuple], word_dicts: List[Dict[str, Any]],
                                 original_words: List[str], segment_start_time: float,
                                 segment_end_time: float, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process alignment result to handle gaps and merges.
        """
        result = []
        pending_original_words = []  # Words waiting for timestamps
        last_end_time = segment_start_time
        
        # Group alignment by original words
        i = 0
        while i < len(alignment):
            orig_idx, mfa_idx = alignment[i]
            
            if orig_idx is not None and mfa_idx is not None:
                # Matched pair - first handle any pending words
                if pending_original_words:
                    current_start = word_dicts[mfa_idx]["start"]
                    interpolated = self.interpolate_timestamps(
                        last_end_time, current_start,
                        len(pending_original_words), pending_original_words, speaker
                    )
                    result.extend(interpolated)
                    pending_original_words = []
                
                # Check if next alignments are target gaps that should merge into this original word
                mfa_start_idx = mfa_idx
                mfa_end_idx = mfa_idx + 1
                
                j = i + 1
                while j < len(alignment):
                    next_orig, next_mfa = alignment[j]
                    if next_orig is None and next_mfa is not None:
                        # Target extra word - merge it
                        mfa_end_idx = next_mfa + 1
                        j += 1
                    else:
                        break
                
                # Get merged timestamps
                start_time, end_time = self._merge_mfa_timestamps(word_dicts, mfa_start_idx, mfa_end_idx)
                
                result.append({
                    "text": original_words[orig_idx],
                    "start": start_time,
                    "end": end_time,
                    "speaker": word_dicts[mfa_idx].get("speaker", speaker)
                })
                
                last_end_time = end_time
                i = j  # Skip merged target words
                
            elif orig_idx is not None and mfa_idx is None:
                # Original word with no target match - queue for interpolation
                pending_original_words.append(original_words[orig_idx])
                i += 1
                
            elif orig_idx is None and mfa_idx is not None:
                # Target extra word with no original match at this position
                self.logger.debug(f"Unexpected target extra word at position {mfa_idx}: '{word_dicts[mfa_idx]['text']}'")
                i += 1
                
            else:
                # Both None - shouldn't happen
                i += 1
        
        # Handle any remaining pending words at the end
        if pending_original_words:
            interpolated = self.interpolate_timestamps(
                last_end_time, segment_end_time,
                len(pending_original_words), pending_original_words, speaker
            )
            result.extend(interpolated)
        
        # Final validation
        if len(result) != len(original_words):
            self.logger.warning(
                f"Word count mismatch after alignment: expected {len(original_words)}, got {len(result)}. "
                "Falling back to simple distribution."
            )
            return self.create_simple_alignment(" ".join(original_words), segment_start_time,
                                              segment_end_time - segment_start_time, speaker, None)
        
        return result
    
    def _merge_mfa_timestamps(self, mfa_words: List[Dict[str, Any]], start_idx: int, end_idx: int) -> tuple:
        """
        Get merged start/end times from a range of MFA words.
        """
        if start_idx >= end_idx or start_idx >= len(mfa_words):
            return (0.0, 0.0)
        
        start_time = mfa_words[start_idx]["start"]
        end_time = mfa_words[min(end_idx - 1, len(mfa_words) - 1)]["end"]
        
        return (start_time, end_time)