"""
Edit applicator service.

Applies stored edits to TranscriptFlow objects to produce edited transcripts.
Edits are stored in the database and applied on-demand when retrieving transcripts.
"""

import copy
from typing import List, Dict, Any, Optional

from web_api.database import Edit


class EditApplicator:
    """
    Applies edits to TranscriptFlow transcripts.
    
    Supports the following edit types:
    - word_change: Change a word's text at a specific index
    - word_insert: Insert word(s) at a specific index
    - word_delete: Remove word(s) at a specific index range
    - speaker_change: Change the speaker for a turn
    - merge_words: Merge words at specified indices
    - split_word: Split a word into multiple words
    """
    
    def __init__(self, transcript_data: Dict[str, Any]):
        """
        Initialize with transcript data.
        
        Args:
            transcript_data: TranscriptFlow as dictionary (from JSON)
        """
        self.transcript = copy.deepcopy(transcript_data)
        self.turns = self.transcript.get("turns", [])
    
    def apply_edits(self, edits: List[Edit]) -> Dict[str, Any]:
        """
        Apply a list of edits in order.
        
        Args:
            edits: List of Edit objects to apply (in order)
            
        Returns:
            Modified transcript data
        """
        for edit in edits:
            self._apply_single_edit(edit)
        
        return self.transcript
    
    def _apply_single_edit(self, edit: Edit) -> None:
        """Apply a single edit to the transcript."""
        handler = {
            "word_change": self._apply_word_change,
            "word_insert": self._apply_word_insert,
            "word_delete": self._apply_word_delete,
            "speaker_change": self._apply_speaker_change,
            "merge_words": self._apply_merge_words,
            "split_word": self._apply_split_word,
        }.get(edit.edit_type)
        
        if handler:
            handler(edit)
        else:
            # Unknown edit type - skip with warning
            pass
    
    def _find_turn(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """Find a turn by its turn_id."""
        for turn in self.turns:
            if turn.get("turn_id") == turn_id:
                return turn
        return None
    
    def _apply_word_change(self, edit: Edit) -> None:
        """
        Change a word's text at a specific position.
        
        Uses: turn_id, start_index, new_value
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        if edit.start_index is not None and 0 <= edit.start_index < len(words):
            words[edit.start_index]["word"] = edit.new_value
    
    def _apply_word_insert(self, edit: Edit) -> None:
        """
        Insert word(s) at a specific position.
        
        Uses: turn_id, start_index, new_value (space-separated words)
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        insert_idx = edit.start_index if edit.start_index is not None else len(words)
        
        # Clamp to valid range
        insert_idx = max(0, min(insert_idx, len(words)))
        
        # Create new word segments for inserted words
        new_words = []
        for word_text in (edit.new_value or "").split():
            # Create a minimal word segment
            # Timing will be approximate - we can't know exact timestamps
            new_word = {
                "word": word_text,
                "start_time": 0.0,  # Will be recalculated
                "end_time": 0.0,
                "confidence": 1.0,  # User-inserted words have full confidence
            }
            new_words.append(new_word)
        
        # Insert the new words
        for i, new_word in enumerate(new_words):
            words.insert(insert_idx + i, new_word)
        
        # Recalculate word timing within turn (distribute evenly)
        self._recalculate_word_timing(turn)
    
    def _apply_word_delete(self, edit: Edit) -> None:
        """
        Delete word(s) from a specific range.
        
        Uses: turn_id, start_index, end_index
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        start = edit.start_index if edit.start_index is not None else 0
        end = edit.end_index if edit.end_index is not None else start
        
        # Delete words in range (inclusive)
        if 0 <= start <= end < len(words):
            del words[start:end + 1]
        
        # Recalculate timing
        self._recalculate_word_timing(turn)
    
    def _apply_speaker_change(self, edit: Edit) -> None:
        """
        Change the speaker for a turn.
        
        Uses: turn_id, new_value (new speaker name/ID)
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        turn["speaker"] = edit.new_value
    
    def _apply_merge_words(self, edit: Edit) -> None:
        """
        Merge consecutive words into a single word.
        
        Uses: turn_id, start_index, end_index
        Example: "to" "gether" -> "together"
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        start = edit.start_index if edit.start_index is not None else 0
        end = edit.end_index if edit.end_index is not None else start + 1
        
        if not (0 <= start < end < len(words)):
            return
        
        # Merge word texts
        merged_text = "".join(w["word"] for w in words[start:end + 1])
        
        # Keep timing from first and last word
        merged_word = {
            "word": merged_text,
            "start_time": words[start].get("start_time", 0.0),
            "end_time": words[end].get("end_time", 0.0),
            "confidence": min(w.get("confidence", 1.0) for w in words[start:end + 1]),
        }
        
        # Replace range with merged word
        words[start:end + 1] = [merged_word]
    
    def _apply_split_word(self, edit: Edit) -> None:
        """
        Split a word into multiple words.
        
        Uses: turn_id, start_index, new_value (space-separated split words)
        Example: "cannot" -> "can not"
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        if edit.start_index is None or edit.start_index >= len(words):
            return
        
        original = words[edit.start_index]
        split_texts = (edit.new_value or "").split()
        
        if len(split_texts) < 2:
            return  # Nothing to split
        
        # Calculate timing for split words
        start_time = original.get("start_time", 0.0)
        end_time = original.get("end_time", 0.0)
        duration = end_time - start_time
        duration_per_word = duration / len(split_texts) if duration > 0 else 0
        
        split_words = []
        for i, text in enumerate(split_texts):
            split_words.append({
                "word": text,
                "start_time": start_time + i * duration_per_word,
                "end_time": start_time + (i + 1) * duration_per_word,
                "confidence": original.get("confidence", 1.0),
            })
        
        # Replace original word with split words
        words[edit.start_index:edit.start_index + 1] = split_words
    
    def _recalculate_word_timing(self, turn: Dict[str, Any]) -> None:
        """
        Recalculate word timing within a turn after edits.
        
        Distributes timing evenly across words within the turn's time span.
        This is approximate for VAD mode where we don't have true word alignment.
        """
        words = turn.get("words", [])
        if not words:
            return
        
        start_time = turn.get("start_time", 0.0)
        end_time = turn.get("end_time", 0.0)
        
        # If turn doesn't have timing, try to get from first/last word
        if start_time == 0 and end_time == 0 and len(words) > 0:
            start_time = words[0].get("start_time", 0.0)
            end_time = words[-1].get("end_time", 0.0)
        
        if end_time <= start_time:
            return  # Can't calculate without valid timing
        
        duration = end_time - start_time
        duration_per_word = duration / len(words)
        
        for i, word in enumerate(words):
            word["start_time"] = start_time + i * duration_per_word
            word["end_time"] = start_time + (i + 1) * duration_per_word
    
    def get_transcript(self) -> Dict[str, Any]:
        """Get the current (possibly edited) transcript."""
        return self.transcript


def apply_edits_to_transcript(
    transcript_data: Dict[str, Any], 
    edits: List[Edit],
) -> Dict[str, Any]:
    """
    Convenience function to apply edits to a transcript.
    
    Args:
        transcript_data: TranscriptFlow as dictionary
        edits: List of edits to apply
        
    Returns:
        Modified transcript data
    """
    applicator = EditApplicator(transcript_data)
    return applicator.apply_edits(edits)
