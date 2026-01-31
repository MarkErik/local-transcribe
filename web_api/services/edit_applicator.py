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
            "toggle_interjection": self._apply_toggle_interjection,
            "insert_annotation": self._apply_insert_annotation,
            "turn_merge": self._apply_turn_merge,
            "turn_split": self._apply_turn_split,
            "pii_redact": self._apply_pii_redact,
            "pii_unredact": self._apply_pii_unredact,
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
            })
        
        # Replace original word with split words
        words[edit.start_index:edit.start_index + 1] = split_words
    
    def _apply_toggle_interjection(self, edit: Edit) -> None:
        """
        Toggle a turn between primary turn and interjection status.
        
        Uses: turn_id (turn to convert), target_turn_id (parent turn if converting to interjection)
        
        When converting TO interjection:
        - Remove turn from turns list
        - Add as interjection to target_turn_id
        
        When converting FROM interjection:
        - Remove interjection from parent turn
        - Create new turn in turns list
        """
        turn = self._find_turn(edit.turn_id)
        
        if turn:
            # Converting primary turn to interjection
            if edit.target_turn_id is not None:
                target_turn = self._find_turn(edit.target_turn_id)
                if target_turn:
                    # Create interjection from turn
                    interjection = {
                        "speaker": turn.get("primary_speaker", turn.get("speaker", "Unknown")),
                        "text": turn.get("text", ""),
                        "start_time": turn.get("start_time", 0.0),
                        "end_time": turn.get("end_time", 0.0),
                        "words": turn.get("words", []),
                    }
                    
                    # Add to target turn's interjections
                    if "interjections" not in target_turn:
                        target_turn["interjections"] = []
                    target_turn["interjections"].append(interjection)
                    
                    # Remove from turns list
                    self.turns = [t for t in self.turns if t.get("turn_id") != edit.turn_id]
                    self.transcript["turns"] = self.turns
        else:
            # Converting interjection back to primary turn
            # Find the interjection in all turns
            for t in self.turns:
                interjections = t.get("interjections", [])
                for i, interj in enumerate(interjections):
                    # Match by start_time since interjections don't have turn_id
                    if abs(interj.get("start_time", 0) - (edit.start_index or 0)) < 0.01:
                        # Create new turn from interjection
                        new_turn_id = max(t.get("turn_id", 0) for t in self.turns) + 1
                        new_turn = {
                            "turn_id": new_turn_id,
                            "primary_speaker": interj.get("speaker", "Unknown"),
                            "speaker": interj.get("speaker", "Unknown"),
                            "text": interj.get("text", ""),
                            "start_time": interj.get("start_time", 0.0),
                            "end_time": interj.get("end_time", 0.0),
                            "words": interj.get("words", []),
                            "interjections": [],
                        }
                        
                        # Remove interjection
                        del interjections[i]
                        
                        # Insert turn in correct position by start_time
                        insert_idx = 0
                        for idx, existing in enumerate(self.turns):
                            if existing.get("start_time", 0) > new_turn["start_time"]:
                                insert_idx = idx
                                break
                            insert_idx = idx + 1
                        
                        self.turns.insert(insert_idx, new_turn)
                        self.transcript["turns"] = self.turns
                        return
    
    def _apply_insert_annotation(self, edit: Edit) -> None:
        """
        Insert an annotation marker at a specific word position.
        
        Uses: turn_id, start_index, annotation_type
        
        Annotations are stored as special word segments with type marker.
        Common types: [laughter], [pause], [inaudible], [crosstalk]
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        insert_idx = edit.start_index if edit.start_index is not None else len(words)
        insert_idx = max(0, min(insert_idx, len(words)))
        
        # Create annotation word segment
        annotation_type = getattr(edit, 'annotation_type', None) or edit.new_value or "pause"
        annotation = {
            "word": f"[{annotation_type}]",
            "start_time": 0.0,
            "end_time": 0.0,
            "is_annotation": True,
            "annotation_type": annotation_type,
        }
        
        words.insert(insert_idx, annotation)
        self._recalculate_word_timing(turn)
        
        # Rebuild turn text
        turn["text"] = " ".join(w["word"] for w in words)
    
    def _apply_turn_merge(self, edit: Edit) -> None:
        """
        Merge two consecutive turns into one.
        
        Uses: turn_id (first turn), target_turn_id (second turn to merge into first)
        
        The second turn's content is appended to the first turn.
        """
        first_turn = self._find_turn(edit.turn_id)
        target_turn_id = getattr(edit, 'target_turn_id', None)
        if target_turn_id is None:
            return
        second_turn = self._find_turn(target_turn_id)
        
        if not first_turn or not second_turn:
            return
        
        # Merge words
        first_words = first_turn.get("words", [])
        second_words = second_turn.get("words", [])
        first_turn["words"] = first_words + second_words
        
        # Update timing
        first_turn["end_time"] = second_turn.get("end_time", first_turn.get("end_time", 0.0))
        
        # Rebuild text
        first_turn["text"] = " ".join(w["word"] for w in first_turn["words"])
        
        # Merge interjections
        first_interj = first_turn.get("interjections", [])
        second_interj = second_turn.get("interjections", [])
        first_turn["interjections"] = first_interj + second_interj
        
        # Merge source block IDs if present
        first_blocks = first_turn.get("source_block_ids", [])
        second_blocks = second_turn.get("source_block_ids", [])
        if first_blocks or second_blocks:
            first_turn["source_block_ids"] = first_blocks + second_blocks
        
        # Remove second turn
        self.turns = [t for t in self.turns if t.get("turn_id") != target_turn_id]
        self.transcript["turns"] = self.turns
    
    def _apply_turn_split(self, edit: Edit) -> None:
        """
        Split a turn at a specific word index.
        
        Uses: turn_id, start_index (word index where split occurs)
        
        Creates two turns: words [0:start_index) and [start_index:]
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        split_idx = edit.start_index if edit.start_index is not None else len(words) // 2
        
        if split_idx <= 0 or split_idx >= len(words):
            return  # Can't split at edges
        
        # Words for each part
        first_words = words[:split_idx]
        second_words = words[split_idx:]
        
        if not first_words or not second_words:
            return
        
        # Create new turn ID
        new_turn_id = max(t.get("turn_id", 0) for t in self.turns) + 1
        
        # Update first turn
        turn["words"] = first_words
        turn["text"] = " ".join(w["word"] for w in first_words)
        turn["end_time"] = first_words[-1].get("end_time", turn.get("end_time", 0.0))
        
        # Create second turn
        second_turn = {
            "turn_id": new_turn_id,
            "primary_speaker": turn.get("primary_speaker", turn.get("speaker", "Unknown")),
            "speaker": turn.get("speaker", turn.get("primary_speaker", "Unknown")),
            "text": " ".join(w["word"] for w in second_words),
            "start_time": second_words[0].get("start_time", turn.get("end_time", 0.0)),
            "end_time": words[-1].get("end_time", turn.get("end_time", 0.0)),
            "words": second_words,
            "interjections": [],  # Interjections stay with first turn
        }
        
        # Insert after original turn
        turn_idx = self.turns.index(turn)
        self.turns.insert(turn_idx + 1, second_turn)
        self.transcript["turns"] = self.turns
    
    def _apply_pii_redact(self, edit: Edit) -> None:
        """
        Redact word(s) as PII, replacing with [NAME] or custom placeholder.
        
        Uses: turn_id, start_index, end_index, new_value (replacement text)
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        start = edit.start_index if edit.start_index is not None else 0
        end = edit.end_index if edit.end_index is not None else start
        
        replacement_text = edit.new_value or "[NAME]"
        
        # Replace each word in range with the redaction placeholder
        for i in range(start, min(end + 1, len(words))):
            # Store original for audit trail (in a metadata field if needed)
            original_text = words[i].get("word", "")
            words[i]["word"] = replacement_text
            # Mark as redacted
            words[i]["is_redacted"] = True
            words[i]["original_text"] = original_text
        
        # Update turn text
        turn["text"] = " ".join(w["word"] for w in words)
    
    def _apply_pii_unredact(self, edit: Edit) -> None:
        """
        Restore original text that was incorrectly marked as PII.
        
        Uses: turn_id, start_index, end_index, original_value (original text to restore)
        """
        turn = self._find_turn(edit.turn_id)
        if not turn:
            return
        
        words = turn.get("words", [])
        start = edit.start_index if edit.start_index is not None else 0
        end = edit.end_index if edit.end_index is not None else start
        
        # Restore original text from edit or from stored metadata
        for i in range(start, min(end + 1, len(words))):
            # Try to get original from edit, or from word metadata
            original = edit.original_value
            if not original and words[i].get("original_text"):
                original = words[i]["original_text"]
            
            if original:
                words[i]["word"] = original
                words[i]["is_redacted"] = False
                # Keep original_text for audit trail
        
        # Update turn text
        turn["text"] = " ".join(w["word"] for w in words)
    
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
