#!/usr/bin/env python3
"""
Data structures for hierarchical turn building.

This module defines the core data classes used for representing interview-style
conversations with primary turns and interjections (brief acknowledgments,
questions, and reactions that don't claim the conversational floor).

The primary output format is TranscriptFlow, which preserves the full
hierarchical structure of conversations including:
- HierarchicalTurn: Primary speaking turns with embedded interjections
- InterjectionSegment: Brief utterances that don't claim the floor
- Conversation metrics and speaker statistics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from local_transcribe.framework.plugin_interfaces import WordSegment


@dataclass
class TurnBuilderConfig:
    """
    Configuration for the turn building algorithm.
    
    These thresholds control how segments are classified as interjections
    vs. primary turns, and how turns are merged.
    """
    # Interjection detection thresholds
    max_interjection_duration: float = 2.0  # seconds
    max_interjection_words: int = 5
    
    # Turn merging threshold
    max_gap_to_merge_turns: float = 3.0  # seconds - merge same-speaker turns if gap is smaller
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_interjection_duration": self.max_interjection_duration,
            "max_interjection_words": self.max_interjection_words,
            "max_gap_to_merge_turns": self.max_gap_to_merge_turns,
        }


@dataclass
class RawSegment:
    """
    A contiguous segment of words from a single speaker.
    
    This is an intermediate representation used during turn building,
    before classification as primary turn or interjection.
    """
    speaker: str
    start: float
    end: float
    text: str
    words: List[WordSegment]
    
    # Gap information (set during grouping)
    gap_before: Optional[float] = None  # Gap from previous segment
    gap_after: Optional[float] = None   # Gap to next segment
    
    # Classification results (set during analysis)
    is_interjection: Optional[bool] = None
    
    # Flag for potential diarization errors
    likely_diarization_error: bool = False
    
    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start
    
    @property
    def word_count(self) -> int:
        """Number of words in this segment."""
        return len(self.words)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "word_count": self.word_count,
            "duration": self.duration,
            "gap_before": self.gap_before,
            "gap_after": self.gap_after,
            "is_interjection": self.is_interjection,
            "likely_diarization_error": self.likely_diarization_error
        }


@dataclass
class InterjectionSegment:
    """
    A brief utterance that doesn't claim the conversational floor.
    
    Interjections are typically acknowledgments ("yeah", "uh-huh"),
    brief questions ("really?"), or reactions ("wow", "oh").
    """
    speaker: str
    start: float
    end: float
    text: str
    words: List[WordSegment]
    
    # Flag for potential diarization errors
    likely_diarization_error: bool = False
    
    @property
    def duration(self) -> float:
        """Duration of this interjection in seconds."""
        return self.end - self.start
    
    @property
    def word_count(self) -> int:
        """Number of words in this interjection."""
        return len(self.words)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "word_count": self.word_count,
            "duration": round(self.duration, 3),
            "likely_diarization_error": self.likely_diarization_error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterjectionSegment":
        """
        Create InterjectionSegment from dictionary (JSON deserialization).
        
        Args:
            data: Dictionary representation of an InterjectionSegment
            
        Returns:
            InterjectionSegment instance
        """
        # Reconstruct WordSegment list from serialized words if present
        # Note: InterjectionSegment.to_dict() doesn't serialize words array,
        # so we create empty list or reconstruct from text if needed
        words = []
        if "words" in data:
            words = [
                WordSegment(
                    text=w["text"],
                    start=float(w["start"]),
                    end=float(w["end"]),
                    speaker=w.get("speaker")
                )
                for w in data["words"]
            ]
        
        return cls(
            speaker=data["speaker"],
            start=float(data["start"]),
            end=float(data["end"]),
            text=data["text"],
            words=words,
            likely_diarization_error=data.get("likely_diarization_error", False)
        )


@dataclass
class HierarchicalTurn:
    """
    A primary speaking turn with optional embedded interjections.
    
    This represents the natural flow of conversation where one speaker
    holds the floor while another may briefly interject without
    interrupting the main discourse.
    """
    turn_id: int
    primary_speaker: str
    start: float
    end: float
    text: str
    words: List[WordSegment]
    
    # Hierarchical elements
    interjections: List[InterjectionSegment] = field(default_factory=list)
    
    # Source tracking for VAD mode (block IDs that contributed to this turn)
    source_block_ids: List[int] = field(default_factory=list)
    
    # Metrics (calculated after construction)
    word_count: int = 0
    duration: float = 0.0
    speaking_rate: float = 0.0  # words per minute
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate word count, duration, and speaking rate."""
        self.word_count = len(self.words)
        self.duration = round(self.end - self.start, 3)
        
        # Calculate speaking rate (words per minute)
        if self.duration > 0:
            self.speaking_rate = round((self.word_count / self.duration) * 60, 1)
        else:
            self.speaking_rate = 0.0
    
    def recalculate_metrics(self):
        """Recalculate all metrics. Call after modifying interjections."""
        self._calculate_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "turn_id": self.turn_id,
            "primary_speaker": self.primary_speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "word_count": self.word_count,
            "duration": self.duration,
            "speaking_rate": self.speaking_rate,
            "interjections": [ij.to_dict() for ij in self.interjections]
        }
        # Only include source_block_ids if present (VAD mode)
        if self.source_block_ids:
            result["source_block_ids"] = self.source_block_ids
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchicalTurn":
        """
        Create HierarchicalTurn from dictionary (JSON deserialization).
        
        Args:
            data: Dictionary representation of a HierarchicalTurn
            
        Returns:
            HierarchicalTurn instance
        """
        # Reconstruct WordSegment list
        words = [
            WordSegment(
                text=w["text"],
                start=float(w["start"]),
                end=float(w["end"]),
                speaker=w.get("speaker")
            )
            for w in data.get("words", [])
        ]
        
        # Reconstruct InterjectionSegment list
        interjections = [
            InterjectionSegment.from_dict(ij)
            for ij in data.get("interjections", [])
        ]
        
        # Create the turn (metrics will be calculated in __post_init__)
        turn = cls(
            turn_id=int(data["turn_id"]),
            primary_speaker=data["primary_speaker"],
            start=float(data["start"]),
            end=float(data["end"]),
            text=data["text"],
            words=words,
            interjections=interjections,
            source_block_ids=data.get("source_block_ids", [])
        )
        
        return turn


@dataclass
class TranscriptFlow:
    """
    Complete hierarchical transcript with conversation structure.
    
    This is the primary output format for turn building, containing:
    - Hierarchical turns with embedded interjections
    - Metadata about the transcript
    - Conversation-level metrics
    - Per-speaker statistics
    """
    turns: List[HierarchicalTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_metrics: Dict[str, Any] = field(default_factory=dict)
    speaker_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def total_turns(self) -> int:
        """Total number of primary turns."""
        return len(self.turns)
    
    @property
    def total_interjections(self) -> int:
        """Total number of interjections across all turns."""
        return sum(len(turn.interjections) for turn in self.turns)
    
    @property
    def duration(self) -> float:
        """Total duration of the transcript in seconds."""
        if not self.turns:
            return 0.0
        return self.turns[-1].end - self.turns[0].start
    
    @property
    def speakers(self) -> List[str]:
        """List of unique speakers in the transcript."""
        speaker_set = set()
        for turn in self.turns:
            speaker_set.add(turn.primary_speaker)
            for ij in turn.interjections:
                speaker_set.add(ij.speaker)
        return sorted(speaker_set)
    
    def get_turns_by_speaker(self, speaker: str) -> List[HierarchicalTurn]:
        """Get all turns by a specific speaker."""
        return [t for t in self.turns if t.primary_speaker == speaker]
    
    def get_interjections_by_speaker(self, speaker: str) -> List[InterjectionSegment]:
        """Get all interjections by a specific speaker."""
        interjections = []
        for turn in self.turns:
            interjections.extend([ij for ij in turn.interjections if ij.speaker == speaker])
        return interjections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata,
            "conversation_metrics": self.conversation_metrics,
            "speaker_statistics": self.speaker_statistics,
            "turns": [turn.to_dict() for turn in self.turns]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptFlow":
        """
        Create TranscriptFlow from dictionary (JSON deserialization).
        
        This is the inverse of to_dict() and enables loading edited
        transcripts from JSON checkpoint files for pipeline re-entry.
        
        Args:
            data: Dictionary representation of a TranscriptFlow
            
        Returns:
            TranscriptFlow instance
            
        Raises:
            KeyError: If required fields are missing
            ValueError: If data format is invalid
        """
        if "turns" not in data:
            raise KeyError("TranscriptFlow JSON must contain 'turns' array")
        
        turns = [
            HierarchicalTurn.from_dict(turn_data)
            for turn_data in data["turns"]
        ]
        
        return cls(
            turns=turns,
            metadata=data.get("metadata", {}),
            conversation_metrics=data.get("conversation_metrics", {}),
            speaker_statistics=data.get("speaker_statistics", {})
        )
    
    def __repr__(self) -> str:
        return (
            f"TranscriptFlow(turns={len(self.turns)}, "
            f"interjections={self.total_interjections}, "
            f"speakers={self.speakers}, "
            f"duration={self.duration:.1f}s)"
        )
