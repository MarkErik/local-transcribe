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
    
    # Interjection patterns by type
    interjection_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "acknowledgment": [
            "yeah", "yea", "yep", "yup", "yes",
            "uh-huh", "uh huh", "uhuh",
            "mm-hmm", "mm hmm", "mmhmm", "mhm", "mm", "hmm", "hm",
            "right", "okay", "ok", "sure", "gotcha", "got it",
            "i see", "i know", "true", "exactly", "absolutely",
            "definitely", "totally", "for sure"
        ],
        "question": [
            "what", "why", "how", "really", "huh", "pardon",
            "sorry", "excuse me", "come again"
        ],
        "reaction": [
            "wow", "oh", "ah", "whoa", "nice", "cool", 
            "interesting", "amazing", "great", "awesome",
            "no way", "seriously", "oh my", "oh god", "oh no"
        ],
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_interjection_duration": self.max_interjection_duration,
            "max_interjection_words": self.max_interjection_words,
            "max_gap_to_merge_turns": self.max_gap_to_merge_turns,
            "interjection_patterns": self.interjection_patterns
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
    interjection_confidence: float = 0.0
    interjection_type: str = "unclear"
    interrupt_level: str = "none"
    classification_method: str = "unclassified"
    
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
            "interjection_confidence": self.interjection_confidence,
            "interjection_type": self.interjection_type,
            "interrupt_level": self.interrupt_level,
            "classification_method": self.classification_method,
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
    
    # Classification details
    confidence: float  # 0-1, how confident we are this is an interjection
    interjection_type: str  # "acknowledgment", "question", "reaction", "unclear"
    interrupt_level: str  # "none", "low", "medium", "high"
    classification_method: str  # "rule", "llm", "hybrid"
    
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
            "confidence": round(self.confidence, 3),
            "interjection_type": self.interjection_type,
            "interrupt_level": self.interrupt_level,
            "classification_method": self.classification_method,
            "likely_diarization_error": self.likely_diarization_error
        }


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
    
    # Metrics (calculated after construction)
    flow_continuity: float = 1.0  # 0-1, how uninterrupted (1.0 = no interjections)
    turn_type: str = "monologue"  # "monologue", "acknowledged", "interrupted"
    word_count: int = 0
    duration: float = 0.0
    speaking_rate: float = 0.0  # words per minute
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate word count, duration, speaking rate, and flow continuity."""
        self.word_count = len(self.words)
        self.duration = round(self.end - self.start, 3)
        
        # Calculate speaking rate (words per minute)
        if self.duration > 0:
            self.speaking_rate = round((self.word_count / self.duration) * 60, 1)
        else:
            self.speaking_rate = 0.0
        
        # Calculate flow continuity based on interjections
        self._calculate_flow_continuity()
        
        # Determine turn type
        self._determine_turn_type()
    
    def _calculate_flow_continuity(self):
        """
        Calculate flow continuity (0-1) based on interjections.
        
        1.0 = completely uninterrupted
        Lower values indicate more interruptions
        """
        if not self.interjections or self.duration == 0:
            self.flow_continuity = 1.0
            return
        
        # Calculate total interjection duration
        interjection_duration = sum(ij.duration for ij in self.interjections)
        
        # Primary ratio: how much of the turn duration is actual primary speech
        primary_ratio = max(0, (self.duration - interjection_duration) / self.duration)
        
        # Penalty for number of interjections (each one disrupts flow somewhat)
        interjection_penalty = min(len(self.interjections) * 0.1, 0.5)
        
        # Additional penalty for high interrupt levels
        high_interrupt_penalty = sum(
            0.1 for ij in self.interjections 
            if ij.interrupt_level in ("medium", "high")
        )
        high_interrupt_penalty = min(high_interrupt_penalty, 0.3)
        
        self.flow_continuity = round(
            max(0, primary_ratio * (1 - interjection_penalty) - high_interrupt_penalty),
            3
        )
    
    def _determine_turn_type(self):
        """Determine turn type based on interjections."""
        if not self.interjections:
            self.turn_type = "monologue"
        elif any(ij.interrupt_level in ("medium", "high") for ij in self.interjections):
            self.turn_type = "interrupted"
        else:
            self.turn_type = "acknowledged"
    
    def recalculate_metrics(self):
        """Recalculate all metrics. Call after modifying interjections."""
        self._calculate_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_id": self.turn_id,
            "primary_speaker": self.primary_speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "word_count": self.word_count,
            "duration": self.duration,
            "speaking_rate": self.speaking_rate,
            "flow_continuity": self.flow_continuity,
            "turn_type": self.turn_type,
            "interjections": [ij.to_dict() for ij in self.interjections]
        }


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
    
    def __repr__(self) -> str:
        return (
            f"TranscriptFlow(turns={len(self.turns)}, "
            f"interjections={self.total_interjections}, "
            f"speakers={self.speakers}, "
            f"duration={self.duration:.1f}s)"
        )
