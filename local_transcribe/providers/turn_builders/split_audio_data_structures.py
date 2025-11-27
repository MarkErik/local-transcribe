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
    classification_method: str = "unclassified"
    
    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end - self.start
    
    @property
    def word_count(self) -> int:
        """Number of words in this segment."""
        return len(self.words)


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
            "duration": self.duration,
            "confidence": self.confidence,
            "interjection_type": self.interjection_type,
            "interrupt_level": self.interrupt_level,
            "classification_method": self.classification_method
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
        self.word_count = len(self.words)
        self.duration = self.end - self.start
        if self.duration > 0:
            self.speaking_rate = (self.word_count / self.duration) * 60
        self._calculate_flow_continuity()
        self._determine_turn_type()
    
    def _calculate_flow_continuity(self):
        """Calculate how uninterrupted this turn is (0-1)."""
        if not self.interjections or self.duration <= 0:
            self.flow_continuity = 1.0
            return
        
        # Calculate total interjection time
        interjection_time = sum(ij.duration for ij in self.interjections)
        
        # Flow continuity is the ratio of primary speaker time
        primary_time = self.duration - interjection_time
        self.flow_continuity = max(0.0, min(1.0, primary_time / self.duration))
    
    def _determine_turn_type(self):
        """Determine turn type based on interjections."""
        if not self.interjections:
            self.turn_type = "monologue"
        elif any(ij.interrupt_level in ("medium", "high") for ij in self.interjections):
            self.turn_type = "interrupted"
        else:
            self.turn_type = "acknowledged"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_id": self.turn_id,
            "primary_speaker": self.primary_speaker,
            "start": self.start,
            "end": self.end,
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
    Complete transcript with hierarchical structure and metrics.
    
    This is the primary output format for turn builders, representing the full
    structure of an interview conversation including:
    - Hierarchical turns with embedded interjections
    - Conversation-level metrics
    - Speaker statistics
    
    This format preserves the rich conversational dynamics rather than
    flattening to a simple list of turns.
    """
    turns: List[HierarchicalTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    speaker_statistics: Dict[str, Any] = field(default_factory=dict)
    conversation_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_turns(self) -> int:
        """Total number of primary turns."""
        return len(self.turns)
    
    @property
    def total_interjections(self) -> int:
        """Total number of interjections across all turns."""
        return sum(len(t.interjections) for t in self.turns)
    
    @property
    def speakers(self) -> List[str]:
        """List of unique speakers in the transcript."""
        speakers = set()
        for turn in self.turns:
            speakers.add(turn.primary_speaker)
            for ij in turn.interjections:
                speakers.add(ij.speaker)
        return sorted(speakers)
    
    @property
    def duration(self) -> float:
        """Total duration of the transcript in seconds."""
        if not self.turns:
            return 0.0
        return self.turns[-1].end - self.turns[0].start
    
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


@dataclass
class TurnBuilderConfig:
    """
    Configuration for turn building algorithms.
    
    These parameters control how segments are classified as interjections
    vs primary turns, and how turns are assembled from segments.
    """
    # Duration thresholds
    max_interjection_duration: float = 2.0  # seconds - segments longer than this are never interjections
    max_interjection_words: int = 5  # segments with more words are never interjections
    min_primary_turn_duration: float = 0.3  # seconds - very short turns may be interjections
    
    # Gap analysis
    max_gap_before_interjection: float = 1.0  # seconds - interjections occur near other speech
    max_gap_to_merge_turns: float = 3.0  # seconds - same speaker turns within this gap are merged
    
    # Overlap detection and timestamp tolerance
    overlap_tolerance: float = 0.1  # seconds - timing imprecision buffer
    timestamp_tolerance: float = 0.5  # seconds - tolerance window for split-audio timestamp misalignment
    
    # Interjection patterns (lowercase for matching)
    acknowledgment_patterns: List[str] = field(default_factory=lambda: [
        "yeah", "yea", "yep", "yes", "uh-huh", "uh huh", "mm-hmm",
        "mm hmm", "mhm", "right", "okay", "ok", "sure", "got it",
        "i see", "makes sense", "exactly", "absolutely", "definitely",
        "totally", "true", "correct", "that", "this", "well", "so",
        "like", "um", "uh", "ah", "oh", "hm", "mmm"
    ])
    question_patterns: List[str] = field(default_factory=lambda: [
        "what", "why", "how", "when", "where", "really", "huh",
        "seriously", "is that right", "you think so", "you mean"
    ])
    reaction_patterns: List[str] = field(default_factory=lambda: [
        "wow", "oh", "ah", "hmm", "interesting", "cool", "nice",
        "great", "amazing", "oh my", "no way", "whoa", "sorry"
    ])
    # Incomplete/fragment patterns - very short utterances that are likely interjections
    fragment_patterns: List[str] = field(default_factory=lambda: [
        "like to", "to me", "is that", "this is", "kind of", "that's",
        "i mean", "you know", "them", "to try", "try like", "like what"
    ])
    
    # Classification thresholds
    high_confidence_threshold: float = 0.55  # Above this, classify as interjection
    low_confidence_threshold: float = 0.25   # Below this, definitely not interjection
    
    # Very short segment handling - segments this short are almost always interjections
    very_short_word_count: int = 2  # Segments with <= this many words get boosted confidence
    very_short_duration: float = 1.0  # Segments shorter than this get boosted confidence
    
    # LLM settings (for LLM-enhanced variant)
    llm_url: Optional[str] = None
    llm_timeout: int = 120  # seconds
    llm_context_turns: int = 2  # How many surrounding turns to include in context
    llm_confidence_threshold: float = 0.5  # Use LLM if rule confidence is between low and high
    parse_harmony: bool = True  # Parse Harmony format responses
    temperature: float = 0.3  # Low temperature for consistent classification
    max_retries: int = 3  # Number of retries on validation failure
    temperature_decay: float = 0.1  # Reduce temperature by this much on each retry
