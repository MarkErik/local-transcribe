#!/usr/bin/env python3
"""
Data structures for hierarchical turn building in split-audio mode.

This module defines the core data classes used for representing interview-style
conversations with primary turns and interjections (brief acknowledgments,
questions, and reactions that don't claim the conversational floor).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from local_transcribe.framework.plugin_interfaces import WordSegment, Turn


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
    
    def to_turn(self) -> Turn:
        """Convert to a flat Turn object for output writers."""
        return Turn(
            speaker=self.primary_speaker,
            start=self.start,
            end=self.end,
            text=self.text
        )
    
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
class EnrichedTranscript:
    """
    Complete transcript with hierarchical structure and metrics.
    
    This is the full representation of an interview conversation,
    including all turns, interjections, and conversation-level statistics.
    """
    turns: List[HierarchicalTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    speaker_statistics: Dict[str, Any] = field(default_factory=dict)
    conversation_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_flat_turns(self) -> List[Turn]:
        """
        Convert to a flat list of Turn objects for existing output writers.
        
        This flattens the hierarchical structure by converting each
        HierarchicalTurn to a Turn, and optionally interleaving interjections.
        """
        flat_turns = []
        for hturn in self.turns:
            # Add the primary turn
            flat_turns.append(hturn.to_turn())
        return flat_turns
    
    def to_flat_turns_with_interjections(self) -> List[Turn]:
        """
        Convert to flat turns, interleaving interjections in chronological order.
        
        This produces a more detailed flat representation where interjections
        appear as separate turns in their correct temporal position.
        """
        all_items = []
        
        for hturn in self.turns:
            # Collect the primary turn and all its interjections
            all_items.append(("turn", hturn.start, hturn))
            for ij in hturn.interjections:
                all_items.append(("interjection", ij.start, ij))
        
        # Sort by start time
        all_items.sort(key=lambda x: x[1])
        
        # Convert to Turn objects
        flat_turns = []
        for item_type, _, item in all_items:
            if item_type == "turn":
                flat_turns.append(item.to_turn())
            else:
                # Interjection becomes a small turn
                flat_turns.append(Turn(
                    speaker=item.speaker,
                    start=item.start,
                    end=item.end,
                    text=item.text
                ))
        
        return flat_turns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata,
            "speaker_statistics": self.speaker_statistics,
            "conversation_metrics": self.conversation_metrics
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
    
    # Overlap detection
    overlap_tolerance: float = 0.1  # seconds - timing imprecision buffer
    
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
