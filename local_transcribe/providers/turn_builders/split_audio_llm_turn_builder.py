#!/usr/bin/env python3
"""
Split audio turn builder provider using LLM-enhanced interjection detection.

This module implements a turn builder for split-audio mode that:
1. Merges word streams from multiple speakers into a unified timeline
2. Uses smart grouping that detects interjections DURING grouping (not after)
3. Handles timestamp misalignment with tolerance windows for split-audio mode
4. Uses LLM for semantic verification of ambiguous interjection candidates
5. Assembles hierarchical turns with embedded interjections
6. Returns TranscriptFlow with full hierarchical structure

Key improvements over naive grouping:
- Detects interjections during grouping phase, not after creating micro-segments
- Uses tolerance window for timestamp misalignment between separate audio tracks
- LLM semantic verification for ambiguous cases (not just confidence thresholds)
"""

import re
import json
import time
import requests
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, registry
from local_transcribe.lib.program_logger import log_progress, log_debug, log_intermediate_save

from local_transcribe.providers.turn_builders.split_audio_data_structures import (
    RawSegment,
    TurnBuilderConfig,
    TranscriptFlow,
    HierarchicalTurn,
    InterjectionSegment
)
from local_transcribe.providers.turn_builders.split_audio_base import (
    merge_word_streams,
    smart_group_with_interjection_detection,
    PendingInterjection,
    detect_interjection_type,
    assemble_hierarchical_turns_with_interjections,
    build_transcript_flow
)


class SplitAudioLLMTurnBuilderProvider(TurnBuilderProvider):
    """
    LLM-enhanced split audio turn builder.
    
    This turn builder uses smart grouping that detects interjections during
    the grouping phase rather than creating micro-segments. It handles the
    timestamp misalignment common in split-audio mode with tolerance windows.
    
    The LLM is used for semantic verification of ambiguous interjection
    candidates - cases where rule-based detection is uncertain about whether
    an utterance is an interjection or a real turn change.
    
    If the LLM is unavailable, falls back to rule-based classification.
    """

    def __init__(self):
        """Initialize with default configuration."""
        self.config = TurnBuilderConfig()
        self.llm_url = "http://100.84.208.72:8080"  # Default LLM endpoint
        self.llm_stats = {
            "calls_made": 0,
            "calls_succeeded": 0,
            "calls_failed": 0,
            "total_time_ms": 0,
            "verified_as_interjection": 0,
            "verified_as_turn": 0
        }

    @property
    def name(self) -> str:
        return "split_audio_llm_turn_builder"

    @property
    def short_name(self) -> str:
        return "Split Audio (LLM)"

    @property
    def description(self) -> str:
        return "LLM-enhanced turn builder for split audio mode with intelligent interjection detection"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> TranscriptFlow:
        """
        Build turns from word segments with speaker assignments.
        
        Uses smart grouping that detects interjections during the grouping
        phase, with LLM semantic verification for ambiguous cases.
        
        Args:
            words: Word segments with speaker assignments (from all speakers)
            **kwargs: Configuration options including:
                - llm_url: URL of the LLM server (default: http://100.84.208.72:8080)
                - intermediate_dir: Path to save intermediate files
                - max_interjection_duration: Override default (2.0s)
                - max_interjection_words: Override default (5)
                - max_gap_to_merge_turns: Override default (3.0s)
                - llm_timeout: Timeout for LLM requests in seconds
                - timestamp_tolerance: Tolerance window for timestamp misalignment (0.5s)
            
        Returns:
            TranscriptFlow with hierarchical turn structure
        """
        if not words:
            log_progress("No words provided to turn builder")
            return TranscriptFlow(turns=[], metadata={"builder": self.name, "error": "no_words"})
        
        # Update config from kwargs
        self._update_config_from_kwargs(kwargs)
        
        # Get LLM URL
        self.llm_url = kwargs.get('llm_url', self.llm_url)
        if not self.llm_url.startswith(('http://', 'https://')):
            self.llm_url = f'http://{self.llm_url}'
        
        # Reset LLM stats
        self.llm_stats = {
            "calls_made": 0, "calls_succeeded": 0, "calls_failed": 0, 
            "total_time_ms": 0, "verified_as_interjection": 0, "verified_as_turn": 0
        }
        
        # Get intermediate directory
        intermediate_dir = kwargs.get('intermediate_dir')
        
        log_progress(f"Building turns from {len(words)} word segments (LLM-enhanced)")
        log_progress(f"LLM endpoint: {self.llm_url}")
        
        # Step 1: Merge word streams into unified timeline
        log_debug("Step 1: Merging word streams")
        merged_words = merge_word_streams(words)
        log_progress(f"Merged into {len(merged_words)} words in timeline")
        
        # Step 2: Smart grouping with interjection detection during grouping
        log_debug("Step 2: Smart grouping with interjection detection")
        primary_segments, pending_interjections = smart_group_with_interjection_detection(
            merged_words, self.config
        )
        log_progress(f"Smart grouping: {len(primary_segments)} primary segments, "
                    f"{len(pending_interjections)} pending interjections")
        
        # Step 3: LLM semantic verification of ambiguous interjections
        log_debug("Step 3: LLM semantic verification of pending interjections")
        verified_interjections, promoted_to_turns = self._verify_interjections_with_llm(
            pending_interjections, primary_segments
        )
        
        log_progress(f"LLM verification: {len(verified_interjections)} confirmed interjections, "
                    f"{len(promoted_to_turns)} promoted to turns")
        log_progress(f"LLM stats: {self.llm_stats['calls_made']} calls, "
                    f"{self.llm_stats['calls_succeeded']} succeeded, "
                    f"{self.llm_stats['calls_failed']} failed")
        
        # Step 4: Merge promoted turns back into primary segments
        if promoted_to_turns:
            primary_segments = self._merge_promoted_turns(primary_segments, promoted_to_turns)
            log_debug(f"After merging promoted turns: {len(primary_segments)} primary segments")
        
        # Step 5: Assemble hierarchical turns with verified interjections
        log_debug("Step 5: Assembling hierarchical turns")
        hierarchical_turns = assemble_hierarchical_turns_with_interjections(
            primary_segments, verified_interjections, self.config
        )
        log_progress(f"Assembled {len(hierarchical_turns)} hierarchical turns")
        
        # Step 6: Build TranscriptFlow with metrics
        log_debug("Step 6: Building TranscriptFlow")
        transcript_flow = build_transcript_flow(
            hierarchical_turns,
            self.config,
            metadata={
                "builder": self.name,
                "timestamp": datetime.now().isoformat(),
                "total_words": len(words),
                "primary_segments": len(primary_segments),
                "verified_interjections": len(verified_interjections),
                "promoted_to_turns": len(promoted_to_turns),
                "llm_url": self.llm_url,
                "llm_stats": self.llm_stats.copy()
            }
        )
        
        # Save intermediate hierarchical output if directory provided
        if intermediate_dir:
            self._save_intermediate_output(transcript_flow, intermediate_dir)
        
        log_progress(f"Turn building complete: {transcript_flow.total_turns} turns, "
                    f"{transcript_flow.total_interjections} interjections")
        
        # Log summary
        self._log_summary(transcript_flow)
        
        return transcript_flow

    def _update_config_from_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Update configuration from kwargs."""
        if 'max_interjection_duration' in kwargs:
            self.config.max_interjection_duration = kwargs['max_interjection_duration']
        if 'max_interjection_words' in kwargs:
            self.config.max_interjection_words = kwargs['max_interjection_words']
        if 'max_gap_to_merge_turns' in kwargs:
            self.config.max_gap_to_merge_turns = kwargs['max_gap_to_merge_turns']
        if 'high_confidence_threshold' in kwargs:
            self.config.high_confidence_threshold = kwargs['high_confidence_threshold']
        if 'low_confidence_threshold' in kwargs:
            self.config.low_confidence_threshold = kwargs['low_confidence_threshold']
        if 'llm_timeout' in kwargs:
            self.config.llm_timeout = kwargs['llm_timeout']
        if 'llm_context_turns' in kwargs:
            self.config.llm_context_turns = kwargs['llm_context_turns']
        if 'timestamp_tolerance' in kwargs:
            self.config.timestamp_tolerance = kwargs['timestamp_tolerance']

    def _verify_interjections_with_llm(
        self,
        pending_interjections: List[PendingInterjection],
        primary_segments: List[RawSegment]
    ) -> Tuple[List[InterjectionSegment], List[RawSegment]]:
        """
        Use LLM to semantically verify pending interjections.
        
        The smart grouping phase identified these as *potential* interjections
        based on structural heuristics. Now we use the LLM to semantically
        verify whether they are truly interjections or should be promoted
        to primary turns.
        
        Args:
            pending_interjections: Interjection candidates from smart grouping
            primary_segments: The primary turn segments for context
            
        Returns:
            Tuple of (verified_interjections, promoted_to_turns)
        """
        verified_interjections: List[InterjectionSegment] = []
        promoted_to_turns: List[RawSegment] = []
        
        for pending in pending_interjections:
            # Find surrounding context from primary segments
            context_before, context_after = self._find_context_for_interjection(
                pending, primary_segments
            )
            
            # Determine if we need LLM verification
            needs_llm = self._needs_llm_verification(pending)
            
            if needs_llm:
                # Use LLM for semantic verification
                llm_result = self._verify_with_llm(pending, context_before, context_after)
                
                if llm_result is not None:
                    if llm_result['is_interjection']:
                        self.llm_stats['verified_as_interjection'] += 1
                        interjection = self._create_interjection_segment(
                            pending, 
                            llm_result.get('interjection_type', 'unclear'),
                            llm_result.get('confidence', 0.8),
                            "llm_verified"
                        )
                        verified_interjections.append(interjection)
                    else:
                        self.llm_stats['verified_as_turn'] += 1
                        # Promote to primary turn
                        segment = self._create_raw_segment_from_pending(pending)
                        segment.classification_method = "llm_promoted_to_turn"
                        promoted_to_turns.append(segment)
                else:
                    # LLM failed - use rule-based fallback
                    interjection = self._create_interjection_segment(
                        pending,
                        detect_interjection_type(pending.text, self.config) or "unclear",
                        0.6,
                        "rule_fallback"
                    )
                    verified_interjections.append(interjection)
            else:
                # High confidence from rules - no LLM needed
                interjection = self._create_interjection_segment(
                    pending,
                    detect_interjection_type(pending.text, self.config) or "unclear",
                    0.9,
                    "rule_high_confidence"
                )
                verified_interjections.append(interjection)
        
        return verified_interjections, promoted_to_turns
    
    def _needs_llm_verification(self, pending: PendingInterjection) -> bool:
        """
        Determine if a pending interjection needs LLM verification.
        
        Very short, pattern-matching utterances don't need LLM verification.
        Longer or ambiguous ones do.
        """
        # Very short (1-2 words) with pattern match - high confidence
        if pending.word_count <= 2:
            if detect_interjection_type(pending.text, self.config):
                return False
        
        # 3-5 words or no pattern match - needs verification
        if pending.word_count >= 3:
            return True
        
        # Short but no pattern match - verify
        if not detect_interjection_type(pending.text, self.config):
            return True
        
        return False
    
    def _find_context_for_interjection(
        self,
        pending: PendingInterjection,
        primary_segments: List[RawSegment]
    ) -> Tuple[Optional[RawSegment], Optional[RawSegment]]:
        """Find the primary segments before and after this interjection."""
        context_before = None
        context_after = None
        
        for segment in primary_segments:
            if segment.end <= pending.start:
                context_before = segment
            elif segment.start >= pending.end and context_after is None:
                context_after = segment
                break
        
        return context_before, context_after
    
    def _create_interjection_segment(
        self,
        pending: PendingInterjection,
        interjection_type: str,
        confidence: float,
        classification_method: str
    ) -> InterjectionSegment:
        """Create an InterjectionSegment from a PendingInterjection."""
        return InterjectionSegment(
            speaker=pending.speaker,
            start=pending.start,
            end=pending.end,
            text=pending.text,
            words=pending.words,
            confidence=confidence,
            interjection_type=interjection_type,
            interrupt_level="low",  # Will be recalculated during assembly
            classification_method=classification_method
        )
    
    def _create_raw_segment_from_pending(self, pending: PendingInterjection) -> RawSegment:
        """Convert a PendingInterjection to a RawSegment (for promotion to turn)."""
        return RawSegment(
            speaker=pending.speaker,
            start=pending.start,
            end=pending.end,
            text=pending.text,
            words=pending.words,
            is_interjection=False,
            interjection_confidence=0.0
        )
    
    def _merge_promoted_turns(
        self,
        primary_segments: List[RawSegment],
        promoted_turns: List[RawSegment]
    ) -> List[RawSegment]:
        """Merge promoted turns back into primary segments, sorted by time."""
        all_segments = primary_segments + promoted_turns
        all_segments.sort(key=lambda s: s.start)
        return all_segments

    def _verify_with_llm(
        self,
        pending: PendingInterjection,
        context_before: Optional[RawSegment],
        context_after: Optional[RawSegment]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to semantically verify if this is truly an interjection.
        
        Args:
            pending: The pending interjection to verify
            context_before: Previous primary segment for context
            context_after: Next primary segment for context
            
        Returns:
            Dict with 'is_interjection', 'confidence', 'interjection_type', 'reasoning'
            or None if LLM call failed
        """
        self.llm_stats["calls_made"] += 1
        
        # Build the prompt
        prompt = self._build_verification_prompt(pending, context_before, context_after)
        
        try:
            start_time = time.time()
            
            payload = {
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.llm_url}/chat/completions",
                json=payload,
                timeout=self.config.llm_timeout
            )
            response.raise_for_status()
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.llm_stats["total_time_ms"] += elapsed_ms
            
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"]
            
            # Parse the response (handle Harmony format)
            parsed = self._parse_llm_response(raw_response)
            
            if parsed:
                self.llm_stats["calls_succeeded"] += 1
                log_debug(f"LLM verified '{pending.text[:30]}...' as "
                         f"{'interjection' if parsed['is_interjection'] else 'turn'} "
                         f"(confidence: {parsed.get('confidence', 'N/A')})")
                return parsed
            else:
                self.llm_stats["calls_failed"] += 1
                log_debug(f"Failed to parse LLM response for '{pending.text[:30]}...'")
                return None
                
        except requests.RequestException as e:
            self.llm_stats["calls_failed"] += 1
            log_debug(f"LLM request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            self.llm_stats["calls_failed"] += 1
            log_debug(f"LLM response parsing error: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for semantic verification."""
        return """You are an expert at analyzing interview conversations.

Your task is to determine whether an utterance is:
1. An INTERJECTION - a brief acknowledgment, reaction, or backchannel that does NOT claim the conversational floor
2. A TURN - a substantive contribution that claims speaking rights and advances the conversation

INTERJECTIONS include:
- Acknowledgments: "yeah", "uh-huh", "mm-hmm", "right", "okay"
- Brief reactions: "really?", "wow", "oh", "interesting"  
- Backchannels that show listening without claiming the floor

TURNS include:
- Starting a new topic or thought
- Answering a question substantively
- Asking a real question that expects an answer
- Making a statement that advances the conversation
- Taking over the conversational floor

KEY CONTEXT: This is from an interview where one person (usually the Participant) often speaks at length while the other (Interviewer) provides brief acknowledgments. If the utterance appears during the other speaker's extended turn, it's more likely an interjection.

Respond with ONLY valid JSON (no markdown, no explanation):
{"classification": "interjection" or "turn", "confidence": 0.0-1.0, "type": "acknowledgment"/"question"/"reaction"/"unclear" or null, "reasoning": "brief explanation"}"""

    def _build_verification_prompt(
        self,
        pending: PendingInterjection,
        context_before: Optional[RawSegment],
        context_after: Optional[RawSegment]
    ) -> str:
        """Build the user prompt for semantic verification."""
        lines = [
            "Analyze this utterance from an interview conversation:",
            "",
            "CONTEXT:"
        ]
        
        # Previous context
        if context_before:
            prev_text = context_before.text[:300] + "..." if len(context_before.text) > 300 else context_before.text
            lines.append(f"  Before: [{context_before.speaker}] \"{prev_text}\"")
            lines.append(f"          ({context_before.duration:.1f}s, {context_before.word_count} words)")
        else:
            lines.append("  Before: [start of conversation]")
        
        lines.append("")
        
        # Target utterance
        lines.append(f"TARGET UTTERANCE:")
        lines.append(f"  [{pending.speaker}] \"{pending.text}\"")
        lines.append(f"  ({pending.duration:.1f}s, {pending.word_count} words)")
        lines.append(f"  Detected during {pending.detected_during_turn_of}'s speaking turn")
        
        lines.append("")
        
        # Next context
        if context_after:
            next_text = context_after.text[:300] + "..." if len(context_after.text) > 300 else context_after.text
            lines.append(f"  After: [{context_after.speaker}] \"{next_text}\"")
            lines.append(f"         ({context_after.duration:.1f}s, {context_after.word_count} words)")
        else:
            lines.append("  After: [end of conversation]")
        
        lines.append("")
        lines.append("Is the TARGET UTTERANCE an interjection or a substantive turn?")
        
        return "\n".join(lines)

    def _parse_llm_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response, handling Harmony format if present.
        
        Returns parsed dict or None if parsing fails.
        """
        # First, try to extract from Harmony format
        content = self._parse_harmony_response(raw_response)
        
        # Try to parse as JSON
        try:
            # Find JSON in the response (in case there's extra text)
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)
            
            # Validate and normalize the response
            classification = data.get('classification', '').lower()
            
            if classification in ('interjection', 'turn'):
                return {
                    'is_interjection': classification == 'interjection',
                    'confidence': float(data.get('confidence', 0.5)),
                    'interjection_type': data.get('type'),
                    'reasoning': data.get('reasoning', '')
                }
            else:
                log_debug(f"Invalid classification value: {classification}")
                return None
                
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log_debug(f"Failed to parse LLM response as JSON: {e}")
            log_debug(f"Raw response: {content[:200]}")
            
            # Try simple text parsing as fallback
            content_lower = content.lower()
            if 'interjection' in content_lower and 'turn' not in content_lower[:50]:
                return {'is_interjection': True, 'confidence': 0.5, 'interjection_type': 'unclear', 'reasoning': 'parsed from text'}
            elif 'turn' in content_lower and 'interjection' not in content_lower[:50]:
                return {'is_interjection': False, 'confidence': 0.5, 'interjection_type': None, 'reasoning': 'parsed from text'}
            
            return None

    def _parse_harmony_response(self, raw_response: str) -> str:
        """
        Parse Harmony format response to extract final content.
        
        Harmony format uses tokens like <|channel|>final<|message|>content<|end|>
        """
        # Pattern to extract channel and content
        pattern = r'<\|channel\|>(\w+)<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)'
        
        matches = re.findall(pattern, raw_response, re.DOTALL)
        
        # Look for 'final' channel content
        for channel, content in matches:
            if channel == 'final':
                return content.strip()
        
        # If no Harmony format detected, check if there are any harmony tokens
        if '<|' not in raw_response:
            return raw_response.strip()
        
        # Try a simpler extraction - get content after last <|message|>
        last_message = raw_response.split('<|message|>')
        if len(last_message) > 1:
            content = last_message[-1]
            # Remove trailing tokens
            content = re.sub(r'<\|[^|]+\|>.*$', '', content, flags=re.DOTALL)
            return content.strip()
        
        return raw_response.strip()

    def _save_intermediate_output(self, transcript_flow: TranscriptFlow, intermediate_dir: Path) -> None:
        """Save intermediate hierarchical output for debugging/analysis."""
        try:
            if isinstance(intermediate_dir, str):
                intermediate_dir = Path(intermediate_dir)
            
            turns_dir = intermediate_dir / "turns"
            turns_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full TranscriptFlow
            flow_file = turns_dir / "transcript_flow_llm.json"
            with open(flow_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_flow.to_dict(), f, indent=2, ensure_ascii=False)
            log_intermediate_save(str(flow_file), "TranscriptFlow (LLM) saved to")
            
            # Save conversation metrics separately
            metrics_file = turns_dir / "conversation_metrics_llm.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_flow.conversation_metrics, f, indent=2, ensure_ascii=False)
            log_intermediate_save(str(metrics_file), "Conversation metrics (LLM) saved to")
            
            # Save LLM stats
            llm_stats_file = turns_dir / "llm_classification_stats.json"
            with open(llm_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_stats, f, indent=2)
            log_intermediate_save(str(llm_stats_file), "LLM stats saved to")
            
        except Exception as e:
            log_debug(f"Failed to save intermediate output: {e}")

    def _log_summary(self, transcript_flow: TranscriptFlow) -> None:
        """Log a summary of the turn building results."""
        metrics = transcript_flow.conversation_metrics
        
        log_progress(f"Turn building summary (LLM-enhanced):")
        log_progress(f"  Total turns: {metrics.get('total_turns', 0)}")
        log_progress(f"  Total interjections: {metrics.get('total_interjections', 0)}")
        log_progress(f"  Average flow continuity: {metrics.get('average_flow_continuity', 1.0):.2f}")
        log_progress(f"  LLM calls: {self.llm_stats['calls_made']} "
                    f"(success: {self.llm_stats['calls_succeeded']}, "
                    f"failed: {self.llm_stats['calls_failed']})")
        log_progress(f"  LLM results: {self.llm_stats['verified_as_interjection']} interjections, "
                    f"{self.llm_stats['verified_as_turn']} promoted to turns")
        
        if self.llm_stats['calls_succeeded'] > 0:
            avg_time = self.llm_stats['total_time_ms'] / self.llm_stats['calls_succeeded']
            log_progress(f"  Average LLM response time: {avg_time:.0f}ms")


def register_turn_builder_plugins():
    """Register LLM turn builder plugin."""
    registry.register_turn_builder_provider(SplitAudioLLMTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()
