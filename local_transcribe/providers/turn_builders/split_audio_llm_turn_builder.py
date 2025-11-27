#!/usr/bin/env python3
"""
Split audio turn builder provider using LLM-enhanced interjection detection.

This module implements a turn builder for split-audio mode that:
1. Merges word streams from multiple speakers into a unified timeline
2. Groups consecutive words by speaker into segments
3. Classifies clear cases using rules, ambiguous cases using LLM
4. Assembles hierarchical turns with embedded interjections
5. Returns TranscriptFlow with full hierarchical structure

This version uses LLM calls to classify ambiguous segments where
rule-based confidence is not high enough.
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
    HierarchicalTurn
)
from local_transcribe.providers.turn_builders.split_audio_base import (
    merge_word_streams,
    group_by_speaker,
    calculate_interjection_confidence,
    detect_interjection_type,
    assemble_hierarchical_turns,
    build_transcript_flow
)


class SplitAudioLLMTurnBuilderProvider(TurnBuilderProvider):
    """
    LLM-enhanced split audio turn builder.
    
    This turn builder extends the rule-based version by using an LLM to
    classify ambiguous segments - those where rule-based confidence is
    between the low and high thresholds.
    
    Clear cases (very short acknowledgments, or long substantive turns)
    are still classified using rules for efficiency. Only ambiguous cases
    (e.g., 2-3 word utterances that could be either) are sent to the LLM.
    
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
            "total_time_ms": 0
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
        
        Args:
            words: Word segments with speaker assignments (from all speakers)
            **kwargs: Configuration options including:
                - llm_url: URL of the LLM server (default: http://100.84.208.72:8080)
                - intermediate_dir: Path to save intermediate files
                - max_interjection_duration: Override default (2.0s)
                - max_interjection_words: Override default (5)
                - max_gap_to_merge_turns: Override default (3.0s)
                - llm_timeout: Timeout for LLM requests in seconds
            
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
        self.llm_stats = {"calls_made": 0, "calls_succeeded": 0, "calls_failed": 0, "total_time_ms": 0}
        
        # Get intermediate directory
        intermediate_dir = kwargs.get('intermediate_dir')
        
        log_progress(f"Building turns from {len(words)} word segments (LLM-enhanced)")
        log_progress(f"LLM endpoint: {self.llm_url}")
        
        # Step 1: Merge word streams into unified timeline
        log_debug("Step 1: Merging word streams")
        merged_words = merge_word_streams(words)
        log_progress(f"Merged into {len(merged_words)} words in timeline")
        
        # Step 2: Group by speaker into raw segments
        log_debug("Step 2: Grouping by speaker")
        segments = group_by_speaker(merged_words)
        log_progress(f"Grouped into {len(segments)} raw segments")
        
        # Step 3: Classify segments using hybrid approach
        log_debug("Step 3: Classifying segments (LLM-enhanced)")
        self._classify_segments_with_llm(segments)
        
        primary_count = sum(1 for s in segments if not s.is_interjection)
        interjection_count = sum(1 for s in segments if s.is_interjection)
        log_progress(f"Classification: {primary_count} primary, {interjection_count} interjections")
        log_progress(f"LLM stats: {self.llm_stats['calls_made']} calls, "
                    f"{self.llm_stats['calls_succeeded']} succeeded, "
                    f"{self.llm_stats['calls_failed']} failed")
        
        # Step 4: Assemble hierarchical turns
        log_debug("Step 4: Assembling hierarchical turns")
        hierarchical_turns = assemble_hierarchical_turns(segments, self.config)
        log_progress(f"Assembled {len(hierarchical_turns)} hierarchical turns")
        
        # Step 5: Build TranscriptFlow with metrics
        log_debug("Step 5: Building TranscriptFlow")
        transcript_flow = build_transcript_flow(
            hierarchical_turns,
            self.config,
            metadata={
                "builder": self.name,
                "timestamp": datetime.now().isoformat(),
                "total_words": len(words),
                "total_segments": len(segments),
                "llm_url": self.llm_url,
                "llm_stats": self.llm_stats.copy()
            }
        )
        
        # Save intermediate hierarchical output if directory provided
        if intermediate_dir:
            self._save_intermediate_output(transcript_flow, intermediate_dir)
        
        log_progress(f"Turn building complete: {transcript_flow.total_turns} turns, {transcript_flow.total_interjections} interjections")
        
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

    def _classify_segments_with_llm(self, segments: List[RawSegment]) -> None:
        """
        Classify segments using hybrid rule + LLM approach.
        
        Clear cases are handled by rules, ambiguous cases by LLM.
        Modifies segments in place.
        """
        for i, segment in enumerate(segments):
            prev_seg = segments[i - 1] if i > 0 else None
            next_seg = segments[i + 1] if i < len(segments) - 1 else None
            
            # Calculate rule-based confidence
            confidence, interjection_type = calculate_interjection_confidence(
                segment, prev_seg, next_seg, self.config
            )
            segment.interjection_confidence = confidence
            
            # Hard rules: definitely not an interjection
            if segment.duration > self.config.max_interjection_duration:
                segment.is_interjection = False
                segment.classification_method = "rule_duration"
                continue
            
            if segment.word_count > self.config.max_interjection_words:
                segment.is_interjection = False
                segment.classification_method = "rule_word_count"
                continue
            
            # Clear high-confidence interjection
            if confidence >= self.config.high_confidence_threshold:
                segment.is_interjection = True
                segment.classification_method = "rule_high_confidence"
                continue
            
            # Clear low-confidence (not interjection)
            if confidence <= self.config.low_confidence_threshold:
                segment.is_interjection = False
                segment.classification_method = "rule_low_confidence"
                continue
            
            # Ambiguous case - use LLM
            log_debug(f"Ambiguous segment ({confidence:.2f}): '{segment.text[:50]}...' - using LLM")
            
            llm_result = self._classify_with_llm(segment, prev_seg, next_seg)
            
            if llm_result is not None:
                segment.is_interjection = llm_result['is_interjection']
                segment.interjection_confidence = llm_result.get('confidence', confidence)
                segment.classification_method = "llm"
            else:
                # LLM failed - fall back to rule-based default
                # For ambiguous cases, lean toward not-interjection (safer)
                segment.is_interjection = False
                segment.classification_method = "rule_fallback"

    def _classify_with_llm(
        self,
        segment: RawSegment,
        prev_segment: Optional[RawSegment],
        next_segment: Optional[RawSegment]
    ) -> Optional[Dict[str, Any]]:
        """
        Classify a segment using the LLM.
        
        Args:
            segment: The segment to classify
            prev_segment: Previous segment for context
            next_segment: Next segment for context
            
        Returns:
            Dict with 'is_interjection', 'confidence', 'interjection_type', 'reasoning'
            or None if LLM call failed
        """
        self.llm_stats["calls_made"] += 1
        
        # Build the prompt
        prompt = self._build_classification_prompt(segment, prev_segment, next_segment)
        
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
                log_debug(f"LLM classified '{segment.text[:30]}...' as "
                         f"{'interjection' if parsed['is_interjection'] else 'turn'} "
                         f"(confidence: {parsed.get('confidence', 'N/A')})")
                return parsed
            else:
                self.llm_stats["calls_failed"] += 1
                log_debug(f"Failed to parse LLM response for '{segment.text[:30]}...'")
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
        """Get the system prompt for classification."""
        return """You are an expert at analyzing interview conversations.
Your task is to classify whether a short utterance is:
1. An INTERJECTION - a brief acknowledgment, reaction, or backchannel that does NOT claim the conversational floor
2. A TURN - a substantive contribution that claims speaking rights

Examples of INTERJECTIONS:
- "yeah", "uh-huh", "mm-hmm" (acknowledgments)
- "really?", "what?" (brief questions)
- "wow", "oh", "interesting" (reactions)
- "right", "okay", "sure" (agreements)

Examples of TURNS (even if short):
- Starting a new topic or thought
- Answering a question substantively
- Asking a real question that expects an answer
- Making a statement that advances the conversation

Consider the context: if the utterance is sandwiched between the same speaker continuing their thought, it's likely an interjection.

Respond with ONLY valid JSON (no markdown, no explanation):
{"classification": "interjection" or "turn", "confidence": 0.0-1.0, "type": "acknowledgment"/"question"/"reaction"/"unclear" or null, "reasoning": "brief explanation"}"""

    def _build_classification_prompt(
        self,
        segment: RawSegment,
        prev_segment: Optional[RawSegment],
        next_segment: Optional[RawSegment]
    ) -> str:
        """Build the user prompt for classification."""
        lines = ["Analyze this utterance in an interview conversation:\n"]
        
        # Previous context
        if prev_segment:
            prev_text = prev_segment.text[:200] + "..." if len(prev_segment.text) > 200 else prev_segment.text
            lines.append(f"PREVIOUS: [{prev_segment.speaker}] \"{prev_text}\" ({prev_segment.duration:.1f}s, {prev_segment.word_count} words)")
        else:
            lines.append("PREVIOUS: [start of conversation]")
        
        lines.append("")
        
        # Target segment
        lines.append(f"TARGET: [{segment.speaker}] \"{segment.text}\" ({segment.duration:.1f}s, {segment.word_count} words)")
        
        lines.append("")
        
        # Next context
        if next_segment:
            next_text = next_segment.text[:200] + "..." if len(next_segment.text) > 200 else next_segment.text
            lines.append(f"NEXT: [{next_segment.speaker}] \"{next_text}\" ({next_segment.duration:.1f}s, {next_segment.word_count} words)")
        else:
            lines.append("NEXT: [end of conversation]")
        
        lines.append("")
        lines.append("Is the TARGET utterance an interjection or a substantive turn?")
        
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
        
        if self.llm_stats['calls_succeeded'] > 0:
            avg_time = self.llm_stats['total_time_ms'] / self.llm_stats['calls_succeeded']
            log_progress(f"  Average LLM response time: {avg_time:.0f}ms")


def register_turn_builder_plugins():
    """Register LLM turn builder plugin."""
    registry.register_turn_builder_provider(SplitAudioLLMTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()
