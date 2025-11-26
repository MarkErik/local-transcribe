#!/usr/bin/env python3
"""
Split audio turn builder provider using rule-based interjection detection.

This module implements a turn builder for split-audio mode that:
1. Merges word streams from multiple speakers into a unified timeline
2. Groups consecutive words by speaker into segments
3. Classifies segments as primary turns or interjections using rules
4. Assembles hierarchical turns with embedded interjections
5. Returns flat turns for output writers

This is the local (no LLM) version that uses heuristics and pattern matching.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from local_transcribe.framework.plugin_interfaces import TurnBuilderProvider, WordSegment, Turn, registry
from local_transcribe.lib.program_logger import log_progress, log_debug, log_intermediate_save, get_output_context

from local_transcribe.providers.turn_builders.split_audio_data_structures import (
    TurnBuilderConfig,
    EnrichedTranscript,
    HierarchicalTurn
)
from local_transcribe.providers.turn_builders.split_audio_base import (
    merge_word_streams,
    group_by_speaker,
    classify_segments_rule_based,
    assemble_hierarchical_turns,
    build_enriched_transcript
)


class SplitAudioTurnBuilderProvider(TurnBuilderProvider):
    """
    Rule-based split audio turn builder.
    
    This turn builder takes word segments from multiple speakers (each with timestamps
    and speaker attribution), merges them into a unified timeline, and builds
    hierarchical turns with interjection detection.
    
    Interjections (brief acknowledgments like "yeah", "uh-huh", reactions like "wow")
    are identified using rules based on:
    - Duration (< 2 seconds)
    - Word count (≤ 5 words)
    - Pattern matching (known interjection phrases)
    - Context (sandwiched between same-speaker segments)
    
    The output is a flat list of Turn objects for compatibility with existing
    output writers, but intermediate hierarchical data can be saved for analysis.
    """

    def __init__(self):
        """Initialize with default configuration."""
        self.config = TurnBuilderConfig()

    @property
    def name(self) -> str:
        return "split_audio_turn_builder"

    @property
    def short_name(self) -> str:
        return "Split Audio (Local)"

    @property
    def description(self) -> str:
        return "Rule-based turn builder for split audio mode with interjection detection"

    def build_turns(
        self,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Build turns from word segments with speaker assignments.
        
        Args:
            words: Word segments with speaker assignments (from all speakers)
            **kwargs: Configuration options including:
                - intermediate_dir: Path to save intermediate files
                - max_interjection_duration: Override default (2.0s)
                - max_interjection_words: Override default (5)
                - max_gap_to_merge_turns: Override default (3.0s)
                - include_interjections_in_output: If True, interleave interjections (default: False)
            
        Returns:
            List of Turn objects ready for output
        """
        if not words:
            log_progress("No words provided to turn builder")
            return []
        
        # Update config from kwargs
        self._update_config_from_kwargs(kwargs)
        
        # Get intermediate directory for saving debug output
        intermediate_dir = kwargs.get('intermediate_dir')
        
        log_progress(f"Building turns from {len(words)} word segments")
        
        # Step 1: Merge word streams into unified timeline
        log_debug("Step 1: Merging word streams")
        merged_words = merge_word_streams(words)
        log_progress(f"Merged into {len(merged_words)} words in timeline")
        
        # Step 2: Group by speaker into raw segments
        log_debug("Step 2: Grouping by speaker")
        segments = group_by_speaker(merged_words)
        log_progress(f"Grouped into {len(segments)} raw segments")
        
        # Step 3: Classify segments as primary or interjection
        log_debug("Step 3: Classifying segments (rule-based)")
        primary_segments, interjection_segments = classify_segments_rule_based(
            segments, self.config
        )
        log_progress(f"Classification: {len(primary_segments)} primary, {len(interjection_segments)} interjections")
        
        # Mark segments with classification results
        for seg in primary_segments:
            seg.is_interjection = False
        for seg in interjection_segments:
            seg.is_interjection = True
        
        # Combine back for assembly (order matters)
        all_segments_classified = sorted(
            primary_segments + interjection_segments,
            key=lambda s: s.start
        )
        
        # Step 4: Assemble hierarchical turns
        log_debug("Step 4: Assembling hierarchical turns")
        hierarchical_turns = assemble_hierarchical_turns(all_segments_classified, self.config)
        log_progress(f"Assembled {len(hierarchical_turns)} hierarchical turns")
        
        # Step 5: Build enriched transcript with metrics
        log_debug("Step 5: Building enriched transcript")
        enriched = build_enriched_transcript(
            hierarchical_turns,
            self.config,
            metadata={
                "builder": self.name,
                "timestamp": datetime.now().isoformat(),
                "total_words": len(words),
                "total_segments": len(segments)
            }
        )
        
        # Save intermediate hierarchical output if directory provided
        if intermediate_dir:
            self._save_intermediate_output(enriched, intermediate_dir)
        
        # Step 6: Convert to flat turns for output
        include_interjections = kwargs.get('include_interjections_in_output', False)
        if include_interjections:
            flat_turns = enriched.to_flat_turns_with_interjections()
        else:
            flat_turns = enriched.to_flat_turns()
        
        log_progress(f"Turn building complete: {len(flat_turns)} turns")
        
        # Log summary
        self._log_summary(enriched)
        
        return flat_turns

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

    def _save_intermediate_output(self, enriched: EnrichedTranscript, intermediate_dir: Path) -> None:
        """Save intermediate hierarchical output for debugging/analysis."""
        try:
            if isinstance(intermediate_dir, str):
                intermediate_dir = Path(intermediate_dir)
            
            turns_dir = intermediate_dir / "turns"
            turns_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full enriched transcript
            enriched_file = turns_dir / "hierarchical_turns.json"
            with open(enriched_file, 'w', encoding='utf-8') as f:
                json.dump(enriched.to_dict(), f, indent=2, ensure_ascii=False)
            log_intermediate_save(str(enriched_file), "Hierarchical turns saved to")
            
            # Save conversation metrics separately for easy access
            metrics_file = turns_dir / "conversation_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(enriched.conversation_metrics, f, indent=2, ensure_ascii=False)
            log_intermediate_save(str(metrics_file), "Conversation metrics saved to")
            
        except Exception as e:
            log_debug(f"Failed to save intermediate output: {e}")

    def _log_summary(self, enriched: EnrichedTranscript) -> None:
        """Log a summary of the turn building results."""
        metrics = enriched.conversation_metrics
        
        log_progress(f"Turn building summary:")
        log_progress(f"  Total turns: {metrics.get('total_turns', 0)}")
        log_progress(f"  Total interjections: {metrics.get('total_interjections', 0)}")
        log_progress(f"  Average flow continuity: {metrics.get('average_flow_continuity', 1.0):.2f}")
        
        # Log speaker stats
        speaker_stats = metrics.get('speaker_statistics', {})
        for speaker, stats in speaker_stats.items():
            log_debug(f"  {speaker}: {stats.get('turn_count', 0)} turns, "
                     f"{stats.get('interjection_count', 0)} interjections, "
                     f"{stats.get('total_words', 0)} words")


def register_turn_builder_plugins():
    """Register split audio turn builder plugin."""
    registry.register_turn_builder_provider(SplitAudioTurnBuilderProvider())


# Auto-register on import
register_turn_builder_plugins()
register_turn_builder_plugins()