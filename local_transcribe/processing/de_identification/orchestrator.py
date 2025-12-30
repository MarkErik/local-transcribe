#!/usr/bin/env python3
"""
De-identification orchestrator for automated two-pass processing.

This module provides the main entry point for de-identification, automatically
running both first and second passes when appropriate.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.lib.program_logger import log_progress, log_status, log_debug, get_output_context

from .core import (
    DeIdentificationConfig,
    DeIdentificationResult,
    WordReplacement,
    DiscoveredName,
    DEFAULT_CONFIG,
)
from .llm_client import LLMDeIdentifierClient
from .first_pass import de_identify_first_pass, de_identify_text_first_pass
from .second_pass import de_identify_second_pass, build_global_name_list
from .audit import AuditLogger
from .debug import DebugFileWriter


class DeIdentificationOrchestrator:
    """
    Orchestrates the complete de-identification workflow.
    
    Automatically runs both first and second passes, handling:
    - Single speaker transcripts
    - Multi-speaker transcripts (with cross-speaker name sharing)
    - Plain text transcripts
    - Debug output and audit logging
    """
    
    def __init__(
        self,
        llm_url: str = "http://0.0.0.0:8080",
        intermediate_dir: Optional[Path] = None,
        config: Optional[DeIdentificationConfig] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            llm_url: URL of the LLM server
            intermediate_dir: Directory for saving debug/audit files
            config: De-identification configuration
        """
        self.config = config or DeIdentificationConfig(llm_url=llm_url)
        self.config.llm_url = llm_url
        self.intermediate_dir = Path(intermediate_dir) if intermediate_dir else None
        
        # Initialize LLM client (will auto-detect Harmony format)
        self.llm_client = LLMDeIdentifierClient(llm_url, self.config)
        
        # Initialize debug writer if DEBUG logging enabled
        self.debug_writer: Optional[DebugFileWriter] = None
        if self.intermediate_dir:
            debug_enabled = get_output_context().should_log("DEBUG")
            if debug_enabled:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                debug_dir = self.intermediate_dir / "de_identification" / "debug" / timestamp
                self.debug_writer = DebugFileWriter(debug_dir)
                log_debug(f"Debug mode enabled - saving to {debug_dir}")
        
        # Initialize audit logger
        self.audit_logger: Optional[AuditLogger] = None
        if self.intermediate_dir:
            self.audit_logger = AuditLogger(self.intermediate_dir / "de_identification")
    
    def de_identify(
        self,
        segments: List[WordSegment],
        speaker_name: Optional[str] = None,
    ) -> DeIdentificationResult:
        """
        De-identify word segments using two-pass processing.
        
        This is the main entry point for single-speaker de-identification.
        Both passes are run automatically.
        
        Args:
            segments: List of WordSegment objects
            speaker_name: Optional speaker name for logging/audit
            
        Returns:
            DeIdentificationResult with all modifications and metadata
        """
        if not segments:
            return DeIdentificationResult(segments=segments)
        
        log_status(f"Starting de-identification for {len(segments)} segments")
        
        # === First Pass ===
        first_result = de_identify_first_pass(
            segments,
            self.llm_client,
            self.config,
            self.intermediate_dir,
            speaker_name,
            self.debug_writer
        )
        
        # === Second Pass ===
        # Build name list from first pass
        if first_result.discovered_names:
            log_progress(
                f"Discovered {len(first_result.discovered_names)} unique names, "
                "running second pass"
            )
            
            # Convert to DiscoveredName format
            global_names = [
                DiscoveredName(name=name, source_speaker=speaker_name, occurrences=1)
                for name in first_result.discovered_names
            ]
            
            second_result = de_identify_second_pass(
                first_result.segments,
                global_names,
                self.llm_client,
                self.config,
                self.intermediate_dir,
                speaker_name,
                first_result.replacements,
                self.debug_writer
            )
            
            final_segments = second_result.segments
            second_replacements = second_result.additional_replacements
        else:
            log_progress("No names discovered in first pass, skipping second pass")
            final_segments = first_result.segments
            second_replacements = []
        
        # Create combined audit log
        if self.audit_logger:
            all_words = [seg.text for seg in segments]
            self.audit_logger.create_combined_audit_log(
                first_pass_replacements=first_result.replacements,
                second_pass_replacements=second_replacements,
                speaker_name=speaker_name,
                total_words=len(segments),
                all_words=all_words
            )
        
        log_status(
            f"De-identification complete: "
            f"{len(first_result.replacements)} first-pass + "
            f"{len(second_replacements)} second-pass replacements"
        )
        
        return DeIdentificationResult(
            segments=final_segments,
            first_pass_replacements=first_result.replacements,
            second_pass_replacements=second_replacements,
            discovered_names=first_result.discovered_names
        )
    
    def de_identify_multi_speaker(
        self,
        speaker_segments: Dict[str, List[WordSegment]],
    ) -> Dict[str, DeIdentificationResult]:
        """
        De-identify transcripts for multiple speakers with cross-speaker name sharing.
        
        This method:
        1. Runs first pass for all speakers
        2. Collects all discovered names
        3. Runs second pass for all speakers using the combined name list
        
        Args:
            speaker_segments: Dict mapping speaker_name -> list of segments
            
        Returns:
            Dict mapping speaker_name -> DeIdentificationResult
        """
        if not speaker_segments:
            return {}
        
        log_status(f"Starting multi-speaker de-identification for {len(speaker_segments)} speakers")
        
        # === First Pass for All Speakers ===
        first_pass_results: Dict[str, Any] = {}
        all_replacements: Dict[str, List[WordReplacement]] = {}
        
        for speaker_name, segments in speaker_segments.items():
            log_progress(f"First pass for {speaker_name}")
            
            result = de_identify_first_pass(
                segments,
                self.llm_client,
                self.config,
                self.intermediate_dir,
                speaker_name,
                self.debug_writer
            )
            
            first_pass_results[speaker_name] = result
            all_replacements[speaker_name] = result.replacements
            
            log_progress(
                f"First pass for {speaker_name}: "
                f"{len(result.discovered_names)} unique names found"
            )
        
        # === Build Global Name List ===
        global_names = build_global_name_list(all_replacements)
        
        if global_names:
            log_progress(f"Global name list: {', '.join(n.name for n in global_names)}")
        else:
            log_progress("No names discovered across all speakers")
        
        # === Second Pass for All Speakers ===
        final_results: Dict[str, DeIdentificationResult] = {}
        
        for speaker_name, segments in speaker_segments.items():
            first_result = first_pass_results[speaker_name]
            
            if global_names:
                log_progress(f"Second pass for {speaker_name}")
                
                second_result = de_identify_second_pass(
                    first_result.segments,
                    global_names,
                    self.llm_client,
                    self.config,
                    self.intermediate_dir,
                    speaker_name,
                    first_result.replacements,
                    self.debug_writer
                )
                
                final_segments = second_result.segments
                second_replacements = second_result.additional_replacements
                
                if second_replacements:
                    log_progress(
                        f"Second pass for {speaker_name}: "
                        f"{len(second_replacements)} additional names found"
                    )
            else:
                final_segments = first_result.segments
                second_replacements = []
            
            # Create audit log for this speaker
            if self.audit_logger:
                all_words = [seg.text for seg in segments]
                self.audit_logger.create_combined_audit_log(
                    first_pass_replacements=first_result.replacements,
                    second_pass_replacements=second_replacements,
                    speaker_name=speaker_name,
                    total_words=len(segments),
                    global_names=global_names if global_names else None,
                    all_words=all_words
                )
            
            final_results[speaker_name] = DeIdentificationResult(
                segments=final_segments,
                first_pass_replacements=first_result.replacements,
                second_pass_replacements=second_replacements,
                discovered_names=first_result.discovered_names
            )
        
        log_status("Multi-speaker de-identification complete")
        
        return final_results
    
    def de_identify_text(
        self,
        text: str,
    ) -> str:
        """
        De-identify plain text using two-pass processing.
        
        Args:
            text: Plain text to de-identify
            
        Returns:
            De-identified text string
        """
        if not text or not text.strip():
            return text
        
        log_status(f"Starting text de-identification ({len(text.split())} words)")
        
        # First pass
        processed_text, first_replacements, discovered_names = de_identify_text_first_pass(
            text,
            self.llm_client,
            self.config,
            self.debug_writer
        )
        
        # For text mode, we don't run second pass (no multi-speaker context)
        # But we could add it later if needed
        
        log_status(f"Text de-identification complete: {len(first_replacements)} replacements")
        
        return processed_text


# Convenience function for simple use cases
def de_identify(
    segments: List[WordSegment],
    llm_url: str = "http://0.0.0.0:8080",
    intermediate_dir: Optional[Path] = None,
    speaker_name: Optional[str] = None,
) -> DeIdentificationResult:
    """
    Convenience function for de-identifying word segments.
    
    Creates an orchestrator and runs both passes automatically.
    
    Args:
        segments: List of WordSegment objects
        llm_url: URL of the LLM server
        intermediate_dir: Directory for saving files
        speaker_name: Optional speaker name
        
    Returns:
        DeIdentificationResult
    """
    orchestrator = DeIdentificationOrchestrator(
        llm_url=llm_url,
        intermediate_dir=intermediate_dir
    )
    return orchestrator.de_identify(segments, speaker_name)
