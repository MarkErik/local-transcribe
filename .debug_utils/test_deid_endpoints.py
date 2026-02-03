#!/usr/bin/env python3
"""
Test suite for de-identification endpoints using VAD-Split-Audio pipeline.

This test suite:
1. Transcribes P28 audio files using build_turns_vad_split_audio()
2. Extracts WordSegments from the TranscriptFlow
3. Tests all 5 LLM endpoints with first and second pass de-identification
4. Validates word count preservation and collects metrics

Uses Harmony and OpenAI-compatible endpoint formats from test_endpoint_multi.py.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

import requests

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.processing.turn_building.vad_turn_builder import build_turns_vad_split_audio
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow, HierarchicalTurn
from local_transcribe.processing.de_identification.core import DeIdentificationConfig, WordReplacement, ValidationResult, ChunkProcessingResult, DEFAULT_CONFIG
from local_transcribe.processing.de_identification.first_pass import de_identify_first_pass, FirstPassResult
from local_transcribe.processing.de_identification.second_pass import de_identify_second_pass, SecondPassResult, build_global_name_list
from local_transcribe.processing.de_identification.llm_client import LLMDeIdentifierClient


class EndpointType(Enum):
    """Endpoint format type."""
    HARMONY = "Harmony"
    OPENAI_COMPATIBLE = "OpenAI Compatible"


# =============================================================================
# Endpoint Configuration with Type Detection
# =============================================================================

# Map ports to endpoint types (from test_endpoint_multi.py)
ENDPOINT_PORT_MAP = {
    8080: EndpointType.HARMONY,
    8105: EndpointType.HARMONY,
    8107: EndpointType.OPENAI_COMPATIBLE,
#    8083: EndpointType.OPENAI_COMPATIBLE,
    8099: EndpointType.OPENAI_COMPATIBLE,
}

# P28 sample audio files
SPEAKER_AUDIO_FILES = {
    "Interviewer": "samples/audioMA-P28_cropped_30.0min.wav",
    "Participant": "samples/audioP28_cropped_30.0min.wav"
}


# =============================================================================
# Harmony/OpenAI-Compatible Client (from test_endpoint_multi.py patterns)
# =============================================================================

class MultiFormatLLMClient(LLMDeIdentifierClient):
    """
    LLM client that supports both Harmony and OpenAI-compatible formats.
    
    Extends LLMDeIdentifierClient to use the exact request/response formatting
    from test_endpoint_multi.py while maintaining compatibility with existing
    de-identification functions.
    """
    
    def __init__(
        self,
        llm_url: str,
        endpoint_type: EndpointType = EndpointType.OPENAI_COMPATIBLE,
        config: Optional[DeIdentificationConfig] = None
    ):
        """
        Initialize the multi-format LLM client.
        
        Args:
            llm_url: URL of the LLM server
            endpoint_type: Type of endpoint (Harmony or OpenAI-compatible)
            config: Configuration options
        """
        # Initialize parent class
        super().__init__(llm_url=llm_url, config=config)
        
        # Override format detection - use explicit endpoint type
        self._harmony_format_detected = (endpoint_type == EndpointType.HARMONY)
        self._harmony_detection_done = True
        self.endpoint_type = endpoint_type
    
    def _get_port(self) -> int:
        """Extract port from URL."""
        match = re.search(r':(\d+)', self.llm_url)
        return int(match.group(1)) if match else 8080
    
    # -------------------------------------------------------------------------
    # Prompt Building (from test_endpoint_multi.py)
    # -------------------------------------------------------------------------
    
    def build_harmony_system_message(self, reasoning_level: str = "high") -> str:
        """Build a system message in harmony format."""
        return f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {datetime.now().strftime('%Y-%m-%d')}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"""
    
    def build_harmony_developer_message(self, instructions: str = "You are a helpful assistant.") -> str:
        """Build a developer message in harmony format."""
        return f"""<|start|>developer<|message|># Instructions

{instructions}<|end|>"""
    
    def build_harmony_user_message(self, content: str) -> str:
        """Build a user message in harmony format."""
        return f"""<|start|>user<|message|>{content}<|end|>"""
    
    # -------------------------------------------------------------------------
    # Override process_chunk to use endpoint-specific formatting
    # -------------------------------------------------------------------------
    
    def process_chunk(
        self,
        text: str,
        system_prompt: str,
        validator,
        extra_validation_args: Optional[Dict[str, Any]] = None
    ) -> ChunkProcessingResult:
        """
        Process a text chunk with the LLM using endpoint-specific formatting.
        
        Overrides parent method to use Harmony or OpenAI-compatible format
        based on endpoint type.
        """
        extra_validation_args = extra_validation_args or {}
        
        all_attempts: List[Dict[str, Any]] = []
        total_response_time_ms = 0.0
        last_raw_response: Optional[str] = None
        last_validation: Optional[ValidationResult] = None
        
        max_retries = self.config.max_retries
        initial_temperature = self.config.temperature
        temperature_decay = self.config.temperature_decay
        
        for attempt in range(max_retries + 1):
            # Calculate temperature for this attempt
            if attempt == 0:
                current_temperature = initial_temperature
            else:
                current_temperature = max(0.0, initial_temperature - (attempt * temperature_decay))
            
            attempt_number = attempt + 1
            attempt_info: Dict[str, Any] = {
                'attempt': attempt_number,
                'temperature': current_temperature,
            }
            
            if attempt > 0:
                print(f"    Retry {attempt}/{max_retries} with temperature {current_temperature:.2f}")
            
            # Build request based on endpoint type
            if self.endpoint_type == EndpointType.HARMONY:
                # Harmony format: /v1/responses with input text
                url = f"{self.llm_url}/v1/responses"
                
                # Combine system and user messages for Harmony
                developer_msg = "You are a helpful assistant."
                input_text = f"{system_prompt}\n\n{developer_msg}\n\n{text}"
                
                body = {
                    "model": "model",
                    "input": input_text,
                    "max_output_tokens": 4096,
                    "temperature": current_temperature,
                }
            else:
                # OpenAI-compatible format: /v1/chat/completions
                url = f"{self.llm_url}/v1/chat/completions"
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
                
                body = {
                    "model": "model",
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": current_temperature,
                    "stream": False
                }
            
            try:
                start_time = time.time()
                response = requests.post(url, json=body, timeout=self.config.llm_timeout)
                response.raise_for_status()
                response_time_ms = (time.time() - start_time) * 1000
                total_response_time_ms += response_time_ms
                
                result = response.json()
                
                # Parse response based on endpoint type
                if self.endpoint_type == EndpointType.HARMONY:
                    raw_response = self._parse_harmony_raw_response(result)
                else:
                    raw_response = result["choices"][0]["message"]["content"]
                
                last_raw_response = raw_response if isinstance(raw_response, str) else str(raw_response)
                
                # Parse response (handles Harmony format if needed)
                processed_text = self.parse_response(last_raw_response)
                
                # Validate output
                validation = validator(text, processed_text, **extra_validation_args)
                
                attempt_info.update({
                    'validation_passed': validation.passed,
                    'validation_reason': validation.reason,
                    'response_time_ms': response_time_ms,
                    'raw_response': last_raw_response,
                    'processed_text': processed_text,
                    'validation_result': validation.to_dict(),
                })
                all_attempts.append(attempt_info)
                
                if validation.passed:
                    validation.details['total_attempts'] = attempt_number
                    validation.details['final_temperature'] = current_temperature
                    
                    return ChunkProcessingResult(
                        processed_text=processed_text,
                        response_time_ms=total_response_time_ms,
                        validation=validation,
                        raw_response=last_raw_response,
                        attempt_logs=all_attempts,
                    )
                else:
                    last_validation = validation
                    
            except Exception as e:
                print(f"    Request failed (attempt {attempt_number}): {e}")
                attempt_info['error'] = f'{e}'
                all_attempts.append(attempt_info)
                last_validation = ValidationResult(
                    passed=False,
                    reason=f'request failed: {e}',
                    details={}
                )
        
        # All retries exhausted
        final_validation = last_validation or ValidationResult(
            passed=False,
            reason='all attempts failed',
            details={}
        )
        final_validation.details['total_attempts'] = max_retries + 1
        final_validation.details['all_attempts_failed'] = True
        
        return ChunkProcessingResult(
            processed_text=text,  # Fall back to original
            response_time_ms=total_response_time_ms,
            validation=final_validation,
            raw_response=last_raw_response,
            attempt_logs=all_attempts,
        )
    
    def _parse_harmony_raw_response(self, response: dict) -> str:
        """Parse raw Harmony response to extract text content."""
        if not isinstance(response, dict):
            return str(response)
        
        if "output" in response:
            output = response["output"]
            if output and len(output) > 0:
                content_list = output[0].get("content", [])
                if content_list and len(content_list) > 0:
                    text = content_list[0].get("text", "")
                    return text
        return ""


# =============================================================================
# Endpoint Configuration
# =============================================================================

# Output directory for intermediate files and results
DEBUG_OUTPUT_DIR = Path("./debug_output")
RESULTS_FILE = DEBUG_OUTPUT_DIR / "deid_endpoint_test_results.json"

# 5 endpoints to test with type detection (from test_endpoint_multi.py)
ENDPOINTS = [
    {"name": "harmony_8080", "url": "http://100.84.208.72:8080", "port": 8080},
    {"name": "harmony_8105", "url": "http://100.84.208.72:8105", "port": 8105},
    {"name": "openai_compat_8107", "url": "http://100.84.208.72:8107", "port": 8107},
    #{"name": "openai_compat_8083", "url": "http://100.84.208.72:8083", "port": 8083},
    {"name": "openai_compat_8099", "url": "http://100.84.208.72:8099", "port": 8099},
]

# De-identification config
DEID_CONFIG = DeIdentificationConfig(
    chunk_size=400,
    overlap_size=60,
    min_final_chunk=200,
    llm_timeout=360,
    temperature=0.7,
    max_retries=3,
)


# =============================================================================
# Configuration
# =============================================================================

# P28 sample audio files
SPEAKER_AUDIO_FILES = {
    "Interviewer": "samples/audioMA-P28_cropped_30.0min.wav",
    "Participant": "samples/audioP28_cropped_30.0min.wav"
}

# Output directory for intermediate files and results
DEBUG_OUTPUT_DIR = Path("./debug_output")
RESULTS_FILE = DEBUG_OUTPUT_DIR / "deid_endpoint_test_results.json"

# 5 endpoints to test (from test_endpoint_multi.py)
ENDPOINTS = [
    {"name": "harmony_8080", "url": "http://100.84.208.72:8080"},
    {"name": "harmony_8105", "url": "http://100.84.208.72:8105"},
    {"name": "openai_compat_8107", "url": "http://100.84.208.72:8107"},
    #{"name": "openai_compat_8083", "url": "http://100.84.208.72:8083"},
    {"name": "openai_compat_8099", "url": "http://100.84.208.72:8099"},
]

# De-identification config
DEID_CONFIG = DeIdentificationConfig(
    chunk_size=400,
    overlap_size=60,
    min_final_chunk=200,
    llm_timeout=360,
    temperature=0.7,
    max_retries=3,
)


# =============================================================================
# Helper Functions
# =============================================================================

def extract_word_segments(transcript: TranscriptFlow) -> List[WordSegment]:
    """
    Extract all WordSegments from transcript turns and interjections.
    
    Args:
        transcript: TranscriptFlow from VAD-Split-Audio pipeline
        
    Returns:
        List of all WordSegment objects in the transcript
    """
    all_words: List[WordSegment] = []
    
    for turn in transcript.turns:
        # Add words from primary turn
        all_words.extend(turn.words)
        
        # Add words from interjections
        for ij in turn.interjections:
            all_words.extend(ij.words)
    
    return all_words


def create_llm_client(endpoint_config: Dict[str, str]) -> MultiFormatLLMClient:
    """
    Create an LLM client for a specific endpoint with correct format type.
    
    Args:
        endpoint_config: Endpoint configuration with name, url, and port
        
    Returns:
        Configured MultiFormatLLMClient using Harmony or OpenAI-compatible format
    """
    port = int(endpoint_config.get("port", 8080))
    endpoint_type = ENDPOINT_PORT_MAP.get(port, EndpointType.OPENAI_COMPATIBLE)
    
    return MultiFormatLLMClient(
        llm_url=endpoint_config["url"],
        endpoint_type=endpoint_type,
        config=DEID_CONFIG
    )


def run_first_pass(
    segments: List[WordSegment],
    llm_client: MultiFormatLLMClient,
    endpoint_name: str
) -> Dict[str, Any]:
    """
    Run first-pass de-identification.
    
    Args:
        segments: WordSegments to process
        llm_client: Configured LLM client
        endpoint_name: Name for logging
        
    Returns:
        Dictionary with timing, results, and metrics
    """
    start_time = time.time()
    
    try:
        result = de_identify_first_pass(
            segments=segments,
            llm_client=llm_client,
            config=DEID_CONFIG,
            intermediate_dir=DEBUG_OUTPUT_DIR / "first_pass",
            debug_writer=None,
        )
        
        elapsed = time.time() - start_time
        
        # Count redactions
        redaction_count = len(result.replacements)
        
        return {
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
            "word_count": len(segments),
            "redaction_count": redaction_count,
            "discovered_names": list(result.discovered_names),
            "replacements": [r.to_dict() for r in result.replacements],
            "session_data": result.session_data,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
            "word_count": len(segments),
        }


def run_second_pass(
    segments: List[WordSegment],
    first_pass_replacements: List[WordReplacement],
    llm_client: MultiFormatLLMClient,
    endpoint_name: str
) -> Dict[str, Any]:
    """
    Run second-pass de-identification using global name list.
    
    Args:
        segments: WordSegments (modified from first pass)
        first_pass_replacements: Replacements from first pass
        llm_client: Configured LLM client
        endpoint_name: Name for logging
        
    Returns:
        Dictionary with timing, results, and metrics
    """
    start_time = time.time()
    
    try:
        # Build global name list from first pass replacements
        all_speaker_replacements = {"global": first_pass_replacements}
        global_names = build_global_name_list(all_speaker_replacements)
        
        result = de_identify_second_pass(
            segments=segments,
            global_names=global_names,
            llm_client=llm_client,
            config=DEID_CONFIG,
            intermediate_dir=DEBUG_OUTPUT_DIR / "second_pass",
            debug_writer=None,
        )
        
        elapsed = time.time() - start_time
        
        # Count additional redactions
        additional_count = len(result.additional_replacements)
        
        return {
            "success": True,
            "elapsed_seconds": round(elapsed, 2),
            "word_count": len(segments),
            "additional_redactions": additional_count,
            "names_from_list_found": list(result.names_from_list_found),
            "additional_replacements": [r.to_dict() for r in result.additional_replacements],
            "session_data": result.session_data,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "elapsed_seconds": round(elapsed, 2),
            "error": str(e),
            "word_count": len(segments),
        }


def validate_word_count_preservation(
    original_segments: List[WordSegment],
    processed_segments: List[WordSegment]
) -> Dict[str, Any]:
    """
    Validate that word count is preserved after de-identification.
    
    Args:
        original_segments: Original WordSegments
        processed_segments: Processed WordSegments
        
    Returns:
        Validation result dictionary
    """
    original_count = len(original_segments)
    processed_count = len(processed_segments)
    
    # Reconstruct text to check for [REDACTED] tokens
    original_text = " ".join(w.text for w in original_segments)
    processed_text = " ".join(w.text for w in processed_segments)
    
    # Count redaction tokens
    original_redactions = original_text.count("[REDACTED]")
    processed_redactions = processed_text.count("[REDACTED]")
    
    return {
        "original_word_count": original_count,
        "processed_word_count": processed_count,
        "word_count_preserved": original_count == processed_count,
        "original_redaction_tokens": original_redactions,
        "processed_redaction_tokens": processed_redactions,
        "new_redaction_tokens": processed_redactions - original_redactions,
    }


# =============================================================================
# Main Test Runner
# =============================================================================

def run_deid_endpoint_tests():
    """
    Main test runner that:
    1. Transcribes P28 files using VAD-Split-Audio pipeline
    2. Extracts WordSegments from transcript
    3. Tests all endpoints with first and second pass de-identification
    4. Saves comprehensive results to JSON
    """
    print("=" * 80)
    print("DE-IDENTIFICATION ENDPOINT TEST SUITE")
    print("VAD-Split-Audio Pipeline Integration")
    print("=" * 80)
    
    # Ensure output directory exists
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {
        "test_run_timestamp": datetime.now().isoformat(),
        "audio_files": SPEAKER_AUDIO_FILES,
        "deid_config": DEID_CONFIG.to_dict(),
        "endpoints_tested": [],
        "transcription_info": None,
        "word_segment_count": 0,
        "summary": {},
    }
    
    # -------------------------------------------------------------------------
    # Phase 1: VAD + ASR Transcription
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Transcribing P28 audio files with VAD-Split-Audio pipeline...")
    print("-" * 60)
    
    from local_transcribe.providers.transcribers.remote_transcriber import RemoteTranscriberProvider
    
    # Configure remote transcriber to use remote server at 100.84.208.72:7070
    transcriber = RemoteTranscriberProvider()
    transcriber.configure(server_url="http://100.84.208.72:7070")
    
    transcription_start = time.time()
    
    try:
        transcript = build_turns_vad_split_audio(
            speaker_audio_files=SPEAKER_AUDIO_FILES,
            transcriber_provider=transcriber,
            intermediate_dir=DEBUG_OUTPUT_DIR / "vad_transcription",
        )
        
        transcription_elapsed = time.time() - transcription_start
        
        print(f"✓ Transcription complete in {transcription_elapsed:.2f}s")
        print(f"  - Total turns: {transcript.total_turns}")
        print(f"  - Total interjections: {transcript.total_interjections}")
        
        # Get speaker statistics
        for speaker, stats in transcript.speaker_statistics.items():
            print(f"  - {speaker}: {stats.get('word_count', 0)} words, {stats.get('turn_count', 0)} turns")
        
        results["transcription_info"] = {
            "success": True,
            "elapsed_seconds": round(transcription_elapsed, 2),
            "total_turns": transcript.total_turns,
            "total_interjections": transcript.total_interjections,
            "speaker_statistics": transcript.speaker_statistics,
        }
        
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        results["transcription_info"] = {
            "success": False,
            "error": str(e),
        }
        results["endpoints_tested"] = []
        _save_results(results)
        return results
    
    # -------------------------------------------------------------------------
    # Phase 2: Extract WordSegments
    # -------------------------------------------------------------------------
    print("\n[PHASE 2] Extracting WordSegments from transcript...")
    print("-" * 60)
    
    word_segments = extract_word_segments(transcript)
    print(f"✓ Extracted {len(word_segments)} WordSegments")
    
    results["word_segment_count"] = len(word_segments)
    
    # Show sample of extracted words
    if word_segments:
        sample_text = " ".join(w.text for w in word_segments[:50])
        print(f"  Sample (first 50 words): {sample_text}...")
    
    # -------------------------------------------------------------------------
    # Phase 3: Test Each Endpoint
    # -------------------------------------------------------------------------
    print("\n[PHASE 3] Testing de-identification on all endpoints...")
    print("-" * 60)
    
    endpoint_results = []
    
    for endpoint in ENDPOINTS:
        endpoint_name = endpoint["name"]
        endpoint_url = endpoint["url"]
        
        print(f"\n{'='*60}")
        print(f"Testing endpoint: {endpoint_name} ({endpoint_url})")
        print("=" * 60)
        
        endpoint_result = {
            "endpoint_name": endpoint_name,
            "endpoint_url": endpoint_url,
            "first_pass": None,
            "second_pass": None,
            "validation": None,
        }
        
        # Create LLM client
        llm_client = create_llm_client(endpoint)
        
        # ---- First Pass ----
        print(f"\n  [First Pass]")
        first_pass_result = run_first_pass(
            segments=word_segments,
            llm_client=llm_client,
            endpoint_name=endpoint_name,
        )
        
        if first_pass_result["success"]:
            print(f"    ✓ Completed in {first_pass_result['elapsed_seconds']}s")
            print(f"    - Redactions: {first_pass_result['redaction_count']}")
            print(f"    - Discovered names: {len(first_pass_result['discovered_names'])}")
        else:
            print(f"    ✗ Failed: {first_pass_result.get('error')}")
        
        endpoint_result["first_pass"] = first_pass_result
        
        # ---- Second Pass ----
        if first_pass_result["success"]:
            print(f"\n  [Second Pass]")
            
            # Get processed segments from first pass result
            def dict_to_word_replacement(r: Dict[str, Any]) -> WordReplacement:
                """Convert dict to WordReplacement, handling 'pass' -> 'pass_number' mapping."""
                if isinstance(r, WordReplacement):
                    return r
                # Map 'pass' (reserved keyword) to 'pass_number'
                kwargs = {k: v for k, v in r.items() if k != 'pass'}
                if 'pass' in r:
                    kwargs['pass_number'] = r['pass']
                return WordReplacement(**kwargs)
            
            first_pass_obj = FirstPassResult(
                segments=[],  # Not needed for second pass
                replacements=[
                    dict_to_word_replacement(r) if isinstance(r, dict) else r
                    for r in first_pass_result.get("replacements", [])
                ],
                discovered_names=set(first_pass_result.get("discovered_names", [])),
                session_data=first_pass_result.get("session_data", {}),
            )
            
            # Reconstruct segments with redactions (simulated)
            # In real usage, the segments would be modified in place
            processed_segments = _apply_replacements_to_segments(
                word_segments, 
                first_pass_obj.replacements
            )
            
            second_pass_result = run_second_pass(
                segments=processed_segments,
                first_pass_replacements=first_pass_obj.replacements,
                llm_client=llm_client,
                endpoint_name=endpoint_name,
            )
            
            if second_pass_result["success"]:
                print(f"    ✓ Completed in {second_pass_result['elapsed_seconds']}s")
                print(f"    - Additional redactions: {second_pass_result['additional_redactions']}")
                print(f"    - Names from list found: {len(second_pass_result['names_from_list_found'])}")
            else:
                print(f"    ✗ Failed: {second_pass_result.get('error')}")
            
            endpoint_result["second_pass"] = second_pass_result
            
            # ---- Validation ----
            print(f"\n  [Validation]")
            validation = validate_word_count_preservation(
                word_segments,
                processed_segments
            )
            
            if validation["word_count_preserved"]:
                print(f"    ✓ Word count preserved: {validation['original_word_count']} → {validation['processed_word_count']}")
            else:
                print(f"    ✗ Word count mismatch: {validation['original_word_count']} → {validation['processed_word_count']}")
            
            print(f"    - Total redaction tokens: {validation['processed_redaction_tokens']}")
            
            endpoint_result["validation"] = validation
        
        endpoint_results.append(endpoint_result)
    
    results["endpoints_tested"] = endpoint_results
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful_endpoints = sum(1 for e in endpoint_results if e["first_pass"]["success"])
    total_first_pass_time = sum(
        e["first_pass"].get("elapsed_seconds", 0) 
        for e in endpoint_results 
        if e["first_pass"]["success"]
    )
    total_second_pass_time = sum(
        e["second_pass"].get("elapsed_seconds", 0) 
        for e in endpoint_results 
        if e["second_pass"] and e["second_pass"]["success"]
    )
    
    results["summary"] = {
        "successful_endpoints": successful_endpoints,
        "total_endpoints": len(ENDPOINTS),
        "total_word_segments": len(word_segments),
        "avg_first_pass_time": round(total_first_pass_time / successful_endpoints, 2) if successful_endpoints else 0,
        "avg_second_pass_time": round(total_second_pass_time / successful_endpoints, 2) if successful_endpoints else 0,
    }
    
    print(f"  - Successful endpoints: {successful_endpoints}/{len(ENDPOINTS)}")
    print(f"  - Total WordSegments processed: {len(word_segments)}")
    print(f"  - Avg first pass time: {results['summary']['avg_first_pass_time']}s")
    print(f"  - Avg second pass time: {results['summary']['avg_second_pass_time']}s")
    
    # Save results
    _save_results(results)
    
    return results


def _apply_replacements_to_segments(
    segments: List[WordSegment],
    replacements: List[WordReplacement]
) -> List[WordSegment]:
    """
    Apply redactions to segments by replacing original words with [REDACTED].
    
    Args:
        segments: Original WordSegments
        replacements: List of WordReplacement objects
        
    Returns:
        New list of WordSegments with redactions applied
    """
    # Create a mapping of word index to replacement
    redaction_map = {r.word_index: "[REDACTED]" for r in replacements}
    
    # Apply redactions
    new_segments = []
    for i, segment in enumerate(segments):
        if i in redaction_map:
            new_segment = WordSegment(
                text=redaction_map[i],
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
            )
            new_segments.append(new_segment)
        else:
            new_segments.append(segment)
    
    return new_segments


def _save_results(results: Dict[str, Any]):
    """Save test results to JSON file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {RESULTS_FILE}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results = run_deid_endpoint_tests()
    
    # Print final status
    if results["transcription_info"]["success"]:
        print("\n" + "=" * 80)
        print("TEST SUITE COMPLETED SUCCESSFULLY")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TEST SUITE FAILED - Transcription error")
        print("=" * 80)
