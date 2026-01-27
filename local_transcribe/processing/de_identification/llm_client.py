#!/usr/bin/env python3
"""
LLM client for de-identification with auto-detection of response format.

Handles communication with the LLM server, including retry logic,
response parsing, and automatic detection of Harmony format.
"""

import json
import re
import time
import requests
from typing import Optional, Tuple, List, Dict, Any, Callable

from local_transcribe.lib.program_logger import log_progress, log_debug

from .core import (
    DeIdentificationConfig,
    ValidationResult,
    ChunkProcessingResult,
    DEFAULT_CONFIG,
)


class LLMDeIdentifierClient:
    """
    Client for communicating with LLM server for de-identification.
    
    Features:
    - Auto-detection of Harmony response format
    - Retry logic with temperature decay
    - Configurable validation
    """
    
    # Test prompt for format detection
    HARMONY_DETECTION_PROMPT = "Reply with exactly: TEST_RESPONSE"
    HARMONY_DETECTION_SYSTEM = "You are a helpful assistant. Respond only with the exact text requested."
    
    def __init__(
        self,
        llm_url: str = "http://0.0.0.0:8080",
        config: Optional[DeIdentificationConfig] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            llm_url: URL of the LLM server
            config: Configuration options (uses defaults if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        
        # Ensure URL has scheme
        if not llm_url.startswith(('http://', 'https://')):
            llm_url = f'http://{llm_url}'
        self.llm_url = llm_url
        
        # Harmony format detection (lazy initialization)
        self._harmony_format_detected: Optional[bool] = None
        self._harmony_detection_done: bool = False
    
    @property
    def uses_harmony_format(self) -> bool:
        """
        Check if the LLM uses Harmony response format.
        
        Auto-detects on first call if not explicitly configured.
        """
        # If explicitly configured, use that
        if self.config.parse_harmony is not None:
            return self.config.parse_harmony
        
        # Auto-detect if not done yet
        if not self._harmony_detection_done:
            self._detect_harmony_format()
        
        return self._harmony_format_detected or False
    
    def _detect_harmony_format(self) -> None:
        """
        Detect if the LLM server returns Harmony-formatted responses.
        
        Sends a simple test request and checks if the response contains
        Harmony format tokens like <|channel|>, <|message|>, etc.
        """
        self._harmony_detection_done = True
        self._harmony_format_detected = False
        
        log_debug("Detecting LLM response format...")
        
        payload = {
            "messages": [
                {"role": "system", "content": self.HARMONY_DETECTION_SYSTEM},
                {"role": "user", "content": self.HARMONY_DETECTION_PROMPT}
            ],
            "temperature": 0.0,  # Deterministic for detection
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.llm_url}/chat/completions",
                json=payload,
                timeout=30  # Short timeout for detection
            )
            response.raise_for_status()
            
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"]
            
            # Check for Harmony format markers
            harmony_markers = ['<|channel|>', '<|message|>', '<|end|>', '<|return|>', '<|start|>']
            has_harmony_markers = any(marker in raw_response for marker in harmony_markers)
            
            if has_harmony_markers:
                log_progress("LLM response format detected: Harmony (will parse special tokens)")
                self._harmony_format_detected = True
            else:
                log_progress("LLM response format detected: Plain text")
                self._harmony_format_detected = False
                
            log_debug(f"Detection response sample: {raw_response[:200]}...")
            
        except Exception as e:
            log_progress(f"Could not auto-detect LLM format (using plain text): {e}")
            self._harmony_format_detected = False
    
    def parse_response(self, raw_response: str) -> str:
        """
        Parse LLM response, handling Harmony format if detected.
        
        Args:
            raw_response: Raw response from LLM
            
        Returns:
            Extracted text content
        """
        if not self.uses_harmony_format:
            return raw_response.strip()
        
        return self._parse_harmony_response(raw_response)
    
    def _parse_harmony_response(self, raw_response: str) -> str:
        """
        Parse a Harmony-formatted response to extract the final channel content.
        
        Harmony format (used by gpt-oss models) uses special tokens:
        - <|channel|>analysis<|message|>... - Chain of thought (internal)
        - <|channel|>final<|message|>... - Final user-facing response
        - <|end|> / <|return|> - End markers
        
        Returns:
            The content from the 'final' channel, or the raw response if no format detected.
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
    
    def process_chunk(
        self,
        text: str,
        system_prompt: str,
        validator: Callable[[str, str], ValidationResult],
        extra_validation_args: Optional[Dict[str, Any]] = None
    ) -> ChunkProcessingResult:
        """
        Process a text chunk with the LLM, with retry logic.
        
        Args:
            text: Input text to process
            system_prompt: System prompt for the LLM
            validator: Function to validate output (original, processed) -> ValidationResult
            extra_validation_args: Additional args to pass to validator
            
        Returns:
            ChunkProcessingResult with processed text and metadata
        """
        extra_validation_args = extra_validation_args or {}
        
        # Track all attempts for debugging
        all_attempts: List[Dict[str, Any]] = []
        total_response_time_ms = 0.0
        last_raw_response: Optional[str] = None
        last_validation: Optional[ValidationResult] = None
        
        max_retries = self.config.max_retries
        initial_temperature = self.config.temperature
        temperature_decay = self.config.temperature_decay
        
        # Try with retries, decreasing temperature on each failure
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
                log_progress(f"Retry {attempt}/{max_retries} with temperature {current_temperature:.2f}")
            
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "temperature": current_temperature,
                "stream": False
            }
            
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.llm_url}/chat/completions",
                    json=payload,
                    timeout=self.config.llm_timeout
                )
                response.raise_for_status()
                response_time_ms = (time.time() - start_time) * 1000
                total_response_time_ms += response_time_ms
                
                result = response.json()
                raw_response = result["choices"][0]["message"]["content"]
                last_raw_response = raw_response
                
                # Parse response (handles Harmony format if needed)
                processed_text = self.parse_response(raw_response)
                
                # Validate output
                validation = validator(text, processed_text, **extra_validation_args)
                
                attempt_info.update({
                    'validation_passed': validation.passed,
                    'validation_reason': validation.reason,
                    'response_time_ms': response_time_ms,
                    'raw_response': raw_response,
                    'processed_text': processed_text,
                    'validation_result': validation.to_dict(),
                })
                all_attempts.append(attempt_info)
                
                if validation.passed:
                    # Add retry info to validation
                    # Note: Don't store all_attempts reference here to avoid circular reference
                    # when serializing to JSON (attempts are available via attempt_logs)
                    validation.details['total_attempts'] = attempt_number
                    validation.details['final_temperature'] = current_temperature
                    
                    return ChunkProcessingResult(
                        processed_text=processed_text,
                        response_time_ms=total_response_time_ms,
                        validation=validation,
                        raw_response=raw_response,
                        attempt_logs=all_attempts,
                    )
                else:
                    last_validation = validation
                    
            except requests.RequestException as e:
                log_progress(f"LLM request failed (attempt {attempt_number}): {e}")
                attempt_info['error'] = f'request failed: {e}'
                all_attempts.append(attempt_info)
                last_validation = ValidationResult(
                    passed=False,
                    reason=f'request failed: {e}',
                    details={}
                )
                
            except (KeyError, json.JSONDecodeError) as e:
                log_progress(f"LLM response parsing error (attempt {attempt_number}): {e}")
                attempt_info['error'] = f'parsing error: {e}'
                all_attempts.append(attempt_info)
                last_validation = ValidationResult(
                    passed=False,
                    reason=f'parsing error: {e}',
                    details={}
                )
        
        # All retries exhausted - fall back to original text
        log_progress(
            f"All {max_retries + 1} attempts failed, "
            "falling back to original text (no de-identification for this chunk)"
        )
        
        # Build final validation result with all attempt info
        final_validation = last_validation or ValidationResult(
            passed=False,
            reason='all attempts failed',
            details={}
        )
        # Note: Don't store all_attempts reference here to avoid circular reference
        # when serializing to JSON (attempts are available via attempt_logs)
        final_validation.details['total_attempts'] = max_retries + 1
        final_validation.details['all_attempts_failed'] = True
        
        return ChunkProcessingResult(
            processed_text=text,  # Fall back to original
            response_time_ms=total_response_time_ms,
            validation=final_validation,
            raw_response=last_raw_response,
            attempt_logs=all_attempts,
        )
    
    def is_available(self) -> bool:
        """Check if the LLM server is available."""
        try:
            # Use a simple health check or the detection endpoint
            response = requests.get(
                f"{self.llm_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            # Try the models endpoint as fallback
            try:
                response = requests.get(
                    f"{self.llm_url}/v1/models",
                    timeout=5
                )
                return response.status_code == 200
            except Exception:
                return False
