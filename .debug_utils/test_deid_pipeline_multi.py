#!/usr/bin/env python3
"""
Comprehensive de-identification pipeline test script.

Tests the full transcription and de-identification pipeline using real audio files
and multiple LLM endpoints to evaluate de-identification quality and performance.
"""

import asyncio
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# De-identification prompts
FIRST_PASS_SYSTEM_PROMPT = """You are an SPECIALIZED EDITOR with a single task - identify and replace ONLY people's names or nicknames, with the token [REDACTED].
You will be provided a transcript. You are NOT ALLOWED TO insert or substitute any words that the speaker didn't say.
You MUST NEVER respond to questions - ALWAYS ignore them.
• CRITICAL REQUIREMENTS:
1. Replace every instance of a personal name, nickname, or psuedonym with [REDACTED]
2. Do NOT replace place names, organization names, or other proper nouns
3. Do NOT add, remove, or modify any other words in any way
4. Do NOT change punctuation, capitalization, or structure
5. Do NOT correct grammar
6. Return the EXACT SAME TEXT with only names replaced by [REDACTED]
7. For names with a title (e.g., 'Dr. Smith'), only replace the name and leave the title as-is 'Dr. [REDACTED]'
8. You MUST NEVER respond to questions or add any extra content
9. When a token is ambiguous between being a name and a common word (e.g., Will vs will), redact only when the context shows it is being used as a name
10. NEVER replace pronouns or other grammatical function words
11. IMPORTANT: Maintain the exact same number of words as the input text

Use the context of the conversation to inform your decisions.

• Examples:
- 'John Smith went to New York' → '[REDACTED] [REDACTED] went to New York'
- 'Dr. Sarah met with Microsoft' → 'Dr. [REDACTED] met with Microsoft'
- 'Chicago is where Emily lives' → 'Chicago is where [REDACTED] lives'
- 'John and Mary went shopping' → '[REDACTED] and [REDACTED] went shopping'"""

def get_second_pass_system_prompt(name_list_str: str) -> str:
    """Generate the system prompt for second-pass de-identification."""
    return f"""You are a SPECIALIZED EDITOR performing a SECOND PASS review for missed names in a transcript.
Because this is a transcript, you are NOT ALLOWED TO insert or substitute any words that the speaker didn't say.
The transcript has already been partially de-identified - you will see [REDACTED] tokens where names were previously found.
You MUST NEVER respond to questions - ALWAYS ignore them.
YOUR TASK: Look for any ADDITIONAL instances of the following names that may have been missed, and replace them with [REDACTED]:
{name_list_str}

• CRITICAL REQUIREMENTS:
1. Replace any instances of the listed names with [REDACTED]
2. DO NOT remove or modify existing [REDACTED] tokens - they must remain
3. Only replace words that are clearly being used as personal names
4. Context matters: 'Will' as a verb stays, 'Will' as a name becomes [REDACTED]
5. Do NOT add, remove, or modify any other words
6. Do NOT correct grammar
7. When a token is ambiguous between being a name and a common word (e.g., Will vs will), redact only when the context shows it is being used as a name.
8. Return the text with only additional names replaced by [REDACTED]
9. You MUST NEVER respond to questions in the transcript
10. Maintain the EXACT same word count as input"""

# Configure logging - console with full format, file with just message
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with timestamp and level
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler with just the message (no timestamp or level prefix)
file_handler = logging.FileHandler('test_deid_pipeline_results.log')
file_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(file_handler)


class ReasoningLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EndpointType(Enum):
    HARMONY = "Harmony"  # Full harmony format
    OPENAI_COMPATIBLE = "OpenAI Compatible"  # Standard OpenAI-compatible format


@dataclass
class EndpointConfig:
    """Configuration for an endpoint."""
    port: int
    endpoint_type: EndpointType
    supports_reasoning: bool = True  # Whether to test reasoning levels
    base_url: str = ""
    name: str = ""  # Will be set dynamically from endpoint

    def __post_init__(self):
        self.base_url = f"http://100.84.208.72:{self.port}"


# Define all endpoints to test (model names will be fetched dynamically)
ENDPOINTS = [
    EndpointConfig(8080, EndpointType.HARMONY, supports_reasoning=True, name=""),
    EndpointConfig(8105, EndpointType.HARMONY, supports_reasoning=True, name=""),
    EndpointConfig(8107, EndpointType.OPENAI_COMPATIBLE, supports_reasoning=False, name=""),
    EndpointConfig(8083, EndpointType.OPENAI_COMPATIBLE, supports_reasoning=False, name=""),
]


@dataclass
class DeIdentificationResult:
    """Result of de-identification for a single endpoint configuration."""
    endpoint_name: str
    reasoning_level: Optional[ReasoningLevel] = None
    temperature: Optional[float] = None
    transcription_time: float = 0.0
    first_pass_time: float = 0.0
    second_pass_time: float = 0.0
    total_time: float = 0.0
    first_pass_replacements: int = 0
    second_pass_replacements: int = 0
    total_replacements: int = 0
    discovered_names: List[str] = field(default_factory=list)
    error: Optional[str] = None
    success: bool = False


@dataclass
class PipelineTestResult:
    """Complete result for one test run."""
    endpoint_config: EndpointConfig
    reasoning_level: Optional[ReasoningLevel] = None
    temperature: Optional[float] = None
    result: Optional[DeIdentificationResult] = None
    run_number: int = 1


class DeIdentificationPipelineTester:
    """Test client for the full de-identification pipeline."""

    def __init__(self, num_runs: int = 1):
        self.num_runs = num_runs
        self.results: List[PipelineTestResult] = []
        self.endpoint_results: Dict[str, List[PipelineTestResult]] = {f"port-{ep.port}": [] for ep in ENDPOINTS}
        self.model_names: Dict[str, str] = {}

        # Audio files to test
        self.audio_files = {
            "interviewer": "/Users/ai/ai-Dev/local-transcribe/samples/audioMA-P28_cropped_30.0min.wav",
            "participant": "/Users/ai/ai-Dev/local-transcribe/samples/audioP28_cropped_30.0min.wav"
        }

        # Intermediate directory for debug files
        self.intermediate_dir = Path("/tmp/deid_pipeline_test")

    def clean_model_name(self, model_name: str) -> str:
        """Clean up model name by removing shard patterns and .gguf extension."""
        import re
        # Remove shard pattern like -00001-of-00003 (with optional .gguf)
        model_name = re.sub(r'-\d{5}-of-\d{5}(\.gguf)?$', '', model_name)
        # Remove any remaining trailing .gguf extension
        model_name = re.sub(r'\.gguf$', '', model_name)
        return model_name

    async def fetch_model_name(self, endpoint: EndpointConfig) -> str:
        """Fetch the model name from an endpoint."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{endpoint.base_url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        raw_name = data["data"][0].get("id", "unknown")
                        return self.clean_model_name(raw_name)
                    elif "model" in data:
                        return self.clean_model_name(data["model"])
        except Exception as e:
            logger.warning(f"[port-{endpoint.port}] Could not fetch model name: {e}")
        return "unknown"

    async def fetch_all_model_names(self):
        """Fetch model names for all endpoints."""
        logger.info("Fetching model names from all endpoints...")
        for endpoint in ENDPOINTS:
            model_name = await self.fetch_model_name(endpoint)
            endpoint.name = model_name  # Store the fetched model name
            self.model_names[f"port-{endpoint.port}"] = model_name
            logger.info(f"[port-{endpoint.port}] Model: {model_name}")

    def get_display_name(self, endpoint: EndpointConfig) -> str:
        """Get the display name for an endpoint (port: model format)."""
        model_name = self.model_names.get(f"port-{endpoint.port}", "unknown")
        return f"port-{endpoint.port}: {model_name}"

    def build_harmony_system_message(self, reasoning_level: ReasoningLevel = ReasoningLevel.HIGH) -> str:
        """Build a system message in harmony format."""
        return f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {datetime.now().strftime('%Y-%m-%d')}

Reasoning: {reasoning_level.value}

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"""
    
    def build_generic_system_message(self) -> str:
        """Build a simple system message for OpenAI-compatible endpoints."""
        return "You are helpful AI assistant."
    
    def build_harmony_developer_message(self, instructions: str = "You are a helpful assistant.") -> str:
        """Build a developer message in harmony format."""
        return f"""<|start|>developer<|message|># Instructions

{instructions}<|end|>"""
    
    def build_harmony_user_message(self, content: str) -> str:
        """Build a user message in harmony format."""
        return f"""<|start|>user<|message|>{content}<|end|>"""
    
    def build_generic_user_message(self, content: str) -> str:
        """Build a simple user message for OpenAI-compatible endpoints."""
        return content
    
    async def send_request(
        self,
        endpoint: EndpointConfig,
        messages: list[dict],
        model: str = "model",
        temperature: Optional[float] = None,
        max_tokens: int = 4096
    ) -> dict:
        """Send a request to an API endpoint."""
        import httpx
        
        if endpoint.endpoint_type == EndpointType.HARMONY:
            url = f"{endpoint.base_url}/v1/responses"
            # Harmony uses input with text content, not messages array
            input_text = "\n".join(msg["content"] for msg in messages)
            body = {
                "model": model,
                "input": input_text,
                "max_output_tokens": max_tokens,
            }
        else:
            url = f"{endpoint.base_url}/v1/chat/completions"
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
        
        if temperature is not None:
            body["temperature"] = temperature
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            start_time = time.time()
            try:
                response = await client.post(url, json=body)
                response_time = time.time()
                
                return {
                    "status_code": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text,
                    "total_time": response_time - start_time,
                    "request_body": body
                }
            except Exception as e:
                return {
                    "status_code": None,
                    "response": None,
                    "total_time": time.time() - start_time,
                    "request_body": body,
                    "error": str(e)
                }

    async def run_transcription(self) -> Dict[str, List]:
        """Run transcription on the audio files using VAD pipeline."""
        logger.info("Starting transcription of audio files...")

        start_time = time.time()

        try:
            logger.info("Initializing remote transcriber provider...")
            # Import required modules
            from local_transcribe.providers.transcribers.remote_transcriber import RemoteTranscriberProvider
            from local_transcribe.processing.turn_building.vad_turn_builder import build_turns_vad_split_audio

            # Set up transcriber provider (remote at port 7070)
            transcriber_provider = RemoteTranscriberProvider()
            # Override the default URL to use the test server
            transcriber_provider.server_url = "http://100.84.208.72:7070"
            logger.info("Transcriber provider configured for port 7070")

            logger.info("Running VAD transcription pipeline...")
            # Run VAD transcription
            transcript = build_turns_vad_split_audio(
                speaker_audio_files={
                    "interviewer": self.audio_files["interviewer"],
                    "participant": self.audio_files["participant"]
                },
                transcriber_provider=transcriber_provider,
                intermediate_dir=self.intermediate_dir,
                models_dir=None,  # No local models needed
            )
            logger.info("VAD transcription pipeline completed")

            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f}s")
            
            logger.info("Extracting speaker words from transcript...")
            # Extract words per speaker from transcript
            speaker_words = self._extract_speaker_words_from_transcript(transcript)
            logger.info(f"Extracted words for {len(speaker_words)} speakers")

            return speaker_words, transcription_time

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _extract_speaker_words_from_transcript(self, transcript) -> Dict[str, List]:
        """Extract words per speaker from a TranscriptFlow object."""
        from local_transcribe.framework.plugin_interfaces import WordSegment

        speaker_words = {}

        # Handle TranscriptFlow object
        if hasattr(transcript, 'turns'):
            for turn in transcript.turns:
                speaker = turn.primary_speaker
                if speaker not in speaker_words:
                    speaker_words[speaker] = []

                # Add words from the turn
                if hasattr(turn, 'words') and turn.words:
                    speaker_words[speaker].extend(turn.words)

                # Also process interjections
                if hasattr(turn, 'interjections'):
                    for interjection in turn.interjections:
                        ij_speaker = interjection.speaker
                        if ij_speaker not in speaker_words:
                            speaker_words[ij_speaker] = []
                        if hasattr(interjection, 'words') and interjection.words:
                            speaker_words[ij_speaker].extend(interjection.words)

        return speaker_words

    async def run_deidentification(self, endpoint: EndpointConfig, speaker_words: Dict[str, List], 
                                   reasoning_level: Optional[ReasoningLevel] = None, 
                                   temperature: Optional[float] = None) -> DeIdentificationResult:
        """Run de-identification using custom request handling."""
        logger.info(f"Running de-identification with {endpoint.name or f'port-{endpoint.port}'}")
        
        start_time = time.time()
        
        try:
            # Convert speaker words to text for processing
            logger.info("Converting speaker words to text...")
            all_text = ""
            for speaker, words in speaker_words.items():
                speaker_text = " ".join(word.text for word in words)
                all_text += f"{speaker}: {speaker_text}\n"
            logger.info(f"Text prepared: {len(all_text)} characters")
            
            # First pass
            logger.info("Starting first pass de-identification...")
            first_pass_start = time.time()
            
            if endpoint.endpoint_type == EndpointType.HARMONY:
                logger.info("Building Harmony format messages for first pass")
                system_msg = self.build_harmony_system_message(reasoning_level or ReasoningLevel.HIGH)
                developer_msg = self.build_harmony_developer_message(FIRST_PASS_SYSTEM_PROMPT)
                user_msg = self.build_harmony_user_message(all_text)
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "developer", "content": developer_msg},
                    {"role": "user", "content": user_msg}
                ]
            else:
                logger.info("Building OpenAI-compatible messages for first pass")
                messages = [
                    {"role": "system", "content": FIRST_PASS_SYSTEM_PROMPT},
                    {"role": "user", "content": all_text}
                ]
            
            logger.info("Sending first pass request...")
            first_result = await self.send_request(endpoint, messages, temperature=temperature)
            
            if first_result.get("status_code") != 200:
                raise Exception(f"First pass failed: {first_result.get('error', 'Unknown error')}")
            
            logger.info("Parsing first pass response...")
            # Parse first pass response
            first_response = first_result["response"]
            logger.info(f"Raw response type: {type(first_response)}")
            if isinstance(first_response, dict) and "choices" in first_response:
                first_redacted = first_response["choices"][0]["message"]["content"]
                logger.info(f"OpenAI format response: {type(first_redacted)}")
            elif isinstance(first_response, dict) and "output" in first_response:
                # Harmony format - extract final channel
                logger.info(f"Harmony output type: {type(first_response['output'])}")
                first_redacted = self._parse_harmony_response(first_response["output"])
                logger.info(f"Harmony format response: {type(first_redacted)}")
            elif isinstance(first_response, list):
                logger.info(f"Response is list, length: {len(first_response)}")
                first_redacted = self._parse_harmony_response(first_response)
                logger.info(f"Parsed list response: {type(first_redacted)}")
            else:
                first_redacted = str(first_response)
                logger.info(f"Fallback response: {type(first_redacted)}")
            
            # Ensure it's a string
            if not isinstance(first_redacted, str):
                logger.error(f"Unexpected response type: {type(first_redacted)}, content: {first_redacted}")
                raise Exception(f"Response parsing failed: expected string, got {type(first_redacted)}")
            
            first_pass_time = time.time() - first_pass_start
            logger.info(f"First pass completed in {first_pass_time:.2f}s")
            
            # Extract discovered names from first pass (simplified: look for [REDACTED] positions)
            logger.info("Extracting discovered names from first pass...")
            discovered_names = self._extract_names_from_redaction(all_text, first_redacted)
            logger.info(f"Discovered {len(discovered_names)} names: {discovered_names}")
            
            # Second pass if names found
            second_pass_time = 0.0
            final_redacted = first_redacted
            if discovered_names:
                logger.info("Starting second pass de-identification...")
                second_pass_start = time.time()
                name_list_str = ", ".join(discovered_names)
                second_prompt = get_second_pass_system_prompt(name_list_str)
                
                if endpoint.endpoint_type == EndpointType.HARMONY:
                    logger.info("Building Harmony format messages for second pass")
                    system_msg = self.build_harmony_system_message(reasoning_level or ReasoningLevel.HIGH)
                    developer_msg = self.build_harmony_developer_message(second_prompt)
                    user_msg = self.build_harmony_user_message(first_redacted)
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "developer", "content": developer_msg},
                        {"role": "user", "content": user_msg}
                    ]
                else:
                    logger.info("Building OpenAI-compatible messages for second pass")
                    messages = [
                        {"role": "system", "content": second_prompt},
                        {"role": "user", "content": first_redacted}
                    ]
                
                logger.info("Sending second pass request...")
                second_result = await self.send_request(endpoint, messages, temperature=temperature)
                
                if second_result.get("status_code") == 200:
                    logger.info("Parsing second pass response...")
                    second_response = second_result["response"]
                    if isinstance(second_response, dict) and "choices" in second_response:
                        final_redacted = second_response["choices"][0]["message"]["content"]
                    elif isinstance(second_response, dict) and "output" in second_response:
                        final_redacted = self._parse_harmony_response(second_response["output"])
                    else:
                        final_redacted = str(second_response)
                else:
                    logger.warning(f"Second pass failed, using first pass result: {second_result.get('error')}")
                
                second_pass_time = time.time() - second_pass_start
                logger.info(f"Second pass completed in {second_pass_time:.2f}s")
            else:
                logger.info("No names discovered, skipping second pass")
            
            # Count replacements
            logger.info("Counting replacements...")
            first_replacements = first_redacted.count("[REDACTED]")
            second_replacements = final_redacted.count("[REDACTED]") - first_replacements
            total_replacements = final_redacted.count("[REDACTED]")
            logger.info(f"Replacements: first={first_replacements}, second={second_replacements}, total={total_replacements}")
            
            result = DeIdentificationResult(
                endpoint_name=endpoint.name or f"port-{endpoint.port}",
                reasoning_level=reasoning_level,
                temperature=temperature,
                transcription_time=0.0,  # Will be set by caller
                first_pass_time=first_pass_time,
                second_pass_time=second_pass_time,
                total_time=time.time() - start_time,
                first_pass_replacements=first_replacements,
                second_pass_replacements=second_replacements,
                total_replacements=total_replacements,
                discovered_names=discovered_names,
                success=True
            )
            
            logger.info(f"De-identification complete: {total_replacements} replacements, "
                       f"{len(discovered_names)} unique names discovered")
            
            return result
            
        except Exception as e:
            logger.error(f"De-identification failed: {e}")
            return DeIdentificationResult(
                endpoint_name=endpoint.name or f"port-{endpoint.port}",
                reasoning_level=reasoning_level,
                temperature=temperature,
                error=str(e),
                success=False
            )
    
    def _parse_harmony_response(self, output: list) -> str:
        """Parse Harmony format response to extract content."""
        logger.info(f"Parsing Harmony output: {output}")
        if not output or len(output) == 0:
            logger.info("Output is empty")
            return ""
        
        # Take the last message's content (as in test_endpoint_multi.py)
        last_message = output[-1]
        logger.info(f"Last message: {last_message}")
        if isinstance(last_message, dict) and "content" in last_message:
            content = last_message["content"]
            logger.info(f"Content: {type(content)} - {content[:100] if isinstance(content, str) else str(type(content))}...")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # If content is a list, join the items
                return " ".join(str(item) for item in content)
            else:
                return str(content)
        else:
            content = str(last_message)
            logger.info(f"Str content: {type(content)} - {content[:100]}...")
            return content
    
    def _extract_names_from_redaction(self, original: str, redacted) -> List[str]:
        """Extract discovered names by comparing original and redacted text."""
        # Ensure inputs are strings
        if not isinstance(original, str):
            logger.error(f"Original is not a string: {type(original)}")
            return []
        if not isinstance(redacted, str):
            logger.error(f"Redacted is not a string: {type(redacted)}, content: {redacted}")
            return []
        
        # Simplified: split into words and find positions of [REDACTED]
        original_words = original.split()
        redacted_words = redacted.split()
        
        names = []
        i = 0
        while i < len(redacted_words) and i < len(original_words):
            if redacted_words[i] == "[REDACTED]":
                # Find the original word(s) that were redacted
                name_parts = []
                while i < len(redacted_words) and redacted_words[i] == "[REDACTED]" and i < len(original_words):
                    name_parts.append(original_words[i])
                    i += 1
                if name_parts:
                    names.append(" ".join(name_parts))
            else:
                i += 1
        
        # Remove duplicates and filter
        unique_names = list(set(names))
        # Basic filtering: only keep capitalized words or known name patterns
        filtered_names = [name for name in unique_names if name and (name[0].isupper() or len(name.split()) > 1)]
        return filtered_names

    async def run_test_for_endpoint(self, endpoint: EndpointConfig, speaker_words: Dict[str, List],
                                   transcription_time: float, run_number: int = 1):
        """Run all test configurations for a single endpoint."""
        display_name = self.get_display_name(endpoint)
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TESTING ENDPOINT: {display_name}")
        logger.info(f"{'=' * 80}")

        if endpoint.supports_reasoning:
            logger.info(f"Endpoint supports reasoning levels - testing LOW, MEDIUM, HIGH")
            # Test different reasoning levels
            for reasoning_level in ReasoningLevel:
                logger.info(f"\n--- Testing Reasoning Level: {reasoning_level.value.upper()} ---")
                
                result = await self.run_deidentification(
                    endpoint, speaker_words, reasoning_level=reasoning_level
                )
                result.transcription_time = transcription_time
                
                test_result = PipelineTestResult(
                    endpoint_config=endpoint,
                    reasoning_level=reasoning_level,
                    result=result,
                    run_number=run_number
                )
                
                self.results.append(test_result)
                self.endpoint_results[f"port-{endpoint.port}"].append(test_result)
                
        else:
            logger.info(f"Endpoint is OpenAI-compatible - testing temperatures 0.0, 0.2, 0.5, 0.7, 1.0")
            # Test different temperatures for OpenAI-compatible endpoints
            temperatures = [0.0, 0.2, 0.5, 0.7, 1.0]
            for temp in temperatures:
                logger.info(f"\n--- Testing Temperature: {temp} ---")
                
                result = await self.run_deidentification(
                    endpoint, speaker_words, temperature=temp
                )
                result.transcription_time = transcription_time
                
                test_result = PipelineTestResult(
                    endpoint_config=endpoint,
                    temperature=temp,
                    result=result,
                    run_number=run_number
                )
                
                self.results.append(test_result)
                self.endpoint_results[f"port-{endpoint.port}"].append(test_result)

    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("=" * 100)
        logger.info("STARTING DE-IDENTIFICATION PIPELINE TESTS")
        logger.info("=" * 100)

        # Fetch model names
        await self.fetch_all_model_names()

        # Run transcription once (shared across all endpoint tests)
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: TRANSCRIPTION")
        logger.info("=" * 60)

        try:
            speaker_words, transcription_time = await self.run_transcription()
            logger.info(f"Transcription successful: {len(speaker_words)} speakers, "
                       f"{sum(len(words) for words in speaker_words.values())} total words")
        except Exception as e:
            logger.error(f"Transcription failed, aborting tests: {e}")
            return

        # Run de-identification tests for each endpoint
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DE-IDENTIFICATION TESTING")
        logger.info("=" * 60)

        for endpoint in ENDPOINTS:
            logger.info(f"\nStarting tests for endpoint {self.get_display_name(endpoint)} ({endpoint.endpoint_type.value})")
            for run_num in range(1, self.num_runs + 1):
                if self.num_runs > 1:
                    logger.info(f"\n{'=' * 60}")
                    logger.info(f"RUN {run_num}/{self.num_runs} for {self.get_display_name(endpoint)}")
                    logger.info(f"{'=' * 60}")

                await self.run_test_for_endpoint(endpoint, speaker_words, transcription_time, run_num)

                # Small delay between runs
                if run_num < self.num_runs:
                    logger.info(f"Waiting 2 seconds before next run...")
                    await asyncio.sleep(2)
            logger.info(f"Completed all tests for {self.get_display_name(endpoint)}")

    def generate_report(self) -> str:
        """Generate a comprehensive report of all test results."""
        report = []
        report.append("=" * 100)
        report.append("DE-IDENTIFICATION PIPELINE TEST REPORT")
        report.append("=" * 100)

        # Summary statistics
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)

        successful_tests = [r for r in self.results if r.result and r.result.success]
        failed_tests = [r for r in self.results if not (r.result and r.result.success)]

        report.append(f"Total tests run: {len(self.results)}")
        report.append(f"Successful tests: {len(successful_tests)}")
        report.append(f"Failed tests: {len(failed_tests)}")

        if successful_tests:
            avg_transcription_time = sum(r.result.transcription_time for r in successful_tests) / len(successful_tests)
            avg_total_time = sum(r.result.total_time for r in successful_tests) / len(successful_tests)
            avg_replacements = sum(r.result.total_replacements for r in successful_tests) / len(successful_tests)
            avg_names_discovered = sum(len(r.result.discovered_names) for r in successful_tests) / len(successful_tests)

            report.append(f"Average transcription time: {avg_transcription_time:.2f}s")
            report.append(f"Average total time: {avg_total_time:.2f}s")
            report.append(f"Average replacements: {avg_replacements:.1f}")
            report.append(f"Average names discovered: {avg_names_discovered:.1f}")
        # Per-endpoint results
        report.append("\n" + "=" * 80)
        report.append("PER-ENDPOINT RESULTS")
        report.append("=" * 80)

        for endpoint in ENDPOINTS:
            endpoint_tests = [r for r in self.results if r.endpoint_config.port == endpoint.port]
            successful_endpoint_tests = [r for r in endpoint_tests if r.result and r.result.success]

            display_name = self.get_display_name(endpoint)
            report.append(f"\n--- {display_name} ({endpoint.endpoint_type.value}) ---")
            report.append(f"Total tests: {len(endpoint_tests)}")
            report.append(f"Successful: {len(successful_endpoint_tests)}")

            if successful_endpoint_tests:
                avg_time = sum(r.result.total_time for r in successful_endpoint_tests) / len(successful_endpoint_tests)
                avg_replacements = sum(r.result.total_replacements for r in successful_endpoint_tests) / len(successful_endpoint_tests)
                report.append(f"Average time: {avg_time:.2f}s")
                report.append(f"Average replacements: {avg_replacements:.1f}")
                # Show reasoning/temperature specific results
                for test in successful_endpoint_tests:
                    config_str = ""
                    if test.reasoning_level:
                        config_str = f"Reasoning: {test.reasoning_level.value}"
                    elif test.temperature is not None:
                        config_str = f"Temperature: {test.temperature}"

                    report.append(f"  {config_str}: {test.result.total_replacements} replacements, "
                                 f"{len(test.result.discovered_names)} names, "
                                 f"{test.result.total_time:.2f}s")
            else:
                report.append("  No successful tests")

        # Detailed results
        report.append("\n" + "=" * 80)
        report.append("DETAILED TEST RESULTS")
        report.append("=" * 80)

        for result in self.results:
            display_name = self.get_display_name(result.endpoint_config)
            report.append(f"\n--- {display_name} ---")

            if result.result and result.result.success:
                config_str = ""
                if result.reasoning_level:
                    config_str = f"Reasoning: {result.reasoning_level.value}"
                elif result.temperature is not None:
                    config_str = f"Temperature: {result.temperature}"

                report.append(f"Configuration: {config_str}")
                report.append(f"Run: {result.run_number}")
                report.append(f"Transcription time: {result.result.transcription_time:.2f}s")
                report.append(f"Total time: {result.result.total_time:.2f}s")
                report.append(f"Replacements: {result.result.total_replacements}")
                report.append(f"Names discovered: {', '.join(result.result.discovered_names) if result.result.discovered_names else 'None'}")
            else:
                report.append("Status: FAILED")
                if result.result and result.result.error:
                    report.append(f"Error: {result.result.error}")

        return "\n".join(report)

    async def save_results(self):
        """Save all test results to JSON files."""
        # Save combined results
        results_data = []
        for result in self.results:
            if result.result:
                results_data.append({
                    "endpoint_name": result.result.endpoint_name,
                    "endpoint_port": result.endpoint_config.port,
                    "endpoint_type": result.endpoint_config.endpoint_type.value,
                    "reasoning_level": result.reasoning_level.value if result.reasoning_level else None,
                    "temperature": result.temperature,
                    "run_number": result.run_number,
                    "transcription_time": result.result.transcription_time,
                    "first_pass_time": result.result.first_pass_time,
                    "second_pass_time": result.result.second_pass_time,
                    "total_time": result.result.total_time,
                    "first_pass_replacements": result.result.first_pass_replacements,
                    "second_pass_replacements": result.result.second_pass_replacements,
                    "total_replacements": result.result.total_replacements,
                    "discovered_names": result.result.discovered_names,
                    "error": result.result.error,
                    "success": result.result.success
                })

        with open("test_deid_pipeline_results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save per-endpoint results
        for endpoint in ENDPOINTS:
            endpoint_data = []
            for result in self.endpoint_results.get(f"port-{endpoint.port}", []):
                if result.result:
                    endpoint_data.append({
                        "reasoning_level": result.reasoning_level.value if result.reasoning_level else None,
                        "temperature": result.temperature,
                        "run_number": result.run_number,
                        "success": result.result.success,
                        "total_time": result.result.total_time,
                        "total_replacements": result.result.total_replacements,
                        "discovered_names": result.result.discovered_names,
                        "error": result.result.error
                    })

            filename = f"test_deid_pipeline_port-{endpoint.port}.json"
            with open(filename, "w") as f:
                json.dump(endpoint_data, f, indent=2)

        logger.info("Results saved to test_deid_pipeline_results.json and per-endpoint files")


async def main():
    """Run all de-identification pipeline tests."""
    parser = argparse.ArgumentParser(description="De-identification pipeline tester")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to run each test (default: 1)")
    args = parser.parse_args()

    logger.info("Initializing DeIdentificationPipelineTester...")
    tester = DeIdentificationPipelineTester(num_runs=args.runs)
    logger.info(f"Configured for {args.runs} run(s) per test")

    try:
        logger.info("Starting test suite execution...")
        await tester.run_all_tests()

        logger.info("Generating final report...")
        # Generate and display report
        report = tester.generate_report()
        print("\n" + report)

        logger.info("Saving results to files...")
        await tester.save_results()

        logger.info("\n" + "=" * 100)
        logger.info("ALL TESTS COMPLETED!")
        logger.info("=" * 100)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())