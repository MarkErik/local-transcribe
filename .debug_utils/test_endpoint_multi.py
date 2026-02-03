#!/usr/bin/env python3
"""
Comprehensive multi-endpoint test script for LLM endpoints.
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

# Configure logging - console with full format, file with just message
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with timestamp and level
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# File handler with just the message (no timestamp or level prefix)
file_handler = logging.FileHandler('test_results_multi.log')
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
    EndpointConfig(8099, EndpointType.OPENAI_COMPATIBLE, supports_reasoning=False, name=""),
]


@dataclass
class TestResult:
    """Stores the result of a single test."""
    endpoint_name: str
    test_name: str
    run_number: int
    reasoning_level: Optional[ReasoningLevel] = None
    temperature: Optional[float] = None
    request_body: dict = field(default_factory=dict)
    response_body: dict = field(default_factory=dict)
    total_time: float = 0.0
    status_code: Optional[int] = None
    passed: bool = False
    error: Optional[str] = None
    instruction_followed: Optional[bool] = None
    response_content: Optional[str] = None


class MultiEndpointTester:
    """Test client for multiple LLM endpoints."""
    
    def __init__(self, num_runs: int = 2):
        self.num_runs = num_runs
        self.results: List[TestResult] = []
        self.endpoint_results: Dict[str, List[TestResult]] = {f"port-{ep.port}": [] for ep in ENDPOINTS}
        self.model_names: Dict[str, str] = {}
    
    def clean_model_name(self, model_name: str) -> str:
        """Clean up model name by removing shard patterns and .gguf extension."""
        import re
        # Remove shard pattern like -00001-of-00003
        model_name = re.sub(r'-\d{5}-of-\d{3}\.gguf$', '', model_name)
        # Remove trailing .gguf extension
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
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        max_tokens: int = 4096
    ) -> dict:
        """Send a request to an API endpoint."""
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
        
        import httpx
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
    
    def get_messages(self, endpoint: EndpointConfig, system_msg: str, developer_msg: str, user_msg: str) -> list[dict]:
        """Get properly formatted messages based on endpoint type."""
        if endpoint.endpoint_type == EndpointType.HARMONY:
            return [
                {"role": "system", "content": system_msg},
                {"role": "developer", "content": developer_msg},
                {"role": "user", "content": user_msg}
            ]
        else:
            # For OpenAI-compatible endpoints, use simple format
            return [
                {"role": "system", "content": self.build_generic_system_message()},
                {"role": "user", "content": user_msg}
            ]
    
    async def run_test(
        self,
        endpoint: EndpointConfig,
        test_name: str,
        system_message: str,
        user_message: str,
        developer_message: str = "",
        expected_behavior: Optional[str] = None,
        reasoning_level: Optional[ReasoningLevel] = None,
        temperature: Optional[float] = None,
        run_number: int = 1
    ) -> TestResult:
        """Run a single test and record the result."""
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] Running test: {test_name} (run {run_number})")
        
        messages = self.get_messages(endpoint, system_message, developer_message, user_message)
        result = await self.send_request(endpoint, messages, temperature=temperature)
        
        response_content = None
        if result.get("response") and isinstance(result["response"], dict):
            if "choices" in result["response"]:
                # OpenAI-compatible format
                response_content = result["response"]["choices"][0]["message"].get("content", "")
            elif "output" in result["response"]:
                # Harmony format - output is an array of messages
                output = result["response"]["output"]
                if output and len(output) > 0:
                    content_list = output[0].get("content", [])
                    if content_list and len(content_list) > 0:
                        response_content = content_list[0].get("text", "")
        
        test_result = TestResult(
            endpoint_name=endpoint.name or f"port-{endpoint.port}",
            test_name=test_name,
            run_number=run_number,
            reasoning_level=reasoning_level,
            temperature=temperature,
            request_body=result.get("request_body", {}),
            response_body=result.get("response", {}) if isinstance(result.get("response"), dict) else {},
            total_time=result.get("total_time", 0),
            status_code=result.get("status_code"),
            error=result.get("error"),
            response_content=response_content
        )
        
        # Check if instruction was followed
        if expected_behavior and response_content:
            test_result.instruction_followed = expected_behavior.lower() in response_content.lower()
            test_result.passed = test_result.instruction_followed
        elif result.get("status_code") == 200:
            test_result.passed = True
        
        self.results.append(test_result)
        self.endpoint_results[f"port-{endpoint.port}"].append(test_result)

        # Log the result
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] Test '{test_name}' (run {run_number}) - "
                   f"Status: {test_result.status_code}, Time: {test_result.total_time:.2f}s, "
                   f"Passed: {test_result.passed}")
        
        return test_result
    
    async def test_reasoning_levels(self, endpoint: EndpointConfig):
        """Test different reasoning levels (only if endpoint supports it)."""
        if not endpoint.supports_reasoning:
            logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] Skipping reasoning levels test (not supported)")
            return
        
        logger.info("=" * 60)
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] TESTING REASONING LEVELS")
        logger.info("=" * 60)
        
        test_prompts = [
            "What is 15 + 27? Show your work step by step.",
            "Explain the concept of recursion in programming.",
            "What are the pros and cons of renewable energy?",
        ]
        
        for reasoning_level in ReasoningLevel:
            logger.info(f"\n--- [{endpoint.name or f'port-{endpoint.port}'}] Testing Reasoning Level: {reasoning_level.value.upper()} ---")
            
            for i, prompt in enumerate(test_prompts):
                test_name = f"reasoning_{reasoning_level.value}_{i+1}"
                
                system_msg = self.build_harmony_system_message(reasoning_level)
                developer_msg = self.build_harmony_developer_message("You are a helpful assistant.")
                
                await self.run_test(
                    endpoint=endpoint,
                    test_name=test_name,
                    system_message=system_msg,
                    developer_message=developer_msg,
                    user_message=prompt,
                    reasoning_level=reasoning_level
                )
    
    async def test_temperatures(self, endpoint: EndpointConfig):
        """Test different temperature values."""
        logger.info("=" * 60)
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] TESTING TEMPERATURES")
        logger.info("=" * 60)
        
        temperatures = [0.0, 0.3, 0.7, 1.0, 1.2]
        test_prompt = "Write a short creative story about a robot learning to paint."
        
        for temp in temperatures:
            logger.info(f"\n--- [{endpoint.name or f'port-{endpoint.port}'}] Testing Temperature: {temp} ---")
            
            test_name = f"temperature_{temp}"
            
            if endpoint.endpoint_type == EndpointType.HARMONY:
                system_msg = self.build_harmony_system_message(ReasoningLevel.MEDIUM)
                developer_msg = self.build_harmony_developer_message("You are a creative storyteller.")
            else:
                system_msg = self.build_generic_system_message()
            
            if endpoint.endpoint_type == EndpointType.HARMONY:
                await self.run_test(
                    endpoint=endpoint,
                    test_name=test_name,
                    system_message=system_msg,
                    developer_message=developer_msg,
                    user_message=test_prompt,
                    temperature=temp
                )
            else:
                await self.run_test(
                    endpoint=endpoint,
                    test_name=test_name,
                    system_message=system_msg,
                    user_message=test_prompt,
                    temperature=temp
                )
    
    async def test_instruction_following(self, endpoint: EndpointConfig):
        """Test exact instruction following."""
        logger.info("=" * 60)
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] TESTING INSTRUCTION FOLLOWING")
        logger.info("=" * 60)
        
        instruction_tests = [
            {
                "name": "exact_word_count_50",
                "instructions": "You must respond with exactly 50 words. Count them carefully.",
                "user_prompt": "Describe the ocean in exactly 50 words.",
                "expected": "exactly 50 words"
            },
            {
                "name": "no_punctuation",
                "instructions": "You must not use any punctuation marks in your response. No periods, commas, exclamation marks, or question marks.",
                "user_prompt": "Say hello",
                "expected": "hello"
            },
            {
                "name": "single_word_response",
                "instructions": "You must respond with only a single word. No explanations, no punctuation.",
                "user_prompt": "What is the capital of France",
                "expected": "paris"
            },
            {
                "name": "reverse_order",
                "instructions": "You must reverse the order of the words in your response.",
                "user_prompt": "The quick brown fox",
                "expected": "fox brown quick"
            },
            {
                "name": "uppercase_only",
                "instructions": "You must respond in all uppercase letters only.",
                "user_prompt": "Say something in uppercase",
                "expected": "uppercase"
            },
            {
                "name": "no_vowels",
                "instructions": "You must not use any vowels (a, e, i, o, u) in your response.",
                "user_prompt": "Say hello without vowels",
                "expected": "hello"
            },
            {
                "name": "start_with_specific_word",
                "instructions": "Your response must start with the word 'Certainly'.",
                "user_prompt": "Can you help me with this",
                "expected": "certainly"
            },
            {
                "name": "end_with_specific_word",
                "instructions": "Your response must end with the word 'forever'.",
                "user_prompt": "Write a short phrase about love",
                "expected": "forever"
            },
            {
                "name": "json_format",
                "instructions": "You must respond in valid JSON format only. No other text.",
                "user_prompt": "Return a JSON object with a 'name' field set to 'test'",
                "expected": "{\"name\":"
            },
            {
                "name": "numbered_list",
                "instructions": "You must respond as a numbered list with exactly 3 items.",
                "user_prompt": "List three colors",
                "expected": "1."
            }
        ]
        
        for test in instruction_tests:
            logger.info(f"\n--- [{endpoint.name or f'port-{endpoint.port}'}] Testing: {test['name']} ---")
            
            if endpoint.endpoint_type == EndpointType.HARMONY:
                system_msg = self.build_harmony_system_message(ReasoningLevel.HIGH)
                developer_msg = self.build_harmony_developer_message(test["instructions"])
            else:
                system_msg = self.build_generic_system_message()
                developer_msg = ""
            
            await self.run_test(
                endpoint=endpoint,
                test_name=test["name"],
                system_message=system_msg,
                developer_message=developer_msg,
                user_message=test["user_prompt"],
                expected_behavior=test["expected"],
                temperature=0.0
            )
    
    async def test_edge_cases(self, endpoint: EndpointConfig):
        """Test edge cases and error handling."""
        logger.info("=" * 60)
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] TESTING EDGE CASES")
        logger.info("=" * 60)
        
        edge_case_tests = [
            {
                "name": "empty_message",
                "user_prompt": ""
            },
            {
                "name": "very_long_prompt",
                "user_prompt": "Describe " + "x" * 1000
            },
            {
                "name": "special_characters",
                "user_prompt": "What is 2 + 2? @#$%^&*()_+{}|:<>?"
            },
            {
                "name": "unicode_characters",
                "user_prompt": "Translate: 你好世界 Привет мир 🌍"
            },
            {
                "name": "code_block",
                "user_prompt": "Write a Python function that calculates factorial"
            }
        ]
        
        for test in edge_case_tests:
            logger.info(f"\n--- [{endpoint.name or f'port-{endpoint.port}'}] Testing: {test['name']} ---")
            
            if endpoint.endpoint_type == EndpointType.HARMONY:
                system_msg = self.build_harmony_system_message(ReasoningLevel.MEDIUM)
                developer_msg = self.build_harmony_developer_message("You are a helpful assistant.")
            else:
                system_msg = self.build_generic_system_message()
                developer_msg = ""
            
            await self.run_test(
                endpoint=endpoint,
                test_name=test["name"],
                system_message=system_msg,
                developer_message=developer_msg,
                user_message=test["user_prompt"]
            )
    
    def get_response_content(self, response: dict) -> str:
        """Extract content from response based on endpoint type."""
        if "choices" in response:
            # OpenAI-compatible format
            return response["choices"][0]["message"].get("content", "")
        elif "output" in response:
            # Harmony format
            output = response["output"]
            if output and len(output) > 0:
                content_list = output[0].get("content", [])
                if content_list and len(content_list) > 0:
                    return content_list[0].get("text", "")
        return ""
    
    async def test_conversation_history(self, endpoint: EndpointConfig):
        """Test multi-turn conversations."""
        logger.info("=" * 60)
        logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] TESTING CONVERSATION HISTORY")
        logger.info("=" * 60)
        
        if endpoint.endpoint_type == EndpointType.HARMONY:
            system_msg = self.build_harmony_system_message(ReasoningLevel.HIGH)
            developer_msg = self.build_harmony_developer_message("You are a helpful assistant.")
        else:
            system_msg = self.build_generic_system_message()
            developer_msg = ""
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "developer", "content": developer_msg},
            {"role": "user", "content": "My favorite color is blue."},
        ]
        
        result1 = await self.send_request(endpoint, messages)
        
        if result1.get("status_code") == 200:
            response1 = self.get_response_content(result1["response"])
            logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] Turn 1 response: {response1[:100]}...")
            
            # Second turn - include conversation history
            messages.append({"role": "assistant", "content": response1})
            messages.append({"role": "user", "content": "What is my favorite color?"})
            
            result2 = await self.send_request(endpoint, messages)
            
            if result2.get("status_code") == 200:
                response2 = self.get_response_content(result2["response"])
                logger.info(f"[{endpoint.name or f'port-{endpoint.port}'}] Turn 2 response: {response2}")
                
                # Check if the model remembered
                test_result = TestResult(
                    endpoint_name=endpoint.name or f"port-{endpoint.port}",
                    test_name="conversation_history",
                    run_number=1,
                    passed="blue" in response2.lower()
                )
                self.results.append(test_result)
                self.endpoint_results[f"port-{endpoint.port}"].append(test_result)
    
    async def generate_report(self) -> str:
        """Generate a summary report comparing all endpoints."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-ENDPOINT TEST REPORT")
        report.append("=" * 80)
        
        # Per-endpoint summary
        report.append("\n" + "=" * 80)
        report.append("PER-ENDPOINT SUMMARY")
        report.append("=" * 80)
        
        for endpoint in ENDPOINTS:
            results = self.endpoint_results.get(f"port-{endpoint.port}", [])
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            display_name = self.get_display_name(endpoint)
            
            report.append(f"\n--- {display_name} ({endpoint.endpoint_type.value}) ---")
            report.append(f"Total tests: {total}")
            report.append(f"Passed: {passed}")
            report.append(f"Failed: {total - passed}")
            if total > 0:
                report.append(f"Success rate: {100*passed/total:.1f}%")
            
            # Calculate average response time
            if results:
                avg_time = sum(r.total_time for r in results) / total
                report.append(f"Average response time: {avg_time:.2f}s")
        
        # Consistency check (comparing run 1 vs run 2)
        if self.num_runs >= 2:
            report.append("\n" + "=" * 80)
            report.append("CONSISTENCY CHECK (Run 1 vs Run 2)")
            report.append("=" * 80)
            
            for endpoint in ENDPOINTS:
                results = self.endpoint_results.get(f"port-{endpoint.port}", [])
                run1_results = {r.test_name: r for r in results if r.run_number == 1}
                run2_results = {r.test_name: r for r in results if r.run_number == 2}
                
                consistent = 0
                inconsistent = 0
                
                for test_name in run1_results:
                    if test_name in run2_results:
                        r1 = run1_results[test_name]
                        r2 = run2_results[test_name]
                        if r1.passed == r2.passed:
                            consistent += 1
                        else:
                            inconsistent += 1
                
                display_name = self.get_display_name(endpoint)
                report.append(f"\n{display_name}:")
                report.append(f"  Consistent: {consistent}")
                report.append(f"  Inconsistent: {inconsistent}")
        
        # Detailed results
        report.append("\n" + "=" * 80)
        report.append("DETAILED RESULTS BY ENDPOINT")
        report.append("=" * 80)
        
        for endpoint in ENDPOINTS:
            results = self.endpoint_results.get(f"port-{endpoint.port}", [])
            display_name = self.get_display_name(endpoint)
            report.append(f"\n{'=' * 60}")
            report.append(f"ENDPOINT: {display_name}")
            report.append(f"{'=' * 60}")
            
            for result in results:
                report.append(f"\n--- {result.test_name} (run {result.run_number}) ---")
                report.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
                if result.reasoning_level:
                    report.append(f"Reasoning: {result.reasoning_level.value}")
                if result.temperature is not None:
                    report.append(f"Temperature: {result.temperature}")
                report.append(f"Response time: {result.total_time:.2f}s")
                if result.instruction_followed is not None:
                    report.append(f"Instruction followed: {result.instruction_followed}")
                if result.error:
                    report.append(f"Error: {result.error}")
                if result.response_content:
                    content_preview = result.response_content[:200].replace('\n', ' ')
                    report.append(f"Response preview: {content_preview}...")
        
        return "\n".join(report)
    
    async def save_results(self):
        """Save all test results to JSON files."""
        # Save combined results
        results_data = []
        for result in self.results:
            results_data.append({
                "endpoint_name": result.endpoint_name,
                "test_name": result.test_name,
                "run_number": result.run_number,
                "reasoning_level": result.reasoning_level.value if result.reasoning_level else None,
                "temperature": result.temperature,
                "status_code": result.status_code,
                "total_time": result.total_time,
                "passed": result.passed,
                "instruction_followed": result.instruction_followed,
                "error": result.error,
                "response_content": result.response_content,
                "request_body": result.request_body,
                "response_body": result.response_body
            })
        
        with open("test_results_multi.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Save per-endpoint results
        for endpoint in ENDPOINTS:
            endpoint_data = []
            for result in self.endpoint_results.get(f"port-{endpoint.port}", []):
                endpoint_data.append({
                    "test_name": result.test_name,
                    "run_number": result.run_number,
                    "passed": result.passed,
                    "total_time": result.total_time,
                    "response_content": result.response_content
                })
            
            filename = f"test_results_port-{endpoint.port}.json"
            with open(filename, "w") as f:
                json.dump(endpoint_data, f, indent=2)
        
        logger.info("Results saved to test_results_multi.json and per-endpoint files")


async def main():
    """Run all tests across all endpoints."""
    parser = argparse.ArgumentParser(description="Multi-endpoint API tester")
    parser.add_argument("--runs", type=int, default=2, help="Number of times to run each test (default: 2)")
    args = parser.parse_args()
    
    tester = MultiEndpointTester(num_runs=args.runs)
    
    # Fetch model names from all endpoints
    await tester.fetch_all_model_names()
    
    logger.info("=" * 80)
    logger.info("STARTING MULTI-ENDPOINT COMPREHENSIVE API TESTS")
    logger.info("=" * 80)
    logger.info(f"Number of runs per test: {args.runs}")
    logger.info("")
    
    # Run tests for each endpoint
    for endpoint in ENDPOINTS:
        display_name = tester.get_display_name(endpoint)
        logger.info(f"\n{'#' * 80}")
        logger.info(f"# TESTING ENDPOINT: {display_name} ({endpoint.endpoint_type.value})")
        logger.info(f"# Supports reasoning: {endpoint.supports_reasoning}")
        logger.info(f"{'#' * 80}")
        
        for run_num in range(1, args.runs + 1):
            if args.runs > 1:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"# RUN {run_num}/{args.runs}: {display_name}")
                logger.info(f"{'=' * 60}")
            
            # Run all test suites
            await tester.test_reasoning_levels(endpoint)
            await asyncio.sleep(1)
            
            await tester.test_temperatures(endpoint)
            await asyncio.sleep(1)
            
            await tester.test_instruction_following(endpoint)
            await asyncio.sleep(1)
            
            await tester.test_edge_cases(endpoint)
            await asyncio.sleep(1)
            
            await tester.test_conversation_history(endpoint)
            await asyncio.sleep(2)
    
    # Generate and save report
    report = await tester.generate_report()
    print(report)
    
    await tester.save_results()
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS COMPLETED!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
