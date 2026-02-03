# Plan: End-to-End De-Identification Pipeline Test with Multiple LLM Endpoints

## Objective
Create a comprehensive test script that evaluates the full transcription and de-identification pipeline using real audio files (P28 samples) and multiple LLM endpoints. The test will ensure proper request formatting and response parsing for Harmony vs. OpenAI-compatible endpoints, addressing current limitations in the `local_transcribe` program's generic endpoint handling.

## Background
- **Current Issue**: The `local_transcribe` program treats all LLM endpoints generically, leading to suboptimal request formatting. Harmony endpoints require structured messages with system/developer/user roles and reasoning channels, while OpenAI-compatible endpoints use standard chat completions format. The current program de-identification is performing poorly, thus we are undertaking this test to determine which model performs the best for our needs.
- **Pipeline Overview**:
  1. Audio transcription via VAD-split-audio pipeline using remote transcriber (http://100.84.208.72:7070)
  2. First-pass de-identification using LLM (http://100.84.208.72:8080)
  3. Second-pass de-identification using LLM
- **Test Files**: `samples/audioMA-P28_cropped_30.0min.m4a` (interviewer) and `samples/audioP28_cropped_30.0min.m4a` (participant)
- **Endpoints to Test**:
  - Port 8080: Harmony (supports reasoning - test low, medium, high)
  - Port 8105: Harmony (supports reasoning - test low, medium, high)
  - Port 8107: OpenAI Compatible (no reasoning, but test temperature 0.0, 0.2, 0.5, 0.7, 1.0)
  - Port 8083: OpenAI Compatible (no reasoning, but test temperature 0.0, 0.2, 0.5, 0.7, 1.0)

## Test Structure
The test will be a Python script that:
1. Runs transcription on both audio files
2. Processes the combined transcript through de-identification (first and second pass) for each endpoint
3. Validates formatting, parsing, and de-identification quality
4. Generates a comparative report

## Implementation Steps

### 1. Setup and Dependencies
- Create new test file: `test_deid_pipeline_multi.py` in `.debug_utils/` directory
- Import required modules: `httpx`, `asyncio`, `json`, `logging`, `dataclasses`, etc.
- Import existing pipeline components:
  - `DeIdentificationOrchestrator` from `local_transcribe.processing.de_identification`
  - Pipeline runner components for transcription
- Define endpoint configurations matching the provided `EndpointConfig` class
- Set up logging similar to `test_endpoint_multi.py`

### 2. Audio Transcription Phase
- **Use existing vad-split-audio pipeline**: Instead of implementing new transcription functions, leverage the existing pipeline infrastructure
- Create a minimal pipeline context for vad-split-audio mode
- Handle VAD splitting and transcription using existing `build_turns_vad_split_audio` function
- Combine interviewer and participant transcripts using existing logic

### 3. De-Identification Phase
- **Use existing DeIdentificationOrchestrator**: Instead of implementing new de-identification functions, instantiate the existing orchestrator with different endpoint URLs
- For each endpoint configuration:
  - Create a `DeIdentificationOrchestrator` instance with the endpoint's URL
  - The orchestrator's `LLMDeIdentifierClient` will auto-detect Harmony vs OpenAI-compatible format
  - Run `de_identify_multi_speaker()` on the combined transcript data
  - This automatically handles both first-pass and second-pass de-identification
- **Endpoint-specific adjustments**: The existing LLM client already handles different request/response formats:
  - **Harmony endpoints**: Auto-detected and parsed using special tokens (`<|channel|>`, `<|message|>`, etc.)
  - **OpenAI-compatible endpoints**: Standard `/chat/completions` API with JSON response parsing

### 4. Request/Response Handling
**Rebuild this in the test script as it does not work well in the program, use the approach from test_endpoint_multi.py**

### 5. Validation and Metrics
- **Use existing response checks**: The program already checks that the response matches the number of words etc.
- **Formatting Validation**: Verify that the LLM client's auto-detection works correctly for each endpoint
- **Response Parsing**: Confirm that responses are properly extracted using existing parsing logic
- **Performance Metrics**: Measure response times, token usage, success rates from existing client
- **Consistency Check**: Run multiple iterations to check for variability using existing retry logic
- **Formatting Validation**: Ensure requests are correctly structured for each endpoint type

### 6. Report Generation
- Create JSON output with detailed results per endpoint
- Include success rates, response times, and de-identification quality scores
- Generate markdown summary comparing endpoints
- Highlight formatting issues and recommendations

## Expected Outcomes
- Validate that the existing `LLMDeIdentifierClient` auto-detection works correctly across all endpoint types
- Provide data to determine if endpoint-specific optimizations are needed in the main program
- Establish benchmarks for de-identification quality and speed using the existing pipeline
