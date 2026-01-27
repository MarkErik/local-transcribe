#!/usr/bin/env python3
"""
Remote Transcriber Provider

A transcriber plugin that delegates transcription to a remote server.
This provider is model-agnostic - the remote server can run any transcription model.

The client automatically adapts to server capabilities by querying the /info endpoint
to learn about segment length limits and other constraints.
"""

import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
import math
import numpy as np
from numpy.typing import NDArray
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import librosa

from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.program_logger import get_logger, log_progress, log_debug, log_completion
from local_transcribe.processing.chunk_stitching import ChunkStitcher


# =============================================================================
# Exceptions
# =============================================================================

class RemoteTranscriberError(Exception):
    """Base exception for remote transcriber errors."""
    pass


class RemoteTranscriberConnectionError(RemoteTranscriberError):
    """Raised when connection to remote server fails."""
    pass


class RemoteTranscriberTranscriptionError(RemoteTranscriberError):
    """Raised when transcription fails on the remote server."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


# =============================================================================
# Server Capabilities
# =============================================================================

@dataclass
class ServerCapabilities:
    """Capabilities reported by the remote transcription server."""
    max_segment_duration_s: float  # Maximum audio segment length server accepts
    min_segment_duration_s: float  # Minimum audio segment length server accepts
    model_name: str                # Server's model (informational only)
    sample_rate: int               # Expected sample rate


# =============================================================================
# Remote Transcriber Client
# =============================================================================

class RemoteTranscriberClient:
    """
    HTTP client for remote transcription server.
    
    This client provides methods to:
    - Check server health and availability
    - Query server capabilities
    - Send audio for transcription
    - Handle errors and retries with exponential backoff
    
    Example usage:
        client = RemoteTranscriberClient("http://192.168.1.100:7070")
        
        if client.is_available():
            text = client.transcribe_audio(audio_array, sample_rate=16000)
            print(text)
    """
    
    DEFAULT_URL = "http://0.0.0.0:7070"
    DEFAULT_TIMEOUT = 300  # 5 minutes for long segments
    CONNECT_TIMEOUT = 10   # 10 seconds for connection
    
    def __init__(
        self,
        server_url: str = DEFAULT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5
    ):
        """
        Initialize the remote transcriber client.
        
        Args:
            server_url: URL of the remote transcription server
            timeout: Timeout in seconds for transcription requests
            max_retries: Maximum number of retries for failed requests
            retry_backoff_factor: Backoff factor for retry delays
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = get_logger()
        
        # Create session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Cache server info
        self._server_info: Optional[Dict[str, Any]] = None
        self._capabilities: Optional[ServerCapabilities] = None
        self._is_available: Optional[bool] = None
    
    def _encode_audio(self, audio: NDArray[np.floating]) -> str:
        """
        Encode numpy audio array to base64 string.
        
        Args:
            audio: 1D numpy array of audio samples (will be converted to float32)
        
        Returns:
            Base64-encoded string of the audio data
        """
        audio_float32 = audio.astype(np.float32)
        audio_bytes = audio_float32.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64
    
    def check_health(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if the remote server is healthy and ready.
        
        Returns:
            Tuple of (is_healthy, health_info_dict)
        """
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=self.CONNECT_TIMEOUT
            )
            
            if response.status_code == 200:
                health_info = response.json()
                is_healthy = (
                    health_info.get("status") == "healthy" and
                    health_info.get("model_loaded", False)
                )
                return is_healthy, health_info
            else:
                return False, {"error": f"HTTP {response.status_code}"}
                
        except requests.exceptions.ConnectionError as e:
            self.logger.debug(f"Remote server connection failed: {e}")
            return False, {"error": "Connection failed"}
        except requests.exceptions.Timeout:
            self.logger.debug("Remote server health check timed out")
            return False, {"error": "Timeout"}
        except Exception as e:
            self.logger.debug(f"Remote server health check error: {e}")
            return False, {"error": str(e)}
    
    def is_available(self, refresh: bool = False) -> bool:
        """
        Check if the remote server is available.
        
        Args:
            refresh: If True, force a new health check. Otherwise use cached result.
        
        Returns:
            True if server is available and model is loaded
        """
        if refresh or self._is_available is None:
            is_healthy, _ = self.check_health()
            self._is_available = is_healthy
        return self._is_available
    
    def get_server_info(self, refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed server information.
        
        Args:
            refresh: If True, force a new request. Otherwise use cached result.
        
        Returns:
            Server info dictionary or None if unavailable
        """
        if refresh or self._server_info is None:
            try:
                response = self.session.get(
                    f"{self.server_url}/info",
                    timeout=self.CONNECT_TIMEOUT
                )
                if response.status_code == 200:
                    self._server_info = response.json()
                else:
                    self._server_info = None
            except Exception as e:
                self.logger.debug(f"Failed to get server info: {e}")
                self._server_info = None
        
        return self._server_info
    
    def get_capabilities(self, refresh: bool = False) -> ServerCapabilities:
        """
        Get server capabilities, querying the /info endpoint if needed.
        
        Args:
            refresh: If True, force a refresh of capabilities
            
        Returns:
            ServerCapabilities dataclass with server limits
        """
        if refresh or self._capabilities is None:
            info = self.get_server_info(refresh=True)
            
            if info:
                # Parse nested structure from server
                # Server returns: {model: {name: ...}, capabilities: {max_audio_duration_seconds: ...}}
                model_info = info.get("model", {})
                capabilities_info = info.get("capabilities", {})
                
                # Extract model name from nested structure
                model_name = model_info.get("name", "unknown")
                
                # Extract capabilities from nested structure
                max_duration = capabilities_info.get("max_audio_duration_seconds", 60.0)
                min_duration = capabilities_info.get("min_audio_duration_seconds", 1.0)
                
                # Get sample rate - either from capabilities or supported_sample_rates list
                sample_rates = capabilities_info.get("supported_sample_rates", [16000])
                sample_rate = sample_rates[0] if sample_rates else 16000
                
                self._capabilities = ServerCapabilities(
                    max_segment_duration_s=float(max_duration),
                    min_segment_duration_s=float(min_duration),
                    model_name=model_name,
                    sample_rate=int(sample_rate)
                )
            else:
                # Default capabilities if server doesn't provide /info
                self._capabilities = ServerCapabilities(
                    max_segment_duration_s=60.0,
                    min_segment_duration_s=1.0,
                    model_name="unknown",
                    sample_rate=16000
                )
        
        return self._capabilities
    
    def transcribe_audio(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe audio using the remote server.
        
        Args:
            audio: 1D numpy array of audio samples (mono, preferably float32)
            sample_rate: Sample rate of the audio (should be 16000)        
        Returns:
            Transcribed text string
        
        Raises:
            RemoteTranscriberConnectionError: If connection to server fails
            RemoteTranscriberTranscriptionError: If transcription fails
        """
        segment_duration = len(audio) / sample_rate
        
        # Encode audio
        audio_base64 = self._encode_audio(audio)
        
        # Build request payload - only include options the server needs
        payload = {
            "audio_data": audio_base64,
            "sample_rate": sample_rate,
            "audio_format": "float32",
            "segment_duration": segment_duration,
        }
        
        try:
            log_debug(f"Sending {segment_duration:.1f}s audio to remote server")
            
            response = self.session.post(
                f"{self.server_url}/transcribe",
                json=payload,
                timeout=(self.CONNECT_TIMEOUT, self.timeout)
            )
            
            result = response.json()
            
            if response.status_code == 200 and result.get("success"):
                text = result.get("text", "")
                processing_time = result.get("processing_time_ms", 0)
                log_debug(f"Remote transcription completed in {processing_time}ms")
                return text
            else:
                error_msg = result.get("error", f"HTTP {response.status_code}")
                error_code = result.get("error_code")
                raise RemoteTranscriberTranscriptionError(error_msg, error_code)
                
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection to remote server failed: {e}")
            self._is_available = False
            raise RemoteTranscriberConnectionError(f"Connection failed: {e}")
        except requests.exceptions.Timeout:
            self.logger.error("Remote transcription request timed out")
            raise RemoteTranscriberTranscriptionError("Request timed out", "TIMEOUT")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Remote request failed: {e}")
            raise RemoteTranscriberTranscriptionError(str(e))
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __repr__(self) -> str:
        available = "available" if self._is_available else "unavailable" if self._is_available is False else "unknown"
        return f"RemoteTranscriberClient(url={self.server_url!r}, status={available})"


# =============================================================================
# Utility Functions
# =============================================================================

def check_remote_transcriber_available(server_url: str) -> bool:
    """
    Quick check if a remote transcription server is available at the given URL.
    
    Args:
        server_url: URL to check
    
    Returns:
        True if server is available and model is loaded
    """
    try:
        response = requests.get(
            f"{server_url.rstrip('/')}/health",
            timeout=5
        )
        if response.status_code == 200:
            health = response.json()
            return health.get("status") == "healthy" and health.get("model_loaded", False)
    except Exception:
        pass
    return False


def get_remote_server_info(server_url: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information from a remote transcription server.
    
    Args:
        server_url: URL of the remote transcription server
    
    Returns:
        Server info dictionary with model and capabilities, or None if unavailable
    """
    try:
        response = requests.get(
            f"{server_url.rstrip('/')}/info",
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# =============================================================================
# Remote Transcriber Provider
# =============================================================================

class RemoteTranscriberProvider(TranscriberProvider):
    """
    Transcriber provider that delegates to a remote transcription server.
    
    This provider is model-agnostic - the remote server can run any transcription
    model (Granite, Whisper, etc.). The client automatically adapts to server
    capabilities by querying the /info endpoint.
    
    Features:
    - Automatic chunking for long audio files based on server limits
    - Chunk stitching for continuous transcription of long recordings
    - No local model loading required
    
    Note: This provider does NOT support alignment (has_builtin_alignment = False).
    Use it with a separate aligner if word-level timestamps are needed.
    """

    DEFAULT_SERVER_URL = "http://0.0.0.0:7070"

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Remote Transcriber Provider")
        
        # Server configuration
        self.server_url: str = self.DEFAULT_SERVER_URL
        self._client: Optional[RemoteTranscriberClient] = None
        self._capabilities: Optional[ServerCapabilities] = None
        
        # Chunking configuration (will be updated from server capabilities)
        self.chunk_length_seconds: float = 60.0
        self.overlap_seconds: float = 4.0
        self.min_chunk_seconds: float = 7.0

    @property
    def name(self) -> str:
        return "remote"

    @property
    def short_name(self) -> str:
        return "Remote Server"

    @property
    def description(self) -> str:
        return "Remote transcription server (model-agnostic, no alignment)"

    @property
    def has_builtin_alignment(self) -> bool:
        """Remote transcription does not provide word-level alignment."""
        return False

    @property
    def max_audio_chunk_duration_s(self) -> float:
        """Maximum audio chunk duration from server capabilities."""
        if self._capabilities:
            return self._capabilities.max_segment_duration_s
        return 30.0  # Default before capabilities are fetched

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """No local models required for remote transcription."""
        return []

    def get_available_models(self) -> List[str]:
        """Return single 'default' model representing the remote server."""
        return ["default"]

    def preload_models(self, models: List[str], models_dir) -> None:
        """No-op for remote transcriber - no local models to preload."""
        pass

    def check_models_available_offline(self, models: List[str], models_dir) -> List[str]:
        """No models needed offline for remote transcription."""
        return []

    def _get_client(self) -> RemoteTranscriberClient:
        """Get or create the remote transcriber client."""
        if self._client is None or self._client.server_url != self.server_url:
            self._client = RemoteTranscriberClient(self.server_url)
        return self._client

    def _update_chunking_from_capabilities(self) -> None:
        """Update chunking parameters based on server capabilities."""
        client = self._get_client()
        capabilities = client.get_capabilities()
        self._capabilities = capabilities
        
        # Set chunk length to server's max, with some margin
        self.chunk_length_seconds = min(capabilities.max_segment_duration_s, 60.0)
        self.min_chunk_seconds = max(capabilities.min_segment_duration_s, 1.0)
        
        log_debug(f"Server capabilities: max={capabilities.max_segment_duration_s}s, "
                  f"min={capabilities.min_segment_duration_s}s, model={capabilities.model_name}")

    def configure(self, server_url: Optional[str] = None) -> None:
        """
        Configure the remote transcriber.
        
        Args:
            server_url: URL of the remote transcription server
        """
        if server_url:
            self.server_url = server_url
            self._client = None  # Reset client to use new URL
            self._capabilities = None
            self.logger.info(f"Remote transcriber configured: {self.server_url}")

    def is_available(self, refresh: bool = False) -> bool:
        """
        Check if the remote server is available.
        
        Args:
            refresh: Force a new health check
            
        Returns:
            True if server is available
        """
        try:
            client = self._get_client()
            return client.is_available(refresh=refresh)
        except Exception as e:
            self.logger.warning(f"Failed to check remote server availability: {e}")
            return False

    def transcribe(
        self,
        audio_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio using the remote server.
        
        Args:
            audio_path: Path to the audio file
            device: Ignored for remote transcription
            **kwargs: Additional options:
                - server_url: Override the server URL        
        Returns:
            List of dictionaries with chunk data (chunk_id, words, text)
        """
        # Configure server URL if provided
        if 'server_url' in kwargs:
            self.configure(server_url=kwargs['server_url'])
        
        # Check server availability
        client = self._get_client()
        if not client.is_available(refresh=True):
            raise RemoteTranscriberConnectionError(
                f"Remote server not available at {self.server_url}"
            )
        
        # Update chunking parameters from server capabilities
        self._update_chunking_from_capabilities()
        
        log_progress(f"Remote transcription using server at {self.server_url}")
        
        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(wav) / sr
        
        # For short audio, transcribe directly
        if duration <= self.chunk_length_seconds:
            if duration < self.min_chunk_seconds:
                # Pad very short audio with silence
                min_samples = int(self.min_chunk_seconds * sr)
                if len(wav) < min_samples:
                    wav = np.pad(wav, (0, min_samples - len(wav)), mode='constant')
                    log_debug(f"Padded short audio from {duration:.1f}s to {self.min_chunk_seconds}s")
            
            log_progress(f"Audio duration: {duration:.1f}s - transcribing as single segment")
            text = client.transcribe_audio(wav, sr)
            return [{"chunk_id": 0, "words": text.split(), "text": text}]
        
        # For long audio, process in chunks
        log_progress(f"Audio duration: {duration:.1f}s - processing in chunks")
        return self._transcribe_chunked(wav, sr)

    def _transcribe_chunked(
        self,
        wav: NDArray,
        sr: int,
    ) -> List[Dict[str, Any]]:
        """
        Transcribe long audio in chunks with overlap and stitching.
        
        Args:
            wav: Audio samples
            sr: Sample rate            
        Returns:
            List of chunk dictionaries
        """
        client = self._get_client()
        
        chunk_samples = int(self.chunk_length_seconds * sr)
        overlap_samples = int(self.overlap_seconds * sr)
        min_chunk_samples = int(self.min_chunk_seconds * sr)
        
        chunks = []
        total_samples = len(wav)
        effective_chunk_length = chunk_samples - overlap_samples
        total_chunks = math.ceil(total_samples / effective_chunk_length) if effective_chunk_length > 0 else 1
        
        chunk_start = 0
        chunk_num = 0
        prev_chunk_wav = None
        
        while chunk_start < total_samples:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            chunk_wav = wav[chunk_start:chunk_end]
            
            chunk_duration_sec = len(chunk_wav) / sr
            log_progress(f"Processing chunk {chunk_num} of {total_chunks} ({chunk_duration_sec:.1f}s)...")
            
            # Handle short final chunks
            if len(chunk_wav) < min_chunk_samples:
                if prev_chunk_wav is not None and len(chunks) > 0:
                    # Merge with previous chunk
                    non_overlapping_part = chunk_wav[overlap_samples:] if len(chunk_wav) > overlap_samples else chunk_wav
                    merged_wav = np.concatenate([prev_chunk_wav, non_overlapping_part])
                    
                    # Re-transcribe merged chunk
                    chunk_text = client.transcribe_audio(merged_wav, sr)
                    existing_id = chunks[-1]["chunk_id"]
                    chunks[-1] = {"chunk_id": existing_id, "words": chunk_text.split(), "text": chunk_text}
                else:
                    # Pad short chunk
                    padded_wav = np.pad(chunk_wav, (0, min_chunk_samples - len(chunk_wav)), mode='constant')
                    chunk_text = client.transcribe_audio(padded_wav, sr)
                    chunks.append({"chunk_id": chunk_num, "words": chunk_text.split(), "text": chunk_text})
            else:
                # Normal chunk processing
                chunk_text = client.transcribe_audio(chunk_wav, sr)
                chunks.append({"chunk_id": chunk_num, "words": chunk_text.split(), "text": chunk_text})
            
            prev_chunk_wav = chunk_wav
            
            if chunk_end >= total_samples:
                break
            
            chunk_start = chunk_start + chunk_samples - overlap_samples
        
        return chunks

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Not supported - remote transcription does not provide alignment.
        
        Use transcribe() and pair with a separate aligner provider.
        """
        raise NotImplementedError(
            "Remote transcriber does not support alignment. "
            "Use transcribe() and pair with a separate aligner provider."
        )


def register_transcriber_plugins():
    """Register the remote transcriber plugin."""
    registry.register_transcriber_provider(RemoteTranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
