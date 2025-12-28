#!/usr/bin/env python3
"""
Remote Granite Client

HTTP client for communicating with a remote Granite 8B transcription server.
This module provides a clean interface for sending audio data to a remote server
and receiving transcription results.
"""

import base64
from typing import Optional, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from local_transcribe.lib.program_logger import get_logger, log_progress, log_debug, log_completion


class RemoteGraniteError(Exception):
    """Base exception for remote Granite client errors."""
    pass


class RemoteGraniteConnectionError(RemoteGraniteError):
    """Raised when connection to remote server fails."""
    pass


class RemoteGraniteTranscriptionError(RemoteGraniteError):
    """Raised when transcription fails on the remote server."""
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code


class RemoteGraniteClient:
    """
    HTTP client for remote Granite 8B transcription server.
    
    This client provides methods to:
    - Check server health and availability
    - Send audio chunks for transcription
    - Handle errors and retries gracefully
    
    Example usage:
        client = RemoteGraniteClient("http://192.168.1.100:7070")
        
        if client.is_available():
            text = client.transcribe_audio(audio_array, sample_rate=16000)
            print(text)
    """
    
    DEFAULT_URL = "http://0.0.0.0:7070"
    DEFAULT_TIMEOUT = 300  # 2 minutes for long segments
    CONNECT_TIMEOUT = 10    # 5 seconds for connection
    
    def __init__(
        self,
        server_url: str = DEFAULT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5
    ):
        """
        Initialize the remote Granite client.
        
        Args:
            server_url: URL of the remote Granite server (e.g., "http://192.168.1.100:7070")
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
            self.logger.debug(f"Remote Granite server connection failed: {e}")
            return False, {"error": "Connection failed"}
        except requests.exceptions.Timeout:
            self.logger.debug("Remote Granite server health check timed out")
            return False, {"error": "Timeout"}
        except Exception as e:
            self.logger.debug(f"Remote Granite health check error: {e}")
            return False, {"error": str(e)}
    
    def is_available(self, refresh: bool = False) -> bool:
        """
        Check if the remote Granite server is available.
        
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
    
    def transcribe_audio(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000,
        include_disfluencies: bool = True,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 4,
        repetition_penalty: float = 3.0
    ) -> str:
        """
        Transcribe audio using the remote Granite server.
        
        Args:
            audio: 1D numpy array of audio samples (mono, preferably float32)
            sample_rate: Sample rate of the audio (must be 16000)
            include_disfluencies: Whether to include disfluencies (um, uh) in output
            max_new_tokens: Maximum tokens to generate (auto-calculated if None)
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty factor
        
        Returns:
            Transcribed text string
        
        Raises:
            RemoteGraniteConnectionError: If connection to server fails
            RemoteGraniteTranscriptionError: If transcription fails
        """
        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000, got {sample_rate}")
        
        segment_duration = len(audio) / sample_rate
        
        # Auto-calculate max_new_tokens based on duration if not specified
        if max_new_tokens is None:
            if segment_duration < 8.0:
                max_new_tokens = 128
            elif segment_duration < 20.0:
                max_new_tokens = 256
            else:
                max_new_tokens = 512
        
        # Encode audio
        audio_base64 = self._encode_audio(audio)
        
        # Build request payload
        payload = {
            "audio_data": audio_base64,
            "sample_rate": sample_rate,
            "audio_format": "float32",
            "segment_duration": segment_duration,
            "options": {
                "include_disfluencies": include_disfluencies,
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty
            }
        }
        
        try:
            log_debug(f"Sending {segment_duration:.1f}s audio to remote Granite server")
            
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
                raise RemoteGraniteTranscriptionError(error_msg, error_code)
                
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection to remote Granite server failed: {e}")
            self._is_available = False
            raise RemoteGraniteConnectionError(f"Connection failed: {e}")
        except requests.exceptions.Timeout:
            self.logger.error("Remote Granite transcription request timed out")
            raise RemoteGraniteTranscriptionError("Request timed out", "TIMEOUT")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Remote Granite request failed: {e}")
            raise RemoteGraniteTranscriptionError(str(e))
    
    def transcribe_audio_with_fallback(
        self,
        audio: NDArray[np.floating],
        local_transcribe_func,
        sample_rate: int = 16000,
        **kwargs
    ) -> Tuple[str, bool]:
        """
        Attempt remote transcription with automatic fallback to local.
        
        Args:
            audio: Audio samples to transcribe
            local_transcribe_func: Callable that performs local transcription
            sample_rate: Sample rate of audio
            **kwargs: Additional arguments for both remote and local transcription
        
        Returns:
            Tuple of (transcribed_text, used_remote)
        """
        if self.is_available():
            try:
                text = self.transcribe_audio(audio, sample_rate, **kwargs)
                return text, True
            except RemoteGraniteError as e:
                self.logger.warning(f"Remote transcription failed, falling back to local: {e}")
        
        # Fallback to local
        text = local_transcribe_func(audio, **kwargs)
        return text, False
    
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
        return f"RemoteGraniteClient(url={self.server_url!r}, status={available})"


# Singleton instance for shared use across the application
_default_client: Optional[RemoteGraniteClient] = None


def get_remote_granite_client(
    server_url: Optional[str] = None,
    create_if_missing: bool = True
) -> Optional[RemoteGraniteClient]:
    """
    Get the default RemoteGraniteClient instance.
    
    Args:
        server_url: Server URL. If provided and different from current, creates new client.
        create_if_missing: If True, create client if it doesn't exist.
    
    Returns:
        RemoteGraniteClient instance or None
    """
    global _default_client
    
    if server_url is not None:
        # Check if we need to create a new client with different URL
        if _default_client is None or _default_client.server_url != server_url.rstrip('/'):
            if create_if_missing:
                _default_client = RemoteGraniteClient(server_url)
            else:
                return None
    elif _default_client is None and create_if_missing:
        _default_client = RemoteGraniteClient()
    
    return _default_client


def set_remote_granite_client(client: Optional[RemoteGraniteClient]) -> None:
    """
    Set the default RemoteGraniteClient instance.
    
    Args:
        client: Client instance to use, or None to clear.
    """
    global _default_client
    _default_client = client


def check_remote_granite_available(server_url: str) -> bool:
    """
    Quick check if a remote Granite server is available at the given URL.
    
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
