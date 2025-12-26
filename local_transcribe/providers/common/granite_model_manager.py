# Granite Model Manager
# Base implementation for shared Granite model management functionality

import pathlib
import os
import sys
from typing import Optional, Any, Dict, List, Callable
import numpy as np
from numpy.typing import NDArray

# Import system capability utilities
from local_transcribe.lib.system_capability_utils import get_system_capability
from local_transcribe.lib.program_logger import log_progress, log_debug, log_completion


class GraniteModelManager:
    """Base class for managing Granite models across different providers.
    
    Supports both local model execution and remote server execution.
    Remote execution can be enabled by setting use_remote=True and providing
    a remote_url pointing to a Remote Granite Server.
    """
    
    MODEL_MAPPING = {
        "granite-2b": "ibm-granite/granite-speech-3.3-2b",
        "granite-8b": "ibm-granite/granite-speech-3.3-8b"
    }
    
    # Default remote Granite server URL
    DEFAULT_REMOTE_URL = "http://0.0.0.0:7070"
    
    def __init__(
        self,
        logger: Any,
        models_dir: Optional[pathlib.Path] = None,
        use_remote: bool = False,
        remote_url: Optional[str] = None
    ):
        """
        Initialize the model manager.
        
        Args:
            logger: Logger instance for logging messages
            models_dir: Directory for storing/loading models
            use_remote: If True, use remote Granite server for transcription
            remote_url: URL of remote Granite server (default: http://0.0.0.0:7070)
        """
        self.logger = logger
        self.models_dir = models_dir
        self.selected_model: Optional[str] = None
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        
        # Remote transcription configuration
        self.use_remote = use_remote
        self.remote_url = remote_url or self.DEFAULT_REMOTE_URL
        self._remote_client: Optional[Any] = None
        
        # Track whether remote is available (checked lazily)
        self._remote_available: Optional[bool] = None
    
    # =========================================================================
    # Remote Transcription Methods
    # =========================================================================
    
    def configure_remote(self, use_remote: bool, remote_url: Optional[str] = None) -> None:
        """
        Configure remote transcription settings.
        
        Args:
            use_remote: Whether to use remote server for transcription
            remote_url: URL of the remote Granite server
        """
        self.use_remote = use_remote
        if remote_url:
            self.remote_url = remote_url
        
        # Reset cached availability check
        self._remote_available = None
        self._remote_client = None
        
        if use_remote:
            self.logger.info(f"Remote Granite configured: {self.remote_url}")
        else:
            self.logger.info("Remote Granite disabled, using local model")
    
    def _get_remote_client(self) -> Any:
        """
        Get or create the remote Granite client.
        
        Returns:
            RemoteGraniteClient instance
        """
        if self._remote_client is None:
            from local_transcribe.providers.common.remote_granite_client import RemoteGraniteClient
            self._remote_client = RemoteGraniteClient(self.remote_url)
        return self._remote_client
    
    def is_remote_available(self, refresh: bool = False) -> bool:
        """
        Check if remote Granite server is available.
        
        Args:
            refresh: Force a new health check
        
        Returns:
            True if remote server is available and model is loaded
        """
        if not self.use_remote:
            return False
        
        if refresh or self._remote_available is None:
            try:
                client = self._get_remote_client()
                self._remote_available = client.is_available(refresh=True)
                if self._remote_available:
                    log_progress(f"Remote Granite server available at {self.remote_url}")
                else:
                    self.logger.warning(f"Remote Granite server not available at {self.remote_url}")
            except Exception as e:
                self.logger.warning(f"Failed to check remote Granite availability: {e}")
                self._remote_available = False
        
        return self._remote_available
    
    def transcribe_remote(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000,
        segment_duration: Optional[float] = None,
        include_disfluencies: bool = True,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Transcribe audio using remote Granite server.
        
        Args:
            audio: Audio samples as numpy array (mono, float32)
            sample_rate: Sample rate (must be 16000)
            segment_duration: Duration of segment in seconds (auto-calculated if None)
            include_disfluencies: Include disfluencies in output
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Transcribed text
        
        Raises:
            RuntimeError: If remote server is not available or transcription fails
        """
        if not self.use_remote:
            raise RuntimeError("Remote transcription not enabled. Call configure_remote(True) first.")
        
        if not self.is_remote_available():
            raise RuntimeError(f"Remote Granite server not available at {self.remote_url}")
        
        client = self._get_remote_client()
        
        # Calculate segment duration if not provided
        if segment_duration is None:
            segment_duration = len(audio) / sample_rate
        
        log_debug(f"Sending {segment_duration:.1f}s audio to remote Granite server")
        
        try:
            text = client.transcribe_audio(
                audio=audio,
                sample_rate=sample_rate,
                include_disfluencies=include_disfluencies,
                max_new_tokens=max_new_tokens
            )
            return text
        except Exception as e:
            self.logger.error(f"Remote transcription failed: {e}")
            raise RuntimeError(f"Remote transcription failed: {e}")
    
    def should_use_remote(self) -> bool:
        """
        Determine if remote transcription should be used.
        
        Returns:
            True if remote is configured and available
        """
        return self.use_remote and self.is_remote_available()
    
    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """
        Get the list of required model names based on the selected model.
        """
        # Determine which model to use
        model_to_check = selected_model or self.selected_model
        
        # Handle special case: return all available models if "all" is requested
        if model_to_check == "all":
            return list(self.MODEL_MAPPING.values())
        
        # If no model is specified, default to granite-8b
        if not model_to_check:
            self.logger.info("No model selected, defaulting to granite-8b")
            return [self.MODEL_MAPPING["granite-8b"]]
        
        # Validate the selected model
        if model_to_check not in self.MODEL_MAPPING:
            self.logger.error(f"Invalid model selected: {model_to_check}")
            self.logger.info(f"Available models: {list(self.MODEL_MAPPING.keys())}")
            raise ValueError(f"Invalid model '{model_to_check}'. Available models: {list(self.MODEL_MAPPING.keys())}")
        
        # Return the selected model
        return [self.MODEL_MAPPING[model_to_check]]
    
    def validate_model_selection(self, selected_model: Optional[str] = None) -> bool:
        """
        Validate that the selected model exists in MODEL_MAPPING.
        """
        # Determine which model to validate
        model_to_validate = selected_model or self.selected_model
        
        # Handle special case: "all" is always valid
        if model_to_validate == "all":
            self.logger.info("Model validation: 'all' is a valid selection")
            return True
        
        # If no model is specified, default validation passes (will use default in get_required_models)
        if not model_to_validate:
            self.logger.info("Model validation: No model specified, will use default")
            return True
        
        # Check if model exists in mapping
        if model_to_validate in self.MODEL_MAPPING:
            self.logger.info(f"Model validation: '{model_to_validate}' is valid")
            return True
        else:
            self.logger.warning(f"Model validation: '{model_to_validate}' is not valid")
            self.logger.info(f"Available models: {list(self.MODEL_MAPPING.keys())}")
            return False
    
    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """
        Check which models are available offline without downloading.
        """
        missing_models = []
        
        # Get the resolved cache directory
        cache_dir = self._resolve_cache_directory()
        
        for model in models:
            # Check if it's a full model name or a short name
            if model in self.MODEL_MAPPING.values():
                # It's a full model name
                hf_model_name = model.replace("/", "--")
                model_dir = cache_dir / f"models--{hf_model_name}"
            elif model in self.MODEL_MAPPING:
                # It's a short name, convert to full name first
                full_model_name = self.MODEL_MAPPING[model]
                hf_model_name = full_model_name.replace("/", "--")
                model_dir = cache_dir / f"models--{hf_model_name}"
            else:
                # Unknown model
                self.logger.warning(f"Unknown model: {model}, skipping availability check")
                continue
            
            # Check for model files (both .bin and .safetensors formats)
            has_model_files = (
                model_dir.exists() and (
                    any(model_dir.rglob("*.bin")) or
                    any(model_dir.rglob("*.safetensors"))
                )
            )
            
            if not has_model_files:
                missing_models.append(model)
        
        return missing_models
    
    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """
        Ensure models are available by preloading them if needed.
        """
        self.logger.info(f"Ensuring models are available: {models}")
        
        # Check which models are missing
        missing_models = self.check_models_available_offline(models, models_dir)
        
        if not missing_models:
            self.logger.info("All models are already available offline")
            return
        
        self.logger.info(f"Missing models that need to be downloaded: {missing_models}")
        
        # Preload the missing models
        self.preload_models(missing_models, models_dir)
        
        self.logger.info("All models are now available offline")
    
    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """
        Preload Granite models to cache.
        """
        self.logger.info("Starting model preload for Granite models")
        
        # Create cache directory structure using the resolved cache directory
        cache_dir = self._resolve_cache_directory()
        provider_cache_dir = models_dir / "transcribers" / "granite"
        provider_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original environment variables
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
        
        # Ensure offline mode is disabled for downloading
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        try:
            # Reload huggingface_hub modules to ensure fresh configuration
            self._reload_huggingface_modules()
            
            # Import huggingface_hub after module reload
            from huggingface_hub import snapshot_download
            
            # Validate models using existing method
            required_models = self.get_required_models()
            valid_models = []
            
            for model in models:
                if model in self.MODEL_MAPPING.values():
                    valid_models.append(model)
                elif model in self.MODEL_MAPPING:
                    # Convert short name to full name
                    valid_models.append(self.MODEL_MAPPING[model])
                else:
                    self.logger.warning(f"Unknown model: {model}, skipping")
            
            if not valid_models:
                self.logger.error("No valid models to preload")
                return
            
            # Download each model
            for i, model in enumerate(valid_models):
                self.logger.info(f"Downloading model {i+1}/{len(valid_models)}: {model}")
                
                # Set HF_HOME to our cache directory
                os.environ["HF_HOME"] = str(cache_dir)
                
                try:
                    # Get authentication token
                    token = os.getenv("HF_TOKEN")
                    
                    # Download the model
                    snapshot_download(model, token=token)
                    self.logger.info(f"Successfully downloaded {model}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to download {model}: {e}")
                    raise Exception(f"Failed to download {model}: {e}")
                    
        finally:
            # Restore original environment variables
            os.environ["HF_HUB_OFFLINE"] = offline_mode
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)
    
    def _reload_huggingface_modules(self) -> None:
        """
        Reload HuggingFace modules to ensure fresh configuration for model loading.
        """
        import importlib
        import sys
        
        self.logger.debug("Reloading HuggingFace modules for fresh configuration")
        
        try:
            # Remove huggingface_hub modules from cache
            modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    self.logger.debug(f"Removed cached module: {module_name}")
            
            # Remove transformers modules from cache
            modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    self.logger.debug(f"Removed cached module: {module_name}")
            
            self.logger.debug("HuggingFace modules reloaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Error while reloading HuggingFace modules: {e}")
            # Continue with execution even if module reload fails
    
    def _download_model(self, model_name: str, cache_dir: pathlib.Path) -> None:
        """
        Download a specific model from HuggingFace Hub to the cache directory.
        """
        self.logger.info(f"Starting download for model: {model_name}")
        
        # Use the resolved cache directory for standard HuggingFace models
        resolved_cache_dir = self._resolve_cache_directory()
        
        # Store original environment variables
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
        
        # Ensure offline mode is disabled for downloading
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        try:
            # Reload huggingface_hub modules to ensure fresh configuration
            self._reload_huggingface_modules()
            
            # Import huggingface_hub after module reload
            from huggingface_hub import snapshot_download
            
            # Set HF_HOME to our resolved cache directory
            os.environ["HF_HOME"] = str(resolved_cache_dir)
            
            # Get authentication token
            token = os.getenv("HF_TOKEN")
            
            if token:
                self.logger.debug("Using HF_TOKEN for authentication")
            else:
                self.logger.warning("No HF_TOKEN found - downloading without authentication")
            
            # Download the model with progress tracking
            self.logger.info(f"Downloading model to: {resolved_cache_dir}")
            snapshot_download(model_name, token=token)
            
            self.logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to download {model_name}: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
            
        finally:
            # Restore original environment variables
            os.environ["HF_HUB_OFFLINE"] = offline_mode
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)
    
    def _resolve_cache_directory(self) -> pathlib.Path:
        """
        Resolve the cache directory for storing models.
        """
        # Check for XDG_CACHE_HOME environment variable first
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            models_root = pathlib.Path(xdg_cache_home)
        else:
            # Fallback to standard HuggingFace cache location
            models_root = pathlib.Path.home() / ".cache" / "huggingface"
        
        # The models are stored in the standard HuggingFace hub structure
        cache_dir = models_root / "huggingface" / "hub"
        
        try:
            # Create directory structure if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Log the resolved cache directory for debugging
            self.logger.debug(f"Resolved cache directory: {cache_dir}")
            
            return cache_dir
            
        except OSError as e:
            error_msg = f"Failed to create cache directory {cache_dir}: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg)
    
    def _load_model(self, model_name: str) -> None:
        """
        Load a Granite model from the cache directory.
        """
        self.logger.info(f"Loading model: {model_name}")
        
        # Get the resolved cache directory
        cache_dir = self._resolve_cache_directory()
        
        try:
            # Import required modules
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            
            # Get authentication token
            token = os.getenv("HF_TOKEN")
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                local_files_only=True,
                token=token
            )
            # Extract tokenizer from processor with null check
            if self.processor is not None:
                self.tokenizer = getattr(self.processor, 'tokenizer', None)
            else:
                self.tokenizer = None
            
            # Load the model with proper device handling
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                local_files_only=True,
                token=token
            )
            
            # Move model to the appropriate device
            try:
                import torch
                device = get_system_capability()
                
                if self.model is not None:
                    if device == "mps" and torch.backends.mps.is_available():
                        # For MPS, we need to explicitly move the model and set device
                        self.model = self.model.to("mps")
                        self.logger.info(f"Model moved to MPS device: {device}")
                    elif device == "cuda" and torch.cuda.is_available():
                        # For CUDA
                        self.model = self.model.to("cuda")
                        self.logger.info(f"Model moved to CUDA device: {device}")
                    else:
                        # For CPU or other devices
                        self.model = self.model.to("cpu")
                        self.logger.info(f"Model moved to CPU device: {device}")
                        
            except Exception as device_error:
                self.logger.warning(f"Device placement failed ({device_error}), falling back to CPU")
                if self.model is not None:
                    self.model = self.model.to("cpu")
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"Cache directory exists: {cache_dir.exists()}")
            if cache_dir.exists():
                self.logger.debug(f"Cache directory contents: {list(cache_dir.iterdir())}")
            raise Exception(error_msg)
        