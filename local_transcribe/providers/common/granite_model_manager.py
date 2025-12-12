# Granite Model Manager
# Base implementation for shared Granite model management functionality

import pathlib
import os
import sys
from typing import Optional, Any, Dict, List


class GraniteModelManager:
    """Base class for managing Granite models across different providers."""
    
    MODEL_MAPPING = {
        "granite-2b": "ibm-granite/granite-speech-3.3-2b",
        "granite-8b": "ibm-granite/granite-speech-3.3-8b"
    }
    
    def __init__(self, logger: Any, models_dir: Optional[pathlib.Path] = None):
        """Initialize the model manager.
        
        Args:
            logger: Logger instance for logging operations
            models_dir: Optional base directory for model storage
        """
        self.logger = logger
        self.models_dir = models_dir
        self.selected_model: Optional[str] = None
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
    
    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Get the list of required model names based on the selected model.
        
        This method extracts the logic from all existing plugins to determine
        which models need to be loaded based on the selected model configuration.
        
        Args:
            selected_model: Optional model name to use. If None, uses the instance's
                          selected_model or defaults to "granite-8b" if no model is selected.
                          
        Returns:
            List of required model names. Returns empty list if no valid model is selected,
            or returns all available models if selected_model is "all".
            
        Raises:
            ValueError: If the selected_model is not valid and cannot be resolved.
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
        """Validate that the selected model exists in MODEL_MAPPING.
        
        Args:
            selected_model: Model name to validate. If None, uses the instance's
                          selected_model.
                          
        Returns:
            True if the model is valid, False otherwise.
            
        Logs:
            Information about validation results including warnings for invalid models
            and successful validation messages.
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
        """Check which models are available offline without downloading.
        
        This method checks if the specified models are already cached locally
        and available for offline use. It uses the standard HuggingFace cache
        directory structure to locate model files.
        
        Args:
            models: List of model names to check (e.g., ["ibm-granite/granite-speech-3.3-8b"])
            models_dir: Base directory where models should be cached
            
        Returns:
            List of model names that are missing or not available offline
            
        Note:
            This method checks for both .bin and .safetensors model files,
            as different HuggingFace models may use different file formats.
            It also handles both full model names and short names from MODEL_MAPPING.
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
        """Ensure models are available by preloading them if needed.
        
        This method checks if the specified models are available offline and
        downloads them if they're missing. This ensures that models are ready
        for use when needed, without requiring network access during transcription.
        
        Args:
            models: List of model names to ensure are available
            models_dir: Base directory where models should be cached
            
        Raises:
            Exception: If model downloading fails for any of the specified models
            
        Note:
            This method is a convenience wrapper around check_models_available_offline
            and preload_models. It first checks which models are missing, then
            downloads only those that are needed.
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
        """Preload Granite models to cache.
        
        This method downloads and caches the specified Granite models to ensure they're
        available for offline use. It handles environment variable manipulation to
        ensure proper caching and model downloading.
        
        Args:
            models: List of model names to preload (e.g., ["ibm-granite/granite-speech-3.3-8b"])
            models_dir: Base directory where models should be cached
            
        Raises:
            Exception: If model downloading fails for any of the specified models
            
        Note:
            This method temporarily modifies environment variables for HuggingFace Hub
            operations and restores them upon completion. It also reloads HuggingFace
            modules to ensure fresh configuration.
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
        """Reload HuggingFace modules to ensure fresh configuration for model loading.
        
        This private method removes cached HuggingFace modules from sys.modules to ensure
        fresh configuration when downloading or loading models. This is necessary because
        some HuggingFace modules cache configuration that can interfere with offline mode
        settings and cause issues with model loading.
        
        The method specifically targets:
        - huggingface_hub modules (for repository access)
        - transformers modules (for model architecture and loading)
        
        This approach ensures that subsequent imports of these modules will use the
        most current configuration and environment settings.
        
        Note:
            This method is called automatically during model preloading and should
            not typically be called directly by external code.
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
        """Download a specific model from HuggingFace Hub to the cache directory.
        
        This private method handles the downloading of individual Granite models from
        the HuggingFace Hub to ensure they're available for offline use. It includes
        proper error handling, logging, and progress tracking for the download process.
        
        Args:
            model_name: The full model name to download (e.g., "ibm-granite/granite-speech-3.3-8b")
            cache_dir: The directory where the model should be cached
            
        Raises:
            Exception: If the model download fails for any reason
            
        Note:
            This method temporarily modifies environment variables for HuggingFace Hub
            operations and handles authentication via HF_TOKEN environment variable.
            It also reloads HuggingFace modules to ensure fresh configuration before
            attempting to download the model.
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
        """Resolve the cache directory for storing models.
        
        This method determines the appropriate directory for caching models based on
        environment variables and configuration. It follows the standard HuggingFace
        cache resolution pattern:
        1. Uses XDG_CACHE_HOME if set (typically ~/.cache)
        2. Falls back to ~/.cache/huggingface if XDG_CACHE_HOME is not set
        3. Creates the necessary directory structure if it doesn't exist
        
        Returns:
            pathlib.Path: The resolved cache directory path
            
        Raises:
            OSError: If directory creation fails due to permissions or other filesystem issues
            
        Note:
            This method is used internally by model loading and preloading methods
            to ensure consistent cache directory resolution across all operations.
            The returned path will be in the format:
            - {XDG_CACHE_HOME}/huggingface/hub (if XDG_CACHE_HOME is set)
            - ~/.cache/huggingface/hub (if XDG_CACHE_HOME is not set)
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
        """Load a Granite model from the cache directory.
        
        This method loads a pre-downloaded Granite model from the cache directory
        for use in transcription. It handles proper error checking and logging
        for the model loading process.
        
        Args:
            model_name: The full model name to load (e.g., "ibm-granite/granite-speech-3.3-8b")
            
        Raises:
            Exception: If model loading fails for any reason
            
        Note:
            This method assumes the model has already been downloaded and cached.
            It will fail if the model is not available in the cache directory.
            The method uses the resolved cache directory to locate the model.
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
            self.tokenizer = self.processor.tokenizer
            
            # Load the model
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                local_files_only=True,
                token=token
            )
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"Cache directory exists: {cache_dir.exists()}")
            if cache_dir.exists():
                self.logger.debug(f"Cache directory contents: {list(cache_dir.iterdir())}")
            raise Exception(error_msg)