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
        """
        Initialize the model manager.
        """
        self.logger = logger
        self.models_dir = models_dir
        self.selected_model: Optional[str] = None
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
    
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